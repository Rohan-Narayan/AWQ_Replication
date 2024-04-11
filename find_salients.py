import torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict
import functools
import gc
from calib import get_calib_dataset
from helpers import get_blocks, move_device, append_str_prefix, get_op_name, get_named_linears
from scale import auto_scale_block, apply_scale
from clip import auto_clip_block, apply_clip


@torch.no_grad()
def find_s_and_salient_weights(model, enc, group_size, s_val=None):
    num_bits = 3
    n_samples = 128
    seqlen = 512

    layers = get_blocks(model)

    calib_data="pileval"
    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_device(model, "cuda")

    # Hack to get input and kwargs - taken from official AWQ Repo
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError 

    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:
        pass

    # Can get rid of sample - just need activations in model
    del samples

    layers[0] = layers[0].module
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_device(model, "cpu")

    gc.collect()

    s_and_salient_weights = {
        "scale": [],
        "clip": [],
    }
    torch.cuda.empty_cache()

    for i in tqdm(range(len(layers)), desc="Running AWQ..."):
        # Clear GPU Memory
        gc.collect()
        torch.cuda.empty_cache()
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        del handles

        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        scales_list = auto_scale_block(
            layer,
            layer_kwargs,
            num_bits=num_bits,
            group_size=group_size,
            input_feat=input_feat,
            s_val=s_val
        )

        apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

        s_and_salient_weights["scale"] += append_str_prefix(
            scales_list, get_op_name(model, layer) + "."
        )

        # Clear GPU memory
        del scales_list
        gc.collect()
        torch.cuda.empty_cache()

        # Apply weight clipping to reduce MSE
        # Reduces quantization error in delta
        clip_list = auto_clip_block(
            layer,
            num_bits=num_bits,
            group_size=group_size,
            input_feat=input_feat,
        )
        apply_clip(layer, clip_list)

        s_and_salient_weights["clip"] += append_str_prefix(
            clip_list, get_op_name(model, layer) + "."
        )

        layer = layer.cpu()

        # Clear GPU Memory
        del layer
        del input_feat
        del clip_list
        del named_linears
        gc.collect()
        torch.cuda.empty_cache()

    return s_and_salient_weights
