import torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict
import functools
import gc
from calib import get_calib_dataset
from helpers import get_blocks, move_device, append_str_prefix, get_op_name, get_named_linears
from scale import auto_scale_block
from clip import auto_clip_block, apply_clip

def find_s_and_salient_weights(model, enc, w_bit, q_config, n_samples=512, seqlen=512, mse_range=True, auto_scale=True, calib_data="pileval"):
    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_device(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        next_dev = next(model.parameters()).device
        print(next_dev)
        model(samples.to(next_dev))
        del next_dev
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_device(model, "cpu")

    gc.collect()

    awq_results = {
        "scale": [],
        "clip": [],
    }
    torch.cuda.empty_cache()

    # solve layer by layer
    for i in tqdm(range(len(layers)), desc="Running AWQ..."):
        gc.collect()
        torch.cuda.empty_cache()
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
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
        # inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        del handles
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        del scales_list
        gc.collect()
        torch.cuda.empty_cache()

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cpu()
        del layer
        # Haotian: check activation replacement
        del input_feat
        del clip_list
        del named_linears
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results
    # TODO