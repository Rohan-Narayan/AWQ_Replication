import torch
from tqdm import tqdm
import torch
from helpers import set_op_by_name, get_blocks, get_named_linears
from awq_layer import AWQLinear
import gc


def quantize(model, num_bits, group_size):
    # TODO - complete
    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="Quantizing model",
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        for name, module in named_linears.items():
            module.cuda()
            module.weight.data, scales, zeros = quantize_tensor(
                module.weight.data, num_bits=num_bits, group_size=group_size, get_scale_zp=True
            )
            q_linear = AWQLinear.from_linear(
                module, num_bits, group_size, scales, zeros
            )
            module.cpu()
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
            torch.cuda.empty_cache()
            gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

def quantize_tensor(
    w, num_bits, group_size, get_scale_zp=False
):
    original_w_shape = w.shape

    w = w.reshape(-1, group_size)
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**num_bits - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales

    w = w.reshape(original_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
    
