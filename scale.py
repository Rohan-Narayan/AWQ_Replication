import torch
from torch import nn
import gc
from helpers import get_op_name, get_op_by_name
from quant import quantize_tensor
from clip import apply_clip

@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)

@torch.no_grad()
def auto_scale_block(module, module_kwargs, w_bit, q_config, input_feat, s_val=None):
    # TODO - complete
    if w_bit is not None:
        def w_quantize_func(p):
            return quantize_tensor(
                p,
                n_bit=w_bit,
                **q_config,
            ).detach()

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def _search_module_scale(block, linears2scale: list, x, kwargs={}, s_val=None):
        # TODO - complete
        x = x.to(next(block.parameters()).device)
        if s_val is not None:
            scales = torch.full((x.shape[-1],), 2, dtype=x.dtype, device=x.device)

        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}, s_val=None):
        # TODO - complete
        if module2inspect is None:
            # assert len(layers) == 1
            module2inspect = layers[0]
        scales = _search_module_scale(module2inspect, layers, inp, kwargs, s_val)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []

    scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
                s_val=s_val
            )
        )
        # attn out
    scales_list.append(
        _auto_get_scale(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.out_proj],
            inp=input_feat["self_attn.out_proj"],
        )
    )
    # fc1
    scales_list.append(
        _auto_get_scale(
            prev_op=module.final_layer_norm,
            layers=[module.fc1],
            inp=input_feat["fc1"],
        )
    )
    # fc2
    scales_list.append(
        _auto_get_scale(
            prev_op=module.fc1,
            layers=[module.fc2],
            inp=input_feat["fc2"],
        )
    )   
    
    gc.collect()
    torch.cuda.empty_cache()

    return scales_list

# apply_awq
def apply_awq_scaling(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])

def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]
        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()
        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, nn.LayerNorm):
            scale_ln_fcs(prev_op, layers, scales)

        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):

    scales = scales.to(fc1.weight.device)
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


