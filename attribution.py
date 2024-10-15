from collections import namedtuple
import torch as t
from tqdm import tqdm
from numpy import ndindex
from activation_utils import SparseAct
from functools import partial
import gc
import os
from hooks import sae_hook, sae_ig_patching_hook, component_level_hook

DEBUGGING = True

if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan' : True}
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
else:
    tracer_kwargs = {'validate' : False, 'scan' : False}

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def print_gpu_memory():
    if t.cuda.is_available():
        print(f"GPU memory allocated: {t.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {t.cuda.memory_reserved() / 1e9:.2f} GB")

def _pe_attrib(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
):
    
    hidden_states_clean = {}
    grads = {}

    sae_hooks = []

    for i, submodule in enumerate(submodules):
        dictionary = dictionaries[submodule]
        sae_hooks.append((submodule, partial(sae_hook, sae=dictionary, cache=hidden_states_clean)))

    # Forward pass with hooks
    logits = model.run_with_hooks(clean, fwd_hooks=sae_hooks)
    metric_clean = metric_fn(logits, **metric_kwargs)

    # Backward pass
    metric_clean.sum().backward()

    # Collect gradients
    for submodule in submodules:
        if submodule in hidden_states_clean:
            grads[submodule] = hidden_states_clean[submodule].grad
            grads[submodule].res = t.zeros_like(hidden_states_clean[submodule].res)

    #hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    #grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        sae_hooks = []
        for i, submodule in enumerate(submodules):
            dictionary = dictionaries[submodule]
            sae_hooks.append((submodule, partial(sae_hook, sae=dictionary, cache=hidden_states_patch)))
        
        with t.no_grad():
            corr_logits = model.run_with_hooks(patch, fwd_hooks=sae_hooks)
        metric_patch = metric_fn(corr_logits, **metric_kwargs)
        total_effect = (metric_patch - metric_clean).detach()
        #hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state, grad = hidden_states_patch[submodule], hidden_states_clean[submodule], grads[submodule]
        delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None

    del hidden_states_clean, hidden_states_patch
    gc.collect()
    
    return EffectOut(effects, deltas, grads, total_effect)

def _pe_attrib_ig(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
        top_k=10,
        debug=False,
        metric_kwargs=dict(),
    ):
    #first run through and calculate your effects on attribution 
    effects, deltas, grads, total_effect = _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    #filter out the components which have a greater IE
    del deltas, grads, total_effect
    effects = {k : t.sort(t.abs(v.act.sum(dim=-1).mean(dim=0) + v.resc.squeeze().mean(dim=0)), descending=True)[0] for k, v in effects.items()}
    top_component_names = sorted(effects.keys(), key= lambda k: effects[k][0], reverse=True)[:top_k]
    #rerun the attribution with the components that you have filtered
    return _pe_ig(clean, patch, model, top_component_names, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs)

def _pe_ig(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
        metric_kwargs=dict(),
        submodules_to_run=None
):
    print(len(submodules))
    hidden_states_clean = {}
    sae_hooks = []

    # Forward pass through the clean input with hooks to capture hidden states
    for i, submodule in enumerate(submodules):
        dictionary = dictionaries[submodule]
        sae_hooks.append((submodule, partial(sae_hook, sae=dictionary, cache=hidden_states_clean)))
    
    # First pass to get clean logits and metric
    logits_clean = model.run_with_hooks(clean, fwd_hooks=sae_hooks)
    metric_clean = metric_fn(logits_clean, **metric_kwargs)

    if patch is None:
        hidden_states_patch = {
            k: SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        sae_hooks_patch = []
        for i, submodule in enumerate(submodules):
            dictionary = dictionaries[submodule]
            sae_hooks_patch.append((submodule, partial(sae_hook, sae=dictionary, cache=hidden_states_patch)))
        
        with t.no_grad():
            logits_patch = model.run_with_hooks(patch, fwd_hooks=sae_hooks_patch)
        metric_patch = metric_fn(logits_patch, **metric_kwargs)
        total_effect = (metric_patch - metric_clean).detach()

    # Integrated gradients computation
    grads = {}
    effects = {}
    deltas = {}
    
    submodules_to_compute_ig = submodules_to_run if submodules_to_run is not None else hidden_states_clean.keys()
    for submodule in submodules_to_compute_ig:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule].detach()
        patch_state = hidden_states_patch[submodule].detach() if patch is not None else None
        delta = (patch_state - clean_state.detach()) if patch_state is not None else -clean_state.detach()

        for step in range(steps + 1):
            interpolated_state_cache = {}
            alpha = step / steps
            interpolated_state = clean_state * (1 - alpha) + patch_state * alpha if patch is not None else clean_state * (1 - alpha)

            interpolated_state.act.requires_grad_(True)
            interpolated_state.act.retain_grad()
            interpolated_state.res.requires_grad_(True)
            interpolated_state.res.retain_grad()

            sae_hook_ = [(submodule, partial(sae_ig_patching_hook, sae=dictionary, patch=interpolated_state, cache=interpolated_state_cache))]
            
            # Forward pass with hooks
            logits_interpolated = model.run_with_hooks(clean, fwd_hooks=sae_hook_)
            metric = metric_fn(logits_interpolated, **metric_kwargs)

            # Sum the metrics and backpropagate        
            metric.sum().backward(retain_graph=True)
            
            if submodule not in grads:
                grads[submodule] = interpolated_state_cache[submodule].grad.clone()
            else:
                grads[submodule] += interpolated_state_cache[submodule].grad

            if step % (steps // 5) == 0:  # Print every 20% of steps           
                del interpolated_state_cache
                t.cuda.empty_cache()
            
            model.zero_grad(set_to_none=True)

        # Calculate gradients
        grads[submodule] /= steps

        # Compute effects
        effect = grads[submodule] @ delta
        effects[submodule] = effect
        deltas[submodule] = delta

    #assert that for sparseAct, all res, resc and act are the same shape
    for submodule, effect in effects.items():
        assert effect.resc.shape == next(iter(effects.values())).resc.shape, f"resc shape mismatch in {submodule}"
    print(len(submodules))
    for submodule in submodules:
        if submodule not in effects.keys():
            print(f"Submodule {submodule} not in effects")
            effects[submodule] = SparseAct(act=t.zeros_like(hidden_states_clean[submodule].act, device=next(iter(effects.values())).act.device), resc=t.zeros_like(next(iter(effects.values())).resc, device=next(iter(effects.values())).resc.device))
    return EffectOut(effects, deltas, grads, total_effect)

def _pe_exact(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn,
    component_level=False,
    metric_kwargs=dict()
    ):
    if not component_level:
        raise NotImplementedError("Component level attributions only implemented for exact method")
    
    hidden_states_clean = {}
    
    sae_hooks = []
    for i, submodule in enumerate(submodules):
        sae_hooks.append((submodule, partial(component_level_hook, cache=hidden_states_clean)))
    with t.no_grad():
        logits = model.run_with_hooks(clean, fwd_hooks=sae_hooks)
    metric_clean = metric_fn(logits, **metric_kwargs)
    
    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        sae_hooks = []
        for i, submodule in enumerate(submodules):
            sae_hooks.append((submodule, partial(component_level_hook, cache=hidden_states_patch)))
        
        with t.no_grad():
            corr_logits = model.run_with_hooks(patch, fwd_hooks=sae_hooks)
        metric_patch = metric_fn(corr_logits, **metric_kwargs)
        total_effect = (metric_patch - metric_clean).detach()
    
    effects = {}
    deltas = {}
    for submodule in submodules:
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
                
        def component_level_patching_hook(act, hook, token_index, patched_output):
            modified_act = act.clone()
            modified_act[token_index] = patched_output
            return modified_act
        
        
        effect = SparseAct(act=t.zeros(*clean_state.act.shape[:2]), resc=t.zeros(*clean_state.res.shape[:2])).to("cuda" if t.cuda.is_available() else "cpu")
        for idx in tqdm(list(ndindex(effect.act.shape))): #the actual index of the phrase
            sae_hooks = []
            sae_hooks.append((submodule, partial(component_level_patching_hook, token_index=idx, patched_output=patch_state.act[idx])))
            with t.no_grad():
                patched_logits = model.run_with_hooks(clean, fwd_hooks=sae_hooks)
            metric = metric_fn(patched_logits, **metric_kwargs)
            effect.act[tuple(idx)]  = (metric - metric_clean).detach().sum()
            """
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    act = clean_state.act.clone()
                    act[tuple(idx)] = patch_state.act[tuple(idx)]
                    if is_tuple[submodule]:
                        submodule.output[0][:] = act
                    else:
                        submodule.output = act
                    metric = metric_fn(model).save()
                effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()
                """

        effects[submodule] = effect
        deltas[submodule] = patch_state - clean_state
    total_effect = total_effect if total_effect is not None else None

    for k in effects.keys():
        print(k)
        print(effects[k].act.shape)
    for k in deltas.keys():
        print(k)
        print(deltas[k].act.shape)
    return EffectOut(effects, deltas, None, total_effect)


def patching_effect(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='attrib',
        steps=10,
        metric_kwargs=dict(),
        component_level=False,
        submodules_to_run=None
):
    print("Length of submodules", len(submodules))
    assert (component_level and method == 'exact') or not component_level, "Component level attributions only implemented for exact method"
    if method == 'attrib':
        return _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs, submodules_to_run=submodules_to_run)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn, component_level=component_level)
    else:
        raise ValueError(f"Unknown method {method}")