import os
import json
import random
from tqdm import tqdm
import numpy as np
import torch as t
import torch.nn.functional as F

# from dataclasses import dataclass
from utils import BASE_DIR, IdentityDict
from sae_lens import SAE

def load_examples(
    dataset, num_examples, model, seed=12, pad_to_length=None, length=None
):
    examples = []
    dataset_items = open(os.path.join(BASE_DIR, dataset)).readlines()
    random.seed(seed)
    random.shuffle(dataset_items)
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = model.tokenizer(
            data["clean_prefix"], return_tensors="pt", padding=False, add_special_tokens=False
        ).input_ids
        patch_prefix = model.tokenizer(
            data["patch_prefix"], return_tensors="pt", padding=False, add_special_tokens=False
        ).input_ids
        clean_answer = model.tokenizer(
            data["clean_answer"],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids
        patch_answer = model.tokenizer(
            data["patch_answer"],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids
        # only keep examples where answers are single tokens
        if clean_prefix.shape[1] != patch_prefix.shape[1]:
            continue
        # only keep examples where clean and patch inputs are the same length
        if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
            continue
        # if we specify a `length`, filter examples if they don't match
        if length and clean_prefix.shape[1] != length:
            continue
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            model.tokenizer.padding_side = "right"
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(
                F.pad(
                    t.flip(clean_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )
            patch_prefix = t.flip(
                F.pad(
                    t.flip(patch_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )

        example_dict = {
            "clean_prefix": clean_prefix,
            "patch_prefix": patch_prefix,
            "clean_answer": clean_answer.item(),
            "patch_answer": patch_answer.item(),
            "prefix_length_wo_pad": prefix_length_wo_pad,
        }
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples

def load_examples_nopair(dataset, num_examples, model, length=None, pad_to_length=False, seed=42):
    
    examples = []
    dataset_items = open(os.path.join(BASE_DIR, dataset)).readlines()
    random.seed(seed)
    random.shuffle(dataset_items)
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = model.tokenizer(
            data["context"], return_tensors="pt", padding=False, add_special_tokens=False
        ).input_ids
        
        clean_answer = model.tokenizer(
            data["answer"],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids
        
        # if we specify a `length`, filter examples if they don't match
        if length and clean_prefix.shape[1] != length:
            continue
        
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        
        if pad_to_length:
            model.tokenizer.padding_side = "right"
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(
                F.pad(
                    t.flip(clean_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )

        example_dict = {
            "clean_prefix": clean_prefix,
            "clean_answer": clean_answer.item(),
            "prefix_length_wo_pad": prefix_length_wo_pad,
        }
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples

def get_sae_path(dict_path, model_name, component, layer=None, sparsity=None):
    if model_name in ["pythia-70m-deduped", "gpt2"]:
        if component == "embed":
            return f"{dict_path}/embed"
        else:
            return f"{dict_path}/{component}_layer{layer}"
    elif model_name == "1l-arithmetics":
        assert sparsity is not None, "Sparsity must be specified for 1l-arithmetics"
        if component == "embed":
            return f"{dict_path}/l1_{sparsity}/embed"
        else:
            return f"{dict_path}/l1_{sparsity}/{component}_layer{layer}"
    elif model_name == "gemma-2-2b":
        if component == "embed":
            return f"{dict_path}/embed/params.npz"
        else:
            return f"{dict_path}/{component}_layer{layer}/params.npz"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_gemma_2_sae(sae_path, layer, hook_name, device="cuda"):
    
    mapping = {
        "embed": "embed",
        "attn_out": f"blocks.{layer}.attn.hook_z",
        "mlp_out": f"blocks.{layer}.hook_mlp_out",
        "resid_out": f"blocks.{layer}.hook_resid_post",
    }

    cfg_dict = {
        "architecture": "jumprelu",
        "d_in": 2304 if hook_name != "attn_out" else 2048,
        "d_sae": 16384 if hook_name != "embed" else 4096,
        "dtype": "float32",
        "model_name": "gemma-2-2b",
        "hook_name": mapping[hook_name],
        "hook_layer": layer,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
        "device": device,
    }
    
    state_dict = {}
    with np.load(sae_path) as data:
        for key in data.keys():
            state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[state_dict_key] = (
                t.tensor(data[key]).to(dtype=t.float32).to(device)
            )

    # Handle scaling factor
    if "scaling_factor" in state_dict:
        if t.allclose(
            state_dict["scaling_factor"], t.ones_like(state_dict["scaling_factor"])
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict.pop("scaling_factor")
    else:
        cfg_dict["finetuning_scaling_factor"] = False

    sae = SAE.from_dict(cfg_dict)
    sae.load_state_dict(state_dict)
    sae.turn_off_forward_pass_hook_z_reshaping()

    return sae

def load_sae(path, model_name, layer, hook_name, device="cuda"):

    if model_name in ["pythia-70m-deduped", "gpt2"]:
        sae = SAE.load_from_pretrained(
            path, device=device, dtype="float32"
        )
    elif model_name == "gemma-2-2b":
        sae = load_gemma_2_sae(path, layer, hook_name, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return sae


def load_saes(
    dict_path, model_name, cfg, embed, attns, mlps, resids, sparsity=None, device="cuda", verbose=False
):

    dictionaries = {}

    if dict_path is None:
        dictionaries[embed] = IdentityDict(cfg.d_model)

        for i in range(cfg.n_layers):
            dictionaries[attns[i]] = IdentityDict(cfg.n_heads * cfg.d_head)
            dictionaries[mlps[i]] = IdentityDict(cfg.d_model)
            dictionaries[resids[i]] = IdentityDict(cfg.d_model)
    else:
        embed_path = get_sae_path(dict_path, model_name, "embed", sparsity=sparsity)
        if os.path.exists(embed_path):
            dictionaries[embed] = load_sae(embed_path, model_name, None, "embed", device=device)
        else:
            dictionaries[embed] = IdentityDict(cfg.d_model)

        if verbose:
            bar = tqdm(range(cfg.n_layers))
        else:
            bar = range(cfg.n_layers)
        for i in bar:
            attn_path = get_sae_path(
                dict_path, model_name, "attn_out", i, sparsity=sparsity
            )
            if os.path.exists(attn_path):
                dictionaries[attns[i]] = load_sae(attn_path, model_name, i, "attn_out", device=device)
            else:
                print(f"Could not find {attn_path}")
                dictionaries[attns[i]] = IdentityDict(cfg.d_model)

            mlp_path = get_sae_path(
                dict_path, model_name, "mlp_out", i, sparsity=sparsity
            )
            if os.path.exists(mlp_path):
                dictionaries[mlps[i]] = load_sae(mlp_path, model_name, i, "mlp_out", device=device)
            else:
                print(f"Could not find {mlp_path}")
                dictionaries[mlps[i]] = IdentityDict(cfg.d_model)

            resid_path = get_sae_path(
                dict_path, model_name, "resid_out", i, sparsity=sparsity
            )
            if os.path.exists(resid_path):
                dictionaries[resids[i]] = load_sae(resid_path, model_name, i, "resid_out", device=device)
            else:
                print(f"Could not find {resid_path}")
                dictionaries[resids[i]] = IdentityDict(cfg.d_model)

    return dictionaries
