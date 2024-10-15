import torch
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer
import argparse
from loading_utils import load_saes, load_examples
from transformer_lens.utils import get_act_name
from typing import Dict
from functools import partial
import numpy as np
from hooks import sae_features_hook, sae_hook
import os

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=128)
parser.add_argument("-w", "--what", type=str, default="faithfulness")
args = parser.parse_args()

os.makedirs("faithfulness", exist_ok=True)

DATASET_SHORT_NAME: Dict[str, str] = {
    "rc_train": "rc",
    "greater_than_examples": "gtthan",
    "greater_than_examples_random_tokens": "dummy",
    "ioi_examples": "ioi",
}
MODEL_SHORT_NAME: Dict[str, str] = {
    "pythia-70m-deduped": "pythia",
    "gpt2": "gpt2",
}

TASK_LENGTHS = {
    "rc_train": 6,
    "greater_than_examples": 12,
    "greater_than_examples_random_tokens": 12,
    "ioi_examples": 15,
}

component_map = {
    "attn": "attn_out",
    "mlp": "mlp_out",
    "resid": "resid_post",
    "embed": "hook_embed",
}


def read_effects(model_name, task):
    attrib_path = f"circuits/attribandig_{MODEL_SHORT_NAME[model_name]}_{DATASET_SHORT_NAME[task]}/{task}_dict10_l01_methodattrib_node0.1_edge0.01_n100_aggnone.pt"
    attrib_effects = torch.load(attrib_path)["nodes"]

    ig_path = f"circuits/attribandig_{MODEL_SHORT_NAME[model_name]}_{DATASET_SHORT_NAME[task]}/{task}_dict10_l01_methodig_node0.1_edge0.01_n100_aggnone.pt"
    ig_effects = torch.load(ig_path)["nodes"]

    # Map the components to the correct names
    for k in list(attrib_effects.keys()):
        if k == "embed":
            new_k = "hook_embed"
        else:
            k_, l = k.split("_")
            new_k = get_act_name(component_map[k_], int(l))
        attrib_effects[new_k] = attrib_effects.pop(k)
        ig_effects[new_k] = ig_effects.pop(k)

    return attrib_effects, ig_effects


def test_circuit(
    tokens,
    clean_answers,
    patch_answers,
    nodes,
    saes,
    node_threshold,
    k,
    what,
    use_resid=False,
    device="cuda",
):

    import torch

    def top_k_mask(x, k):
        _, top_k_indices = torch.topk(x.abs(), k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, top_k_indices, 1)
        
        return mask

    def threshold_mask(x, threshold):
        mask = (x.abs() > threshold).type(torch.int32)
        return mask

    if k is not None:
        get_mask = partial(top_k_mask, k=k)
    else:
        get_mask = partial(threshold_mask, threshold=node_threshold)

    hooks = []
    masks = []

    for hook_name in saes.keys():
        
        #print(nodes[hook_name].act.shape)
        feature_mask = get_mask(nodes[hook_name].act).sum(0) > 0

        hooks.append(
            (
                hook_name,
                partial(
                    sae_features_hook,
                    sae=saes[hook_name],
                    feature_mask=feature_mask,
                    feature_avg=feature_avg[hook_name],
                    resid=use_resid,
                    ablation=what,
                ),
            )
        )

        masks.append(feature_mask.type(torch.int32))

    masks = [(m > 0).sum().item() for m in masks]

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens.to(device),
            fwd_hooks=hooks,
        ).cpu()
        logits = logits[:, -1]

    clean_ans_logits = torch.gather(logits, 1, clean_answers.unsqueeze(1))
    patch_ans_logits = torch.gather(logits, 1, patch_answers.unsqueeze(1))

    return (clean_ans_logits - patch_ans_logits).squeeze(), np.mean(masks)


def faithfulness(
    tokens,
    clean_answers,
    patch_answers,
    nodes,
    dictionaries,
    node_threshold=None,
    k=None,
    use_resid=False,
    device="cuda",
):

    assert node_threshold is not None or k is not None, "Either node_threshold or k must be provided"
    assert not (node_threshold is not None and k is not None), "Only one of node_threshold or k must be provided"

    # Get the model's logit diff - m(M)
    with torch.no_grad():
        logits = model(tokens.to(device)).cpu()
        logits = logits[:, -1]

    clean_ans_logits = torch.gather(logits, 1, clean_answers.unsqueeze(1))
    patch_ans_logits = torch.gather(logits, 1, patch_answers.unsqueeze(1))

    M = (clean_ans_logits - patch_ans_logits).squeeze().mean().item()

    # Get the circuit's logit diff - m(C)
    C, N = test_circuit(
        tokens,
        clean_answers,
        patch_answers,
        nodes,
        dictionaries,
        node_threshold=node_threshold,
        k=k,
        what=args.what,
        use_resid=use_resid,
        device=device,
    )

    # Get the ablated circuit's logit diff - m(zero)
    zero, _ = test_circuit(
        tokens,
        clean_answers,
        patch_answers,
        nodes,
        dictionaries,
        node_threshold=node_threshold,
        k=k,
        what="empty",
        use_resid=use_resid,
        device=device,
    )

    return (C.mean().item() - zero.mean().item()) / (M - zero.mean().item() + 1e-9), N


# Pruning functions
def prune_by_percentile(effects, percentile):
    sorted_effects = {
        k: torch.sort(torch.abs(v.act.sum(dim=-1) + v.resc.squeeze()), descending=True)[
            0
        ]
        for k, v in effects.items()
    }
    nodes_n = int(len(effects.keys()) * percentile)
    top_component_names = sorted(
        sorted_effects.keys(), key=lambda k: sorted_effects[k][0], reverse=True
    )
    return top_component_names[:nodes_n]


##########
## Main ##
##########

model_name = "pythia-70m-deduped"
model = HookedTransformer.from_pretrained(model_name, device="cuda")

nl = model.cfg.n_layers
n = args.n

embed = "hook_embed"
if model_name in ["pythia-70m-deduped", "gpt2"]:
    attns = [get_act_name("attn_out", layer) for layer in range(nl)]
else:
    attns = [get_act_name("z", layer) for layer in range(nl)]
mlps = [get_act_name("mlp_out", layer) for layer in range(nl)]
resids = [get_act_name("resid_post", layer) for layer in range(nl)]

device = "cuda" if torch.cuda.is_available() else "cpu"

dictionaries = load_saes(
    f"dictionaries/{model_name}",
    model_name,
    model.cfg,
    embed,
    attns,
    mlps,
    resids,
    device=device,
)

scores = {
    "task": [],
    "score": [],
    "N": [],
    "components": [],
    "components_fraction": [],
    "feature_coverage": [],
    "sae_threshold": [],
}

for task in list(DATASET_SHORT_NAME.keys()):
    print(f"\n{task.replace('_', ' ').upper()}")

    train_examples = load_examples(
        f"data/{task}.json", 2 * n, model, length=TASK_LENGTHS[task]
    )[:n]
    test_examples = load_examples(
        f"data/{task}.json", 2 * n, model, length=TASK_LENGTHS[task]
    )[n : 2 * n]

    if len(test_examples) < 64:
        test_examples = load_examples(
            f"data/{task}.json", 2 * n, model, length=TASK_LENGTHS[task]
        )

    assert (
        len(test_examples) > 64
    ), f"Not enough examples can be loaded, train length ({len(train_examples)}) test length ({len(test_examples)})"

    train_tokens = torch.cat([e["clean_prefix"] for e in train_examples])

    for COMPONENT_PERCENTILE in tqdm([0, 0.2, 0.4, 0.6, 0.8, 1]):
        for FEATURE_THRESHOLD in [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600]: #  [0, 1, 2, 4, 5, 10, 20]

            attrib_effects, ig_effects = read_effects(model_name, task)
            # Components pruning
            top_component_names = prune_by_percentile(
                attrib_effects, COMPONENT_PERCENTILE
            )
            sum_total = 0
            sum_our_method = 0
            for k in attrib_effects.keys():
                important_saes_for_this_node = (
                    (attrib_effects[k].act.abs() > FEATURE_THRESHOLD).sum().item()
                )
                sum_total += important_saes_for_this_node
                if k in top_component_names:
                    sum_our_method += important_saes_for_this_node

            # Substitute ig effects of pruned components with zeros
            for k in ig_effects.keys():
                if k not in top_component_names:
                    ig_effects[k].act = attrib_effects[k].act

            feature_cache = {}

            hooks = [
                (
                    hook_name,
                    partial(
                        sae_hook,
                        sae=sae,
                        cache=feature_cache,
                    ),
                )
                for hook_name, sae in dictionaries.items()
            ]

            with torch.no_grad():
                model.run_with_hooks(
                    train_tokens.to(device),
                    fwd_hooks=hooks,
                )

            feature_avg = {k: v.mean(0) for k, v in feature_cache.items()}

            test_tokens = torch.cat([e["clean_prefix"] for e in test_examples])
            clean_answers = torch.tensor([e["clean_answer"] for e in test_examples])
            patch_answers = torch.tensor([e["patch_answer"] for e in test_examples])

            score, N = faithfulness(
                test_tokens,
                clean_answers,
                patch_answers,
                ig_effects,
                dictionaries,
                #node_threshold=FEATURE_THRESHOLD,
                k=FEATURE_THRESHOLD,
                use_resid=True,
                device=device,
            )

            print(
                f"Pruned to {len(top_component_names)} components ({COMPONENT_PERCENTILE*100}%). Feature coverage: {100*sum_our_method/(sum_total + 1e-9):.2f}"
            )
            print(f"{task} -- {args.what.capitalize()}: {score:.2f}\tN={N:.1f}\n")

            scores["task"].append(task)
            scores["score"].append(score)
            scores["N"].append(N)
            scores["components"].append(len(top_component_names))
            scores["components_fraction"].append(COMPONENT_PERCENTILE)
            scores["feature_coverage"].append(sum_our_method / (sum_total + 1e-9))
            scores["sae_threshold"].append(FEATURE_THRESHOLD)

score_df = pd.DataFrame(scores)
score_df.to_csv(f"faithfulness/{model_name}_{args.what}.csv", index=False)
