import argparse
import gc
import math
import os
import logging

import torch as t
from tqdm import tqdm

from attribution import patching_effect

from loading_utils import load_examples, load_examples_nopair, load_saes
from utils import BASE_DIR, keys

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

os.environ["HF_TOKEN"] = keys["huggingface"]
os.environ["HF_HOME"] = "/workspace/huggingface"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_circuit(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum',  # or 'none' for not aggregating across sequence position
        nodes_only=False,
        node_threshold=0.1,
        edge_threshold=0.01,
        method='attrib',
        component_level=False,
        submodules_to_run=None
):
    logger.info("Starting get_circuit function.")
    all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    print("Length of submodules", len(all_submods))
    # First get the patching effect of everything on y
    logger.info("Computing patching effects.")
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method=method,  # get better approximations for early layers by using integrated gradients (ig)
        component_level=component_level,
        submodules_to_run=submodules_to_run
    )
        
    features_by_submod = {}
    
    for submod in all_submods:
        # Convert to tensor and apply threshold
        feature_tensor = effects[submod].to_tensor().flatten().abs()
        
        selected_features = (feature_tensor > node_threshold).nonzero().flatten()
        
        if len(selected_features) > 0:
            # Sort features by their absolute values in descending order and take the top M
            sorted_features = selected_features[feature_tensor[selected_features].argsort(descending=True)]
            constrained_features = sorted_features[:args.max_features].tolist()
        else:
            # If no features exceed the threshold, return an empty list
            constrained_features = []

        features_by_submod[submod] = constrained_features
    
    n_layers = len(resids)

    logger.info("Initializing nodes dictionary.")
    nodes = {'y': total_effect}
    nodes['embed'] = effects[embed]
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    if aggregation == 'sum':
        for k in nodes:
            if k != 'y':
                nodes[k] = nodes[k].sum(dim=1)
    
    for k, v in nodes.items():
        if k != "y":
            nodes[k] = v.mean(dim=0)

    return nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='rc_train',
                        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.")
    parser.add_argument('--num_examples', '-n', type=int, default=192,
                        help="The number of examples from the --dataset over which to average indirect effects.")
    parser.add_argument('--example_length', '-l', type=int, default=None,
                        help="The max length (if using sum aggregation) or exact length (if not aggregating) of examples.")
    parser.add_argument('--model', type=str, default='gpt2',
                        help="The Huggingface ID of the model you wish to test.")
    parser.add_argument("--dict_path", type=str, default="dictionaries",
                        help="Path to all dictionaries for your language model.")
    parser.add_argument('--dict_id', type=str, default=10,
                        help="ID of the dictionaries. Use `id` to obtain circuits on neurons/heads directly.")
    parser.add_argument('--dict_size', type=int, default=32768,
                        help="The width of the dictionary encoder.")
    parser.add_argument('--sparsity', type=int, default=1,
                        help="Sparsity of the dictionary (l1 coefficient of the SAEs).")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Number of examples to process at once when running circuit discovery.")
    parser.add_argument('--method', type=str, default='attrib',
                        help="Method for computing indirect effects. Should be one of `exact`, `attrib` or `ig`.")
    parser.add_argument('--aggregation', type=str, default='none',
                        help="Aggregation across token positions. Should be one of `sum` or `none`.")
    parser.add_argument('--node_threshold', type=float, default=0.08,
                        help="Indirect effect threshold for keeping circuit nodes.")
    parser.add_argument('--edge_threshold', type=float, default=0.01,
                        help="Indirect effect threshold for keeping edges.")
    parser.add_argument('--max_features', type=int, default=1000,
                        help="Maximum number of features to retain per submodule.")
    parser.add_argument('--top_k_modules', type=int, default=20,
                        help="Number of model components to consider when filtering.")
    parser.add_argument('--pen_thickness', type=float, default=1,
                        help="Scales the width of the edges in the circuit plot.")
    parser.add_argument('--nopair', default=False, action="store_true",
                        help="Use if your data does not contain contrastive (minimal) pairs.")
    parser.add_argument('--debug', default=False, action='store_true',)
    parser.add_argument('--plot_circuit', default=False, action='store_true',
                        help="Plot the circuit after discovering it.")
    parser.add_argument('--nodes_only', default=False, action='store_true',
                        help="Only search for causally implicated features; do not draw edges.")
    parser.add_argument('--plot_only', action="store_true",
                        help="Do not run circuit discovery; just plot an existing circuit.")
    parser.add_argument("--circuit_dir", type=str, default="circuits",
                        help="Directory to save/load circuits.")
    parser.add_argument("--plot_dir", type=str, default="circuits/figures",
                        help="Directory to save figures.")
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--component_level', default=False, action='store_true',
                        help="Use if you want to run circuit discovery on a component level.")
    args = parser.parse_args()
    
    dict_path = os.path.join(args.dict_path, args.model) if args.dict_id != 'id' else None

    dict_path = os.path.join(BASE_DIR, dict_path)
    circuit_dir = os.path.join(BASE_DIR, args.circuit_dir)
    plot_dir = os.path.join(BASE_DIR, args.plot_dir)
    annotation_dir = os.path.join(BASE_DIR, 'annotations')

    os.makedirs(circuit_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    device = args.device
    model = HookedTransformer.from_pretrained(args.model, device=device)
    if args.model in ["pythia-70m-deduped", "gpt2"]:
        model.tokenizer.padding_side = "left"
    
    nl = model.cfg.n_layers
    nh = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    embed = "hook_embed"
    if args.model in ["pythia-70m-deduped", "gpt2"]:
        attns = [get_act_name("attn_out", layer) for layer in range(nl)]
    else:
        attns = [get_act_name("z", layer) for layer in range(nl)]
    mlps = [get_act_name("mlp_out", layer) for layer in range(nl)]
    resids = [get_act_name("resid_post", layer) for layer in range(nl)]

    dictionaries = None
    #loading saes
    if not args.component_level and not args.plot_only:
        # SAE loading
        dictionaries = load_saes(dict_path, args.model, model.cfg, embed, attns, mlps, resids, args.sparsity, device=device)
        logger.info("Dictionaries loaded.")
    
    #loading examples
    data_path = f"data/{args.dataset}.json"
    if args.nopair:
        save_basename = os.path.splitext(os.path.basename(args.dataset))[0]
        examples = load_examples_nopair(data_path, args.num_examples, model, length=args.example_length)
    else:
        save_basename = args.dataset

        if args.aggregation == "sum":
            examples = load_examples(data_path, args.num_examples, model, pad_to_length=args.example_length)
        else:
            examples = load_examples(data_path, args.num_examples, model, length=args.example_length)

    batch_size = args.batch_size
    num_examples = min([args.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < args.num_examples: # warn the user
        logger.warning(f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead.")

    running_nodes = None

    if args.method == 'filtered_ig':
        
        #get top component names using attrib
        for batch in tqdm(batches, desc="Batches"):
                
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
            clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

            if not args.example_length:
                args.example_length = clean_inputs.shape[1]

            if args.nopair:
                patch_inputs = None
                def metric_fn(logits):
                    return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(logits[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
            else:
                patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
                def metric_fn(logits):
                    return (
                        t.gather(logits[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                        t.gather(logits[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                    )
            
            nodes = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                method='attrib',
                component_level=args.component_level
            )

            if running_nodes is None:
                running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            else:
                for k in nodes.keys():
                    if k != 'y':
                        running_nodes[k] += len(batch) * nodes[k].to('cpu')
            
            # memory cleanup
            del nodes
            gc.collect()
        
        nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()} 
        effects = {k : t.sort(t.abs(v.act.sum(dim=-1) + v.resc.squeeze()), descending=True)[0] for k, v in nodes.items()}
        print(effects)
        top_component_names = sorted(effects.keys(), key= lambda k: effects[k][0], reverse=True)[:args.top_k_modules]
        print(top_component_names)
        
        #replace top component names with node names
        if 'embed' in top_component_names:
            top_component_names.remove('embed')
            top_component_names.append(embed)
        for i in range(len(resids)):
            if f'resid_{i}' in top_component_names:
                top_component_names.remove(f'resid_{i}')
                top_component_names.append(resids[i])
            if f'attn_{i}' in top_component_names:
                top_component_names.remove(f'attn_{i}')
                top_component_names.append(attns[i])
            if f'mlp_{i}' in top_component_names:
                top_component_names.remove(f'mlp_{i}')
                top_component_names.append(mlps[i])
        print(top_component_names)
        del nodes,effects,running_nodes
        
        running_nodes = None
        
        #run filtered ig
        
        for batch in tqdm(batches, desc="Batches"):
                
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
            clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

            if not args.example_length:
                args.example_length = clean_inputs.shape[1]

            if args.nopair:
                patch_inputs = None
                def metric_fn(logits):
                    return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(logits[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
            else:
                patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
                def metric_fn(logits):
                    return (
                        t.gather(logits[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                        t.gather(logits[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                    )
            
            nodes = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                method='ig',
                component_level=args.component_level,
                submodules_to_run=top_component_names
            )

            if running_nodes is None:
                running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            else:
                for k in nodes.keys():
                    if k != 'y':
                        running_nodes[k] += len(batch) * nodes[k].to('cpu')
            print("LEN RUNNING NODES AFTER IG BATCH", len(running_nodes.keys()))
            
            # memory cleanup
            del nodes
            gc.collect()

        
        nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
        
        save_dict = {
            "examples" : examples,
            "nodes": nodes
        }
        print("LEN NODES", save_dict['nodes'].keys())
        save_path = f'{circuit_dir}/{save_basename}_dict{args.dict_id}_l0{args.sparsity}_method{args.method}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt'
        with open(save_path, 'wb') as outfile:
            t.save(save_dict, outfile)
        logger.info(f"Results saved to {save_path}")
        
        if args.debug:
            
            logger.info("Starting debug mode: counting nodes")
            
            nodes_count = 0
            for k in nodes.keys():
                nodes_count += (nodes[k].act > args.node_threshold).sum().item()
                print("Nodes count in this k", (nodes[k].act > args.node_threshold).sum().item())
            print("NODES COUNT", nodes_count)
                
            running_nodes = None
            
            for batch in tqdm(batches, desc="Batches"):
                    
                clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
                clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

                if not args.example_length:
                    args.example_length = clean_inputs.shape[1]

                if args.nopair:
                    patch_inputs = None
                    def metric_fn(logits):
                        return (
                            -1 * t.gather(
                                t.nn.functional.log_softmax(logits[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                            ).squeeze(-1)
                        )
                else:
                    patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                    patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
                    def metric_fn(logits):
                        return (
                            t.gather(logits[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                            t.gather(logits[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                        )
                
                nodes = get_circuit(
                    clean_inputs,
                    patch_inputs,
                    model,
                    embed,
                    attns,
                    mlps,
                    resids,
                    dictionaries,
                    metric_fn,
                    nodes_only=args.nodes_only,
                    aggregation=args.aggregation,
                    node_threshold=args.node_threshold,
                    edge_threshold=args.edge_threshold,
                    method='ig',
                    component_level=args.component_level
                )

                if running_nodes is None:
                    running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
                else:
                    for k in nodes.keys():
                        if k != 'y':
                            running_nodes[k] += len(batch) * nodes[k].to('cpu')
                
                # memory cleanup
                del nodes
                gc.collect()

            nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
            ig_nodes_count = 0
            for k in nodes.keys():
                ig_nodes_count += (nodes[k].act > args.node_threshold).sum().item()
                print("Nodes count in this k", (nodes[k].act > args.node_threshold).sum().item())
            print("IG NODES COUNT", ig_nodes_count)
            
            number_of_components = len(nodes.keys())
            number_of_filtered_components = len(top_component_names)
            save_path = f'{circuit_dir}/{save_basename}_dict{args.dict_id}_l0{args.sparsity}_method{args.method}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.txt'
            with open(save_path, 'w') as outfile:
                outfile.write(f"Number of components: {number_of_components}\n")
                outfile.write(f"Number of filtered components: {number_of_filtered_components}\n")
                outfile.write(f"Nodes count: {nodes_count}\n")
                outfile.write(f"IG nodes count: {ig_nodes_count}\n")
                outfile.write(f"Arguments:\n")
                for arg in vars(args):
                    outfile.write(f"{arg}: {getattr(args, arg)}\n")
    else:

        for batch in tqdm(batches, desc="Batches"):
                
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
            clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

            if not args.example_length:
                args.example_length = clean_inputs.shape[1]

            if args.nopair:
                patch_inputs = None
                def metric_fn(logits):
                    return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(logits[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
            else:
                patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
                def metric_fn(logits):
                    return (
                        t.gather(logits[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                        t.gather(logits[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                    )
            
            nodes = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                method=args.method,
                component_level=args.component_level
            )

            if running_nodes is None:
                running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            else:
                for k in nodes.keys():
                    if k != 'y':
                        running_nodes[k] += len(batch) * nodes[k].to('cpu')
            
            # memory cleanup
            del nodes
            gc.collect()

        nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}

        save_dict = {
            "examples" : examples,
            "nodes": nodes
        }
        print("LEN NODES", save_dict['nodes'].keys())
        save_path = f'{circuit_dir}/{save_basename}_dict{args.dict_id}_l0{args.sparsity}_method{args.method}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt'
        with open(save_path, 'wb') as outfile:
            t.save(save_dict, outfile)
        logger.info(f"Results saved to {save_path}")