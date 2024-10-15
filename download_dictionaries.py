import os
import requests
import re
import zipfile
from tqdm import tqdm
import argparse
from safetensors.torch import save_file
import torch
import json

from utils import keys

os.environ["HF_TOKEN"] = keys["huggingface"]

def download_file(url, dest_path):
    # Streaming download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses

    total_size = int(response.headers.get('content-length', 0))  # Total size in bytes
    block_size = 8192  # Block size for reading (8 KB)

    if dest_path.endswith('.zip'):
        # If the destination is a zip file
        with open(dest_path, "wb") as file:
            for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB', unit_scale=True, desc="Downloading"):
                file.write(data)

        # Unzip with progress bar
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())
            with tqdm(total=total_files, unit='file', desc="Unzipping") as pbar:
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info)
                    pbar.update(1)

        # Remove the zip file after extraction
        os.remove(dest_path)
    else:
        # If the destination is a regular file
        with open(dest_path, 'wb') as file:
            for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB', unit_scale=True, desc="Downloading"):
                file.write(data)

def find_closest_folder(base_url, layer, width, target=100):
    # Fetch the directory listing
    url = f"{base_url}/layer_{layer}/{width}"
    response = requests.get(url)
    response.raise_for_status()

    # Parse the directory listing to find the closest `average_l0_n` folder
    folders = re.findall(r'average_l0_\d+', response.text)
    closest_folder = min(folders, key=lambda x: abs(int(x.split('_')[-1]) - target))

    return closest_folder

def convert_to_sae_config(raw_config, path):
    # Mapping for specific keys
    mapping = {
        "activation_dim": "d_in",
        "dictionary_size": "d_sae",
        "ctx_len": "context_size",
    }
    
    # Apply the mapping to the raw_config
    path = path.split('/')[-1]
    converted_config = {mapping.get(k, k): v for k, v in raw_config.items()}

    # Add missing keys
    component = '_'.join(path.split('_')[:2])
    layer = path[-1] if 'layer' in path else 0

    missing_keys = {
        "model_name": "pythia-70m-deduped",
        "hook_name": f"blocks.{layer}.hook_{component}",
        "hook_layer": layer,
        "hook_head_index": None,
        "dataset_path": None,
    }

    for k, v in missing_keys.items():
        if k not in converted_config:
            converted_config[k] = v
    
    return converted_config


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="Model name", required=True)
args = parser.parse_args()

if not os.path.isdir('dictionaries'):
    os.makedirs('dictionaries')

# Base URLs
GEMMA_BASE_URL_RESID = "https://huggingface.co/google/gemma-scope-2b-pt-res/resolve/main"
GEMMA_BASE_URL_ATTN = "https://huggingface.co/google/gemma-scope-2b-pt-att/resolve/main"
GEMMA_BASE_URL_MLP = "https://huggingface.co/google/gemma-scope-2b-pt-mlp/resolve/main"

PYTHIA_BASE_URL = "https://huggingface.co/saprmarks/pythia-70m-deduped-saes/resolve/main/dictionaries_pythia-70m-deduped_10.zip"
PYTHIA_FILE = "dictionaries_pythia-70m-deduped_10.zip"

GPT_RESID_URL = "https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs/resolve/main"
GPT_ATTN_URL = "https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs/resolve/main"
GPT_MLP_URL = "https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs/resolve/main"

def download_gemma_2():

    # Download embed dictionaries
    os.makedirs(f"dictionaries/gemma-2-2b/embed", exist_ok=True)
    embed_url = f"{GEMMA_BASE_URL_RESID}/embedding/width_4k/average_l0_111/params.npz"
    embed_dest = f"dictionaries/gemma-2-2b/embed/params.npz"
    download_file(embed_url, embed_dest)

    for i in tqdm(range(26)):
        RESID_DIR = "dictionaries/gemma-2-2b/resid_out_layer"
        ATTN_DIR = "dictionaries/gemma-2-2b/attn_out_layer"
        MLP_DIR = "dictionaries/gemma-2-2b/mlp_out_layer"

        # Create directories if they don't exist
        os.makedirs(f"{RESID_DIR}{i}", exist_ok=True)
        os.makedirs(f"{ATTN_DIR}{i}", exist_ok=True)
        os.makedirs(f"{MLP_DIR}{i}", exist_ok=True)

        WIDTH = "width_16k"

        file_path = "params.npz"

        # Download resid_out dictionaries
        closest_folder = find_closest_folder(GEMMA_BASE_URL_RESID.replace("resolve", "tree"), i, WIDTH)
        resid_url = f"{GEMMA_BASE_URL_RESID}/layer_{i}/{WIDTH}/{closest_folder}/{file_path}"
        resid_dest = f"{RESID_DIR}{i}/params.npz"
        download_file(resid_url, resid_dest)

        # Find the closest average_l0_n folder for attn_out dictionaries
        closest_folder = find_closest_folder(GEMMA_BASE_URL_ATTN.replace("resolve", "tree"), i, WIDTH)
        attn_url = f"{GEMMA_BASE_URL_ATTN}/layer_{i}/{WIDTH}/{closest_folder}/{file_path}"
        attn_dest = f"{ATTN_DIR}{i}/params.npz"
        download_file(attn_url, attn_dest)

        # Download mlp_out dictionaries
        closest_folder = find_closest_folder(GEMMA_BASE_URL_MLP.replace("resolve", "tree"), i, WIDTH)
        mlp_url = f"{GEMMA_BASE_URL_MLP}/layer_{i}/{WIDTH}/{closest_folder}/{file_path}"
        mlp_dest = f"{MLP_DIR}{i}/params.npz"
        download_file(mlp_url, mlp_dest)

    print("Gemma dictionaries downloaded successfully.")

def download_pythia():
    # Download Pythia dictionaries
    download_file(PYTHIA_BASE_URL, PYTHIA_FILE)

    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for subdir, dirs, files in os.walk('dictionaries/pythia-70m-deduped', topdown=False):
        # Check if the current directory contains only one subdirectory and no files
        if len(dirs) == 1 and len(files) == 0:
            only_subdir = dirs[0]
            subdir_path = os.path.join(subdir, only_subdir)
            
            # Move all files from the subdirectory to the current directory
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                # .pt to .safetensors conversion
                if item.endswith('.pt'):
                    state_dict = torch.load(item_path, map_location=device)
                    new_state_dict = {}
                    renaming_map = {
                        "bias": "b_dec",
                        "decoder.weight": "W_dec",
                        "encoder.bias": "b_enc",
                        "encoder.weight": "W_enc",
                    }
                    for key, value in state_dict.items():
                        new_key = renaming_map.get(key, key)
                        if "weight" in key:
                            value = value.T
                        new_state_dict[new_key] = value.contiguous().cpu()  # Move back to CPU for saving
                    save_file(new_state_dict, os.path.join(subdir, 'sae_weights.safetensors'))
                else:
                    # Convert configs to SAE format
                    with open(item_path, 'r') as f:
                        raw_config = json.load(f)
                        converted_config = convert_to_sae_config(raw_config, subdir)
                    with open(os.path.join(subdir, 'cfg.json'), 'w') as f:
                        json.dump(converted_config, f)
                os.remove(item_path)
            
            # Remove the empty subdirectory
            os.rmdir(subdir_path)

    print("Pythia dictionaries downloaded and extracted successfully.")

def download_gpt2():

    for l in range(12):
        resid_path = f"dictionaries/gpt2/resid_out_layer{l}"
        attn_path = f"dictionaries/gpt2/attn_out_layer{l}"
        mlp_path = f"dictionaries/gpt2/mlp_out_layer{l}"

        os.makedirs(resid_path, exist_ok=True)
        os.makedirs(attn_path, exist_ok=True)
        os.makedirs(mlp_path, exist_ok=True)

        for file in ["cfg.json", "sae_weights.safetensors"]:
            # Download resid dictionaries
            resid_dest = f"{resid_path}/{file}"
            download_file(f"{GPT_RESID_URL}/v5_32k_layer_{l}.pt/{file}", resid_dest)

            # Download attn dictionaries
            attn_dest = f"{attn_path}/{file}"
            download_file(f"{GPT_ATTN_URL}/v5_32k_layer_{l}/{file}", attn_dest)

            # Download mlp dictionaries
            mlp_dest = f"{mlp_path}/{file}"
            download_file(f"{GPT_MLP_URL}/v5_32k_layer_{l}/{file}", mlp_dest)

    print("GPT2 dictionaries downloaded successfully.")
    
if args.model == "gemma-2-2b":
    download_gemma_2()
elif args.model == "pythia-70m-deduped":
    download_pythia()
elif args.model == "gpt2":
    download_gpt2()