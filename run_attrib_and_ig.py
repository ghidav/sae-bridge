import subprocess
from typing import Dict, List
import logging

# Constants
MODELS: List[str] = ["pythia-70m-deduped", "gpt2"]
DATASETS: List[str] = ["rc_train", "greater_than_examples", "greater_than_examples_random_tokens", "ioi_examples"]

EXAMPLE_LENGTH_FOR_DATASET: Dict[str, int] = {
    "rc_train": 6,
    "greater_than_examples": 12,
    "greater_than_examples_random_tokens": 12,
    "ioi_examples": 15, 
}

MODEL_SHORT_NAME: Dict[str, str] = {
    "pythia-70m-deduped": "pythia",
    "gemma-2-2b": "gemma",
}

DATASET_SHORT_NAME: Dict[str, str] = {
    "rc_train": "rc",
    "greater_than_examples": "gtthan",
    "greater_than_examples_random_tokens": "dummy",
    "ioi_examples": "ioi",
}

# Configuration
NUM_EXAMPLES: int = 100
DICT_PATH: str = "dictionaries"
DICT_ID: int = 10
NODE_THRESHOLD: float = 0.1
BATCH_SIZE: int = 10
PEN_THICKNESS: int = 10
AGGREGATION: str = "none"
PLOT_DIR: str = "circuits/figures"
SEED: int = 12
DEVICE: str = "cuda:0"
NOPAIR: bool = False
PLOT_CIRCUIT: bool = False
NODES_ONLY: bool = True
PLOT_ONLY: bool = False
DICT_SIZE: int = 32768  # TODO: deprecate
MAX_FEATURES: int = 32768  # TODO: deprecate

# Standard arguments
BASE_COMMAND = [
    "python", "circuit.py",
    "--num_examples", str(NUM_EXAMPLES),
    "--dict_path", DICT_PATH,
    "--node_threshold", str(NODE_THRESHOLD),
    "--dict_id", str(DICT_ID),
    "--dict_size", str(DICT_SIZE),
    "--batch_size", str(BATCH_SIZE),
    "--aggregation", AGGREGATION,
    "--max_features", str(MAX_FEATURES),
    "--pen_thickness", str(PEN_THICKNESS),
    "--plot_dir", PLOT_DIR,
    "--seed", str(SEED),
    "--device", DEVICE,
]
if NOPAIR:
    BASE_COMMAND.append("--nopair")
if PLOT_CIRCUIT:
    BASE_COMMAND.append("--plot_circuit")
if NODES_ONLY:
    BASE_COMMAND.append("--nodes_only")
if PLOT_ONLY:
    BASE_COMMAND.append("--plot_only")

def run_command(command: List[str], method: str) -> None:
    print(f"Running for model {model} and dataset {dataset} with method {method}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        logging.info(f"Script with {method} executed successfully")
        logging.debug(f"Output: {result.stdout}")
    else:
        logging.error(f"Script execution failed for {method}")
        logging.error(f"Error: {result.stderr}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for model in MODELS:
        for dataset in DATASETS:
            assert dataset in EXAMPLE_LENGTH_FOR_DATASET, f"Dataset {dataset} not in EXAMPLE_LENGTH_FOR_DATASET. Available datasets: {EXAMPLE_LENGTH_FOR_DATASET.keys()}"
            
            example_length = EXAMPLE_LENGTH_FOR_DATASET[dataset]
            model_short_name = MODEL_SHORT_NAME.get(model, model)
            dataset_short_name = DATASET_SHORT_NAME.get(dataset, dataset)
            circuit_dir = f"circuits/attribandig_{model_short_name}_{dataset_short_name}"
            
            base_command = BASE_COMMAND + [
                "--example_length", str(example_length),
                "--circuit_dir", circuit_dir,
                "--dataset", dataset,
                "--model", model
            ]

            for method in ["ig", "attrib"]:
                command = base_command + ["--method", method]
                run_command(command, method)