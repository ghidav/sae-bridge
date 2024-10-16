# SAEBridge

This repo contains the code for the paper:

*"Comparing Coarse and Fine-grained Mechanistic Interpretability in Language Models and Getting the Best of Both Worlds"*

This repository has been adapted from https://github.com/saprmarks/feature-circuits for our use case, and has been modified to employ the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and the [SAELens](https://github.com/jbloomAus/SAELens) libraries.

## Setup

To set up the project, run these commands from the root of the repository:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install nvitop
pip install -U "huggingface_hub[cli]"
export WANDB_API_KEY=your_key_here
git lfs install
pip install -r requirements.txt
```

## Computing effects

To replicate the results of the paper, do the following:
1. Download the dictionaries by inserting your huggingface key in keys.json and then running the following command:
```bash
python download_dictionaries.py -m <model_name>
```

2. To obtain the AtP and IG scores for each model and each dataset, use run_attrib_and_ig.py, after having chosen the models and datasets that you want to run.

```bash
python run_attrib_and_ig.py
```
Then use plotting.ipynb to plot the results.

## License

This repository is adapted from [feature-circuits](https://github.com/saprmarks/feature-circuits) by [saprmarks](https://github.com/saprmarks). The original project is licensed under the MIT License, and we have included a copy of this license in our repository.
