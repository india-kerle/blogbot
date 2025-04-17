# 🤖 Blogbot violations model: TL;DR

This repo contains the scripts necessary to:

1. **Generate synthetic data**: by injecting PII into a stratified sample of existing blogposts (by age, gender and topic), downloaded from [kaggle.com](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus). 

```
modal deploy blogbot/llm.py ## deploy your Llm 
modal run --detach blogbot/synthetic_data_generation.py ## Run the synthetic data generation sript
```

2. **Train a model**: to detect PII in blogposts using synthetic data generated in step 1 for the positive class (has violations) and a sample of blogposts without artifically injected PII.

```
modal run --detach blogbot/train.py ##train and test the model on the data
```

3. **Evaluate the model**: on a test set of blogposts with injected PII and without PII. This also happens in the training script. 

## 🗄️ Set up

### Your environment

In order to set up your environment, you can create a python environment, activate it and install dependencies.

```
python -m venv .venv
source .venv/bin/activate 
pip install -e .
```

### Setting up modal

1. Create an account at [modal.com](https://modal.com/)
2. Run pip install modal to install the modal Python package
3. Run modal setup to authenticate (if this doesn’t work, try `python -m modal setup`)

### Unit tests 

To run unit tests (and after setting up your environment), please run:

```
pytest tests/unit
```
