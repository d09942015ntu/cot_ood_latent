# [Implementation] Chain‑of‑Thought Prompting for Out-of-Distribution Samples: A Latent‑Variable Study 


## Installation

### Setting up venv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

### Setting up gpt2_numeric

```bash
cd mymodels
git clone https://huggingface.co/openai-community/gpt2/tree/main
cp gpt2/config.json gpt2_numeric
cp gpt2/model.safetensors gpt2_numeric
cp gpt2/tokenizer.json gpt2_numeric
cp gpt2/tokenizer_config.json gpt2_numeric
```

## Reproduce the result

```bash
python3 run_generate_dataset.py
bash run_trainer.sh
python3 run_testing.py
python3 run_visualize_result.py
```

