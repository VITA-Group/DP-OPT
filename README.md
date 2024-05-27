DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer
====================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch Code for Paper: "DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer" [Junyuan Hong](https://jyhong.gitlab.io/), [Jiachen T. Wang](https://tianhaowang.netlify.app/), [Chenhui Zhang](https://scholar.google.com/citations?user=UYxdrBsAAAAJ&hl=en), [Zhangheng Li](https://scholar.google.com/citations?user=NZCLqZMAAAAJ), [Bo Li](https://aisecure.github.io/), [Zhangyang Wang](https://vita-group.github.io/), *ICLR (Spotlight, top-5%)* 2024.

[paper](https://arxiv.org/abs/2312.03724) / [code](https://github.com/VITA-Group/DP-OPT) / [blog](https://jyhong.gitlab.io/publication/2023dp_opt/)

**TL;DR**: We proposed the first end-to-end privacy-preserving automatic prompt engineering method.

## Overview


![featured](https://jyhong.gitlab.io/publication/2023dp_opt/featured.png)

Large Language Models (LLMs) have emerged as dominant tools for various tasks, particularly when tailored for a specific target by prompt tuning. Nevertheless, concerns surrounding data privacy present obstacles due to the tuned prompts' dependency on sensitive private information. A practical solution is to host a local LLM and optimize a soft prompt privately using data. Yet, hosting a local model becomes problematic when model ownership is protected. Alternative methods, like sending data to the model’s provider for training, intensify these privacy issues facing an untrusted provider. In this paper, we present a novel solution called Differentially-Private Offsite Prompt Tuning (DP-OPT) to address this challenge. Our approach involves tuning a discrete prompt on the client side and then applying it to the desired cloud models. We demonstrate that prompts suggested by LLMs themselves can be transferred without compromising performance significantly. To ensure that the prompts do not leak private information, we introduce the first private prompt generation mechanism, by a differentially-private (DP) ensemble of in-context learning with private demonstrations. With DP-OPT, generating privacy-preserving prompts by Vicuna-7b can yield competitive performance compared to non-private in-context learning on GPT3.5 or local private prompt tuning.

## Get Started

Prepare conda env.
```shell
conda create --name dp-opt python=3.8 -y
conda activate dp-opt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate sentencepiece scikit-learn wandb autodp
# transformers==4.28.1
```

Prepare DLN datasets
```shell
bash setup_data.sh
```

To use openai models, create `openai_config.py` in the root folder. This will be only used for evaluation.
```python
import openai

openai.api_key = "<your-key>"
# openai.organization = "<your-org>"
openai.api_base = "https://api.openai.com/v1"
openai_model_types = ['text-davinci-003']
```

**Example**: Use local model (`lmsys/vicuna-7b-v1.3`) to generate a instruction and test the instruction by OpenAI model (`text-davinci-003`).
* OPT:
```shell
# generate a instruction
python train_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 --holdout_ratio=0.01
# evaluate the instruction
python eval_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 \
--test_model=text-davinci-003
```
* DP-OPT:
```shell
# generate a instruction
python train_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 --holdout_ratio=0.01 \
--target_eps=8. --dp_eps=1.8 --dp_delta=5e-7 --tokenwise_gen=True
# evaluate the instruction
python eval_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 \
--data=sst2 \
--target_eps=8. --dp_eps=1.8 --dp_delta=5e-7 --tokenwise_gen=True \
--test_model=text-davinci-003
```

## Experiments

Wandb sweeps files are under `sweeps/<data_name>/<method>.yml`.
`sweeps/<data_name>/<method>.yml` is used for tuning prompts.
We use `sweeps/<data_name>/<method>_test.yml` to test prompts on different models.

Supported datasets: `sst2`, `trec`, `mpqa`, `disaster`.

![image](https://github.com/VITA-Group/DP-OPT/assets/6964516/8040b268-1c19-4d5a-8583-44ed23a0a090)

Methods (exmaplified on `sst2`):
* 5-shot In-Context Learning (ICL)
```shell
wandb sweep sweeps/sst2/icl.yml
```
* Deep Language Network with One-layer (DLN-1)
```shell
wandb sweep sweeps/sst2/dln1.yml
wandb sweep sweeps/sst2/dln1_test.yml
```
* Offsite Prompt Tuning (OPT)
```shell
wandb sweep sweeps/sst2/opt.yml
wandb sweep sweeps/sst2/opt_test.yml
```
* Differentially-Private Offsite Prompt Tuning (DP-OPT)
```shell
wandb sweep sweeps/sst2/dp-opt.yml
wandb sweep sweeps/sst2/dp-opt_test.yml
```
-----
Part of the codes are based on [deep-language-networks](https://github.com/microsoft/deep-language-networks).

