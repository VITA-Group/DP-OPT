DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer
====================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch Code for Paper: "DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer" [Junyuan Hong](https://jyhong.gitlab.io/), [Jiachen T. Wang](https://tianhaowang.netlify.app/), [Chenhui Zhang](https://scholar.google.com/citations?user=UYxdrBsAAAAJ&hl=en), Zhangheng Li, [Bo Li](https://aisecure.github.io/), [Zhangyang Wang](https://vita-group.github.io/), *Preprint* 2023.

[paper](https://arxiv.org/abs/2312.03724) / [code](https://github.com/VITA-Group/DP-OPT) / [blog](https://jyhong.gitlab.io/publication/2023dp_opt/)

## Overview


![featured](https://jyhong.gitlab.io/publication/2023dp_opt/featured.png)

Large Language Models (LLMs) have emerged as dominant tools for various tasks, particularly when tailored for a specific target by prompt tuning. Nevertheless, concerns surrounding data privacy present obstacles due to the tuned prompts' dependency on sensitive private information. A practical solution is to host a local LLM and optimize a soft prompt privately using data. Yet, hosting a local model becomes problematic when model ownership is protected. Alternative methods, like sending data to the model’s provider for training, intensify these privacy issues facing an untrusted provider. In this paper, we present a novel solution called Differentially-Private Offsite Prompt Tuning (DP-OPT) to address this challenge. Our approach involves tuning a discrete prompt on the client side and then applying it to the desired cloud models. We demonstrate that prompts suggested by LLMs themselves can be transferred without compromising performance significantly. To ensure that the prompts do not leak private information, we introduce the first private prompt generation mechanism, by a differentially-private (DP) ensemble of in-context learning with private demonstrations. With DP-OPT, generating privacy-preserving prompts by Vicuna-7b can yield competitive performance compared to non-private in-context learning on GPT3.5 or local private prompt tuning.
