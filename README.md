# MT-VLA: Task-Semantics-Driven Expert Adaptation for Multi-Task Vision-Language-Action Models

This repository contains the official implementation of **MT-VLA**, a task-semantics-driven expert adaptation framework for unified multi-task post-training of Vision-Language-Action (VLA) models.

MT-VLA aims to improve the scalability and performance of multi-task robot learning. Instead of fine-tuning one policy for each task or training a fully shared multi-task policy, MT-VLA introduces task-semantic conditioning and expert adaptation into the action generation pathway. The framework enables a single VLA policy to share general action priors while preserving task-specific behaviors.

---

## News

- `[2026.xx]` Code and real-robot demos are released.
- `[2026.xx]` MT-VLA is evaluated on LIBERO, Meta-World, RoboTwin 2.0, and real-world ALOHA bimanual tasks.

---

## Overview

MT-VLA addresses the degradation commonly observed in conventional multi-task VLA post-training. It consists of three key components:

1. **Task-Semantic Token (TStoken)**: extracts task semantics from language instructions and injects them into the action generation pathway.
2. **Task-Semantic Mixture of Experts (TSMoE)**: augments action-side feed-forward modules with shared action priors and task-specialized LoRA experts.
3. **Task-Aware Gradient Decoupling (TGD)**: reduces conflicting optimization signals among different task instructions during multi-task training.

The framework is designed to be decoupled from a specific VLA backbone and can be applied to different action generation paradigms, including **π0**, **OpenVLA-OFT**, and **GR00T N1**.

---

## Figure 3: MT-VLA Framework

<p align="center">
  <img src="assets/figure3_framework.png" width="900">
</p>

**Figure 3. Overview of the MT-VLA framework.**  
MT-VLA introduces a unified task-semantic adaptation pipeline for multi-task vision-language-action learning. Given multi-task robot demonstrations, the language instruction is encoded into a task-semantic token, which is injected into the action generation pathway together with visual features, robot states, and action query or noise tokens. The action expert is adapted by TSMoE blocks, where a shared expert preserves task-invariant action priors and lightweight LoRA experts capture task-specific action shifts. The router selects task-relevant experts from contextual hidden states, while TGD aligns instruction-level gradients to reduce negative transfer during multi-task optimization. This design allows one unified VLA policy to retain cross-task sharing while improving task-specific specialization.

---

## Method

### Task-Semantic Token

For each training sample, MT-VLA constructs a task-semantic token from the language instruction. The token is appended to the action-side sequence and provides explicit semantic conditioning for action generation.

### Task-Semantic Mixture of Experts

TSMoE is applied to the action-side feed-forward modules of the VLA policy. It keeps the original feed-forward computation as a shared expert and introduces expert-specific low-rank residual branches for task-adaptive specialization.

### Task-Aware Gradient Decoupling

TGD groups samples according to their instruction semantics and applies gradient projection on shared trainable parameters to reduce conflicts among tasks.

---

## Installation

```bash
conda create -n mtvla python=3.10 -y
conda activate mtvla

git clone https://github.com/your-username/MT-VLA.git
cd MT-VLA

pip install -r requirements.txt