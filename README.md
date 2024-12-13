# Data Laundering

We show that knowledge distillation can be subverted to manipulate language model benchmark scores, revealing a critical vulnerability in current evaluation practices. We introduce "Data Laundering," a three-phase process analogous to financial money laundering, that enables the covert transfer of benchmark-specific knowledge through seemingly legitimate intermediate training steps. Through extensive experiments with a 2-layer BERT student model, we show how this approach can achieve substantial improvements in benchmark accuracy (up to 75\% on GPQA) without developing genuine reasoning capabilities. Our investigation examines various aspects of this vulnerability, including the impact of different loss functions (MSE vs. KL divergence), the role of soft-label weighting ($\alpha$ parameter), the effects of iterative distillation, and the influence of training dataset size. Notably, this method can be exploited intentionally or even unintentionally, as researchers may inadvertently adopt this method that inflates scores using knowledge distillation without realizing the implications. While our findings demonstrate the effectiveness of this technique, we present them as a cautionary tale highlighting the urgent need for more robust evaluation methods in AI. This work aims to contribute to the ongoing discussion about evaluation integrity in AI development and the need for benchmarks that more accurately reflect true model capabilities.

<!-- TOC -->

- [Method](#method)
- [Data](#data)
- [Models](#models)
- [Evaluation](#evaluation)
- [Citation](#citation)

<!-- /TOC -->

## Method

## Data

## Model

## Evaluation

## Citation
