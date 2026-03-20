---
layout: post
title: "Recurrent Confidence Chain: Temporal-Aware Uncertainty Quantification in Large Language Models"
date: 2026-01-19 00:00:00 +0000
categories: research machine-learning nlp
author: Anirudhh Venkat
authors: <strong>Zhenjiang Mao*, Anirudhh Venkat*</strong>
venue: arXiv preprint
image: /images/arxiv.png
---

Paper (arXiv): https://arxiv.org/abs/2601.13368
PDF: https://arxiv.org/pdf/2601.13368.pdf

Abstract

As reasoning modules, such as the chain-of-thought mechanism, are applied to large language models, they achieve strong performance on various tasks such as answering common-sense questions and solving math problems. The main challenge now is to assess the uncertainty of answers, which can help prevent misleading or serious hallucinations for users. Although current methods analyze long reasoning sequences by filtering unrelated tokens and examining potential connections between nearby tokens or sentences, the temporal spread of confidence is often overlooked. This oversight can lead to inflated overall confidence, even when earlier steps exhibit very low confidence. To address this issue, we propose a novel method that incorporates inter-step attention to analyze semantic correlations across steps. For handling long-horizon responses, we introduce a hidden confidence mechanism to retain historical confidence information, which is then combined with stepwise confidence to produce a more accurate overall estimate. We evaluate our method on the GAOKAO math benchmark and the CLadder causal reasoning dataset using mainstream open-source large language models. Our approach is shown to outperform state-of-the-art methods by achieving a superior balance between predictive quality and calibration, demonstrated by strong performance on both Negative Log-Likelihood and Expected Calibration Error.

Cite

Mao, Z., & Venkat, A. (2026). Recurrent Confidence Chain: Temporal-Aware Uncertainty Quantification in Large Language Models. arXiv:2601.13368.
