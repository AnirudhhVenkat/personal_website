---
layout: post
title: "Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning"
date: 2026-01-19 00:00:00 +0000
categories: research machine-learning nlp
author: Anirudhh Venkat
authors: <strong>Zhenjiang Mao*, Anirudhh Venkat*, Artem Bisliouk, Akshat Kothiyal, Sindhura Kumbakonam Subramanian, Saithej Singhu, Ivan Ruchkin</strong>
venue: arXiv preprint
image: /images/arxiv.png
---

Paper (arXiv): https://arxiv.org/abs/2601.13387
PDF: https://arxiv.org/pdf/2601.13387.pdf

Abstract

Large Language Models (LLMs) increasingly rely on long-form, multi-step reasoning to solve complex tasks such as mathematical problem solving and scientific question answering. Despite strong performance, existing confidence estimation methods typically reduce an entire reasoning process to a single scalar score, ignoring how confidence evolves throughout the generation. As a result, these methods are often sensitive to superficial factors such as response length or verbosity, and struggle to distinguish correct reasoning from confidently stated errors. We propose to characterize the stepwise confidence signal using Signal Temporal Logic (STL). Using a discriminative STL mining procedure, we discover temporal formulas that distinguish confidence signals of correct and incorrect responses. Our analysis found that the STL patterns generalize across tasks, and numeric parameters exhibit sensitivity to individual questions. Based on these insights, we develop a confidence estimation approach that informs STL blocks with parameter hypernetworks. Experiments on multiple reasoning tasks show our confidence scores are more calibrated than the baselines.

Cite

Mao, Z., Venkat, A., Bisliouk, A., Kothiyal, A., Subramanian, S. K., Singhu, S., & Ruchkin, I. (2026). Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning. arXiv:2601.13387. https://doi.org/10.48550/arXiv.2601.13387
