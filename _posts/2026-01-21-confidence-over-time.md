---
layout: post
title: "Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning"
date: 2026-01-21 00:00:00 +0000
categories: research machine-learning nlp
author: Anirudhh Venkat
authors: Zhenjiang Mao*, <strong>Anirudhh Venkat*</strong>, Artem Bisliouk, Akshat Kothiyal, Sindhura Kumbakonam Subramanian, Saithej Singhu, Ivan Ruchkin
venue: arXiv preprint
image: /images/arxiv.png
---

<p><a href="https://arxiv.org/abs/2601.13387" target="_blank" rel="noopener" style="display:inline-block;padding:8px 12px;border:1px solid #444;border-radius:6px;background:#f5f5f5;color:#111;text-decoration:none">arXiv</a></p>

Abstract

Large Language Models (LLMs) increasingly rely on long-form, multi-step reasoning to solve complex tasks such as mathematical problem solving and scientific question answering. Despite strong performance, existing confidence estimation methods typically reduce an entire reasoning process to a single scalar score, ignoring how confidence evolves throughout the generation. As a result, these methods are often sensitive to superficial factors such as response length or verbosity, and struggle to distinguish correct reasoning from confidently stated errors. We propose to characterize the stepwise confidence signal using Signal Temporal Logic (STL). Using a discriminative STL mining procedure, we discover temporal formulas that distinguish confidence signals of correct and incorrect responses. Our analysis found that the STL patterns generalize across tasks, and numeric parameters exhibit sensitivity to individual questions. Based on these insights, we develop a confidence estimation approach that informs STL blocks with parameter hypernetworks. Experiments on multiple reasoning tasks show our confidence scores are more calibrated than the baselines.

Cite

Mao, Z., Venkat, A., Bisliouk, A., Kothiyal, A., Subramanian, S. K., Singhu, S., & Ruchkin, I. (2026). Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning. arXiv:2601.13387. https://doi.org/10.48550/arXiv.2601.13387
