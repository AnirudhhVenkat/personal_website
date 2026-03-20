---
layout: post
title: "Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning"
date: 2026-01-21 00:00:00 +0000
categories: research machine-learning nlp
author: Anirudhh Venkat
authors: Zhenjiang Mao*, <strong>Anirudhh Venkat*</strong>, Artem Bisliouk, Akshat Kothiyal, Sindhura Kumbakonam Subramanian, Saithej Singhu, Ivan Ruchkin
venue:
image: /images/arxiv.png
excerpt: "Characterizes stepwise confidence in LLM reasoning using Signal Temporal Logic and trains STL-informed blocks to produce better-calibrated confidence scores."
---

<p><a href="https://arxiv.org/abs/2601.13387" class="paper-link" target="_blank" rel="noopener">arXiv</a></p>

Large Language Models (LLMs) increasingly rely on long-form, multi-step reasoning to solve complex tasks such as mathematical problem solving and scientific question answering. Despite strong performance, existing confidence estimation methods typically reduce an entire reasoning process to a single scalar score, ignoring how confidence evolves throughout the generation. As a result, these methods are often sensitive to superficial factors such as response length or verbosity, and struggle to distinguish correct reasoning from confidently stated errors. We propose to characterize the stepwise confidence signal using Signal Temporal Logic (STL). Using a discriminative STL mining procedure, we discover temporal formulas that distinguish confidence signals of correct and incorrect responses. Our analysis found that the STL patterns generalize across tasks, and numeric parameters exhibit sensitivity to individual questions. Based on these insights, we develop a confidence estimation approach that informs STL blocks with parameter hypernetworks. Experiments on multiple reasoning tasks show our confidence scores are more calibrated than the baselines.

UF Innovate Award at Warren B. Nelms Annual IoT Conference ($500)
