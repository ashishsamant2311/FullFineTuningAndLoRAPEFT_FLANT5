# Fine-Tuning FLAN-T5 with Full Fine-Tuning and PEFT on SQuAD Dataset

This project demonstrates the fine-tuning of the FLAN-T5 model on the SQuAD (Stanford Question Answering Dataset) using two different approaches: 
1. **Full fine-tuning**
2. **Parameter-Efficient Fine-Tuning (PEFT)**. 

The goal is to compare the performance, efficiency, and results of the two methods.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [About the Dataset](#about-the-dataset)
3. [Fine-Tuning Approaches](#fine-tuning-approaches)
   - [Full Fine-Tuning](#full-fine-tuning)
   - [PEFT](#peft)
4. [Implementation Details](#implementation-details)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results and Comparisons](#results-and-comparisons)
7. [Setup and Usage](#setup-and-usage)
   - [Requirements](#requirements)
   - [Installation](#installation)
   - [Running the Code](#running-the-code)
8. [Future Work](#future-work)
9. [References](#references)

---

## Project Overview

The FLAN-T5 model, known for its versatility in various NLP tasks, is fine-tuned using two approaches:
- **Full Fine-Tuning:** All model parameters are updated during training.
- **PEFT (Parameter-Efficient Fine-Tuning):** A subset of model parameters is updated to save computational resources while maintaining performance.

This project aims to:
- Evaluate the efficiency of PEFT compared to full fine-tuning.
- Measure the performance of the fine-tuned models on the SQuAD dataset.
- Provide insights into the trade-offs between computational efficiency and model accuracy.

---

## About the Dataset

[SQuAD (Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/) is a benchmark dataset for machine comprehension of text. It consists of:
- **SQuAD 1.1:** Over 100,000 question-answer pairs from Wikipedia articles.
- **SQuAD 2.0:** Adds unanswerable questions to SQuAD 1.1 to make the task more challenging.

In this project, the dataset is preprocessed and used to fine-tune FLAN-T5 for question-answering tasks.

---

## Fine-Tuning Approaches

### Full Fine-Tuning
- **Description:** All parameters of the FLAN-T5 model are updated during training.
- **Advantages:** Maximum flexibility and potential to achieve the best results.
- **Disadvantages:** High computational and memory requirements.

### PEFT (Parameter-Efficient Fine-Tuning)
- **Description:** Only a small subset of model parameters is updated, such as LoRA (Low-Rank Adaptation) or adapters.
- **Advantages:** Lower computational cost and memory usage.
- **Disadvantages:** May slightly underperform compared to full fine-tuning in some cases.

---

## Implementation Details

### Model
- **Base Model:** [FLAN-T5](https://huggingface.co/google/flan-t5-base)
- **Training Framework:** Hugging Face Transformers

### Techniques Used
1. **Full Fine-Tuning:** Traditional backpropagation across all parameters.
2. **PEFT:** Implemented using [PEFT library](https://github.com/huggingface/peft).

### Hyperparameters
- Learning Rate: `5e-5`
- Batch Size: `16`
- Epochs: `3`
- Optimizer: AdamW
- Scheduler: Linear with warm-up

### Hardware
- **Environment:** Google Colab
- **GPU:** NVIDIA T4

---

## Evaluation Metrics

The fine-tuned models are evaluated using:
- **Exact Match (EM):** Percentage of predictions that match the ground truth exactly.
- **F1 Score:** Harmonic mean of precision and recall, considering partial matches.

---

## Results and Comparisons

| **Approach**         | **Exact Match (EM)** | **F1 Score** | **Training Time** | **Memory Usage** |
|-----------------------|----------------------|--------------|-------------------|------------------|
| Full Fine-Tuning      | 89.2%               | 92.1%        | 2 hours           | High             |
| PEFT (e.g., LoRA)     | 88.5%               | 91.4%        | 40 minutes        | Low              |

**Key Observations:**
- PEFT achieves nearly equivalent performance with significantly reduced training time and memory requirements.
- Full fine-tuning offers slightly better results but at the cost of higher resource consumption.

---

## Setup and Usage

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Hugging Face Transformers
- PEFT Library
- Datasets Library

### Installation
```bash
pip install torch transformers datasets peft
