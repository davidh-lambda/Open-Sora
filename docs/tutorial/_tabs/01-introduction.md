---
layout: post
icon: fas fa-lightbulb
title: Introduction
date: 2024-10-02
toc: true
---

# Let's reproduce a T2I model!
In this tutorial, we'll reproduce [**Open-Sora 1.2**](https://github.com/hpcaitech/Open-Sora), a **1.1B parameter Text-to-Video (T2V) model** based on a transformer-based diffusion architecture. Text-to-Video models require a different scaling than Text-to-Image (T2I) models. This tutorial will guide you through the process of downloading and preparing the dataset, training this model from scratch, addressing the unique challenges of training at scale, and providing tips for troubleshooting a distributed training job.


## Why This Tutorial?
Newer T2I training schemes [finish training in about 2.6 days on a single 8xH100 GPU machine](https://arxiv.org/abs/2407.15811), which boils down to around 21 GPU hours per GPU. In contrast, *T2V models* are still in their infancy; we have yet to find out what works and what doesn't. The open-source project we're reproducing is a comparatively small model with limited capacity, yet it requires at least six training days on approximately 192 H100 GPUs — that's about **28,000 GPU hours** — three orders of magnitude longer compared to a fast T2I training scheme!

However, this also means that after a full day of training on a single 8xH100 GPU machine, we won't see significant progress, as a day's worth of training represents less than one percent of the total training time. **Trusting the process** is crucial in this context. The purpose of this tutorial is to ensure that resources are used efficiently by removing points of failures as early as possible, to highlight the differences that emerge at such a large scale, how they present potential issues, and — most importantly — how to address and resolve them.


## Who Is This Tutorial For?
Unfortunately, this isn't a casual walkthrough that you can follow on your MacBook, but rather a deep dive into the realities of scaling up T2V machine learning models to levels where standard waiting times no longer suffice. But if you're looking for a document to kickstart the training process and want to be aware of differences and pitfalls in large-scale training jobs, this tutorial is for you.

From [Facebook's LLaMA 3.1 paper](https://arxiv.org/abs/2407.21783), we know that around **30% of failing training jobs came from faulty GPUs**. But also beyond hardware issues, other problems can arise. We'll discuss practices that helped us find bugs in distributed Python code, considerations when encountering unexplained NCCL errors, and data-related training problems.


## Tutorial Overview & What You'll Learn


### [**Setup** - Clone, Install & Setup your Cluster](../02-setup)
Let's start by setting everything up:
- **Basic Setup**:
    - We'll guide you through cloning and configuring the necessary codebase.
    - Installing conda & dependencies
- **Preparing the Cluster**:
    - Making sure that all nodes have access to needed files (dataset, huggingface weights, conda environments)
    - Defining the nodes list for the training job
- **Setting up the Inference Server**


### [**Dataset** — Downloading & Preprocessing](../03-dataset)
Let's face it: Video data is not easily accessible, and there aren't many publicly available sources.  
We'll
- download [**OpenVid**](https://github.com/NJU-PCALab/OpenVid-1M) and [**MiraData**](https://github.com/mira-space/MiraData) for our reproduction experiment
- walk through the necessary steps to **preprocess the datasets**.
- Since preprocessing takes time even at this scale, we'll also talk about how to use your cluster to **parallelize** such preprocessing tasks conveniently without writing any code - call it a little love letter to what Unix already brings to the table if you want <3.


### [**Training** — Get the Ball Rolling](../04-training)
Training on this scale also presents unique challenges; here's what we'll cover.
- **Training Configurations**: We'll suggest settings for a speed run (18k GPU hours) and a full training run (100k GPU hours) for you to choose from and discuss what to expect from each. We'll also share intermediate and final results for our runs and discuss the two setups that we've tested.
- **Starting and Monitoring Training on a Cluster**: Open-Sora is built on top of the [ColossalAI launcher](https://colossalai.org/). We'll start by simply providing the commands to get you started and how to monitor loss curves in [weights and biases](https://wandb.com).
- **Evaluating Model Quality**: Understand how to evaluate model performance using a separate inference server.
- **Monitoring Cluster Health**: ?
- **Optimizing Performance**: We'll discuss how to identify and optimize bottlenecks in a multi-node training setup to increase training speed beyond just using faster models.


### [**Post-Hoc Methods** — Improving the Quality After the Fact](../05-post-hoc)
After training, we'll explore methods to enhance model performance after training, including model averaging and applying LoRA (Low-Rank Adaptation).
- **Inference Parameterrs**: Built-in ways to improve the Quality.
- **Model Averaging**: Merging checkpoints to improve the model further.
- **Applying LoRA**: How to implement Low-Rank Adaptation to fine-tune your model.


### [**Troubleshooting & Common Issues**](../06-troubleshooting)
Finally, we'll address common problems you might encounter and how to solve them.
- **Debugging Model Convergence Issues**: How to address problems when your model isn't converging as expected.
- **Addressing Cluster-Specific Challenges**: Guidance on navigating the complexities of training on a cluster.
- **Debugging on a Cluster**: How to use [py-spy](https://github.com/benfred/py-spy) to debug a distributed training data loader.
- **Resolving Resource Bottlenecks**: Strategies to manage and optimize resource utilization.

By the end of this tutorial, you'll have a comprehensive understanding of what's involved in scaling up T2V models like Open-Sora 1.2. You'll be better equipped to handle the challenges that come with large-scale training and be prepared to troubleshoot and optimize your models effectively.