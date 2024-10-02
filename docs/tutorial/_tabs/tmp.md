


## About this Tutorial
- in this tutorial, we will reproduce Open-Sora 1.2, a Text-To-Image (T2I) 1.1B transformer based diffusion model
- why?
    - compared to Text-to-Image (T2I) models, Text-to-**Video** (T2I) requires a different scale.
    - T2I models training finishes in about 2.6 training days on a single 8xH100 machine. calculated down to a single gpu that's around 21 GPU hours.
    - T2V models have not settled yet, we don't know yet what works and what doesn't. the open-source project that we're reproducing here, is a comparably small model with limited capacity, yet it requires already at least 6 training days on about 192 H100 GPUs, so about 28000 GPU hours, this is three orders of magnitude longer! this means, that after a day on a single 8xH100 GPUS machine, we will not see a lot of progress, since we're have not even achieved single percentage of the full training - trust the process is the key here.
    - just as these observation, the goal of this tutorial is to show what differences appear at that scale, and how they manifest in potential problems - and, of course - how to solve them
- who is this tutorial for:
    - This isn't a casual walkthrough; it's a deep dive into the realities of scaling up T2I machine learning models to levels where standard waiting times no longer suffice.
    - The goal is to help you to kickstart the training process and prepare to being aware of differences and pitfalls in a large-scale training job.
    - We know from facebooks llama 3 paper, that around 30% of failing training jobs came from faulty GPUs in their experiments.
    - Apart from that, other problems can arise as well and we will talk about practices that helped us to find bugs in distributed python code, things to consider when coming across unexplained NCCL errors, and, data-related training problems
- you will learn:
    - how to setup the repository
    - how to prepare the dataset
    - how to start training on a cluster and how to monitor and debug jobs at scale
    - and finally, how to improve models post-hoc training.
- this tutorial in short:
    - about the model:
        - Model Type: Text-to-Video (T2V) generation model
        - Architecture: Utilizes transformer-based architectures, specifically enhanced versions like ST-DiT.
        - Diffusion Model: The model employs diffusion for video generation, using rectified flow scheduling.
        - VAE: Incorporates 3D-VAE for temporal dimension compression.
            - compared to a typical 2D-VAE, the idea here is to compress videos also in the time dimension, decreasing computational requirements put onto the diffusion model
        - Video Generation: as other transformer based methods it supports flexible resolutions (144p to 720p), aspect ratios, and lengths (2s to 16s).
    - about the data:
        - video data is not easily available and there are not many sources available publicly
        - we will use OpenVid and MiraData for our reproduction and provide the neccesary steps to download the TODO TB large data set
    - about the training:
        - how to start training at all using the collossalai launcher
        - how to monitor training in weights and biases
        - how to monitor model quality in a separate inference server
        - we will suggest settings for a speed-run (18k GPU hours) and a full-training run (100k GPU hours)
        - show intermediate results during training
        - talk shortly about how to find and optimize bottlenecks in a multi-node training setup to increase training speeds aport from faster models
    - about the post-hoc methods:
        - model averaging
        - how to apply a lora (lego reference)
    - Troubleshooting and Common Issues
        - Resolving Resource Bottlenecks
        - Handling Synchronization and Communication Problems
        - Debugging Model Convergence Issues
        - Addressing Cluster-Specific Challenges
        - Using Debugging and Profiling Tools


## Understanding Text-to-Video Diffusion Models

### Overview of Diffusion Models

### Challenges Unique to Text-to-Video Generation

### Why Large-Scale Training Is Necessary


## Setup and Installation
ansiatr

### Choosing the Right Computing Cluster
nst

### Preparing the Software Environment
nst

#### Installing Dependencies and Libraries
nst

#### Setting Up Distributed Training Frameworks
nst

### Configuring Cluster Management Tools
nst


## Pre-Training Preparations
### Dataset Collection and Preprocessing
#### Sourcing and Curating Video Datasets
#### Data Cleaning and Augmentation Techniques
#### Efficient Data Storage and Access Strategies
### Model Architecture Planning
#### Selecting or Designing Model Architectures
#### Managing Computational Constraints
### Hyperparameter Setup
#### Initial Hyperparameter Selection
#### Strategies for Hyperparameter Optimization

## Training the Model
### Launching Distributed Training Jobs
#### Writing Training Scripts
#### Submitting Jobs to the Cluster
### Resource Monitoring and Management
#### Tracking Real-Time Resource Utilization
#### Adjusting Parameters During Training
### Optimizing Training Performance
#### Techniques for Efficient Training
#### Addressing Training Bottlenecks

## Post-Training Steps
### Evaluating Model Performance
#### Metrics for Assessing Generated Videos
#### Interpreting Results and Insights
### Fine-Tuning and Model Refinement
#### Improving Model Output
#### Iterative Training Strategies
### Model Deployment
#### Optimizing Inference
#### Scaling for Production Use

