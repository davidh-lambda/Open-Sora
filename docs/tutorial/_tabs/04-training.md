---
layout: post
icon: fas fa-play-circle
title: Training
date: 2024-10-02
toc: true
---


# Get the Ball Rolling
We will primarily adhere to the multi-stage training recipe of OpenSora 1.2. The primary distinction between the stages is the resolution of the videos used for training the model.

**Original OpenSora 1.2 training configuration**:  
In detail, the training stages are defined as follows:
1. **Preparatory Stage**: This stage gradually adapts the T2I (Text-to-Image) model's weights, specifically from [PixArt-Σ 2K](https://pixart-alpha.github.io/PixArt-sigma-project/), to the proposed T2V (Text-to-Video) architecture known as [STDiT](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md#efficiency-in-choosing-the-architecture) (an acronym for **S**patio-**T**emporal **Di**ffusion **T**ransformer).
2. **Stage One** ([link to config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage1.py)): The focus here is on `240p` and `360p` video resolutions, with video lengths ranging from 2 to 16 seconds.
3. **Stage Two** ([link to config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage2.py)): This stage emphasizes `360p` and `480p` video resolutions.
4. **Stage Three** ([link to config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage3.py)): The final stage concentrates on `720p` and `1080p` video resolutions.


## Speed Run Configuration
The conversion process used to transform Pixart-Σ 2K weights to STDiT weights lacks both a config and specific commits to reproduce the original results. Although the team outlines the process in their [report notes](https://github.com/hpcaitech/Open-Sora/blob/476b6dc79720e5d9ddfb3cd589680b2308871926/docs/report_03.md#rectified-flow-and-model-adaptation), we will adopt a simpler approach.
Instead of Open-Sora's 35k H100 GPU hour training run (not counting the weight conversion training time), our approach involves training for approximately half that duration and skipping the preparation stage to assess the model's capabilities within this reduced budget. Subsequently, we will invest an additional 7K GPU hours on fresh data to evaluate whether the model's performance improves. Furthermore, we will disclose the intermediate and final outcomes of our runs and examine the two configurations we have experimented with.



**18K hour base run**  
TODO: describe the setup
{: .todo}

- [`lambda/stage1.py`{: .filepath}]()
- [`lambda/stage2.py`{: .filepath}]()
- [`lambda/stage3.py`{: .filepath}]()

**7k hour improve quality**  
TODO: describe the setup
{: .todo}

- [`lambda/stage5.py`{: .filepath}]()
- [`lambda/stage6.py`{: .filepath}]()
- [`lambda/stage7.py`{: .filepath}]()
- [`lambda/stage8.py`{: .filepath}]()


### Training Command Structure
this is how a training command looks like
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage1.py \
--data-path /home/ubuntu/ml-1cc/yunpeng/openvid_1m/data/train/OpenVid-1M-osora_le50MB.csv \
--ckpt-path pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth
```


### 18k GPU Hours of Training

TODO: describe what happens in more detail
{: .todo}

miradata from stage 3

TODO: example of clips generated from validation prompts over the course of training.
{: .todo}

**Training Commands**
{% details Click here to show the training command for stage 1. %}
```
# Stage 1
# 008-STDiT3-XL-2
# https://wandb.ai/lambdalabs/sora_speedrun/runs/gt4gjgwm
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage1.py \
--data-path OpenVid-1M-osora_le50MB.csv \
--ckpt-path pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth
```
{% enddetails %}

{% details Click here to show the training command for stage 2. %}
```
# Stage 2
# 009-STDiT3-XL-2
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage2.py \
--data-path OpenVid-1M-osora_le50MB.csv \
--ckpt-path ./outputs_speedrun/008-STDiT3-XL-2/epoch4-global_step2210/
```
{% enddetails %}

{% details Click here to show the training command for stage 3. %}
```
# Stage 3
# 010-STDiT3-XL-2
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage3.py \
--data-path OpenVid-Miradata-mix.csv \
--ckpt-path ./outputs_speedrun/009-STDiT3-XL-2/epoch4-global_step7099
```
{% enddetails %}

### Additional 7k GPU Hours of Training

TODO: describe what happens in more detail
- no masking
{: .todo}

TODO: example of clips generated from validation prompts over the course of training.
{: .todo}

**Training Commands**
{% details Click here to show the training command for stage 4. %}
```
# Stage 4
# 012-STDiT3-XL-2
# load from stage 3
# 1CycleWarmUp
# No mask
# lowres config
# OpenVid-Miradata-mix.csv
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage4.py \
--data-path OpenVid-Miradata-mix.csv \
--ckpt-path ./outputs_speedrun/010-STDiT3-XL-2/epoch4-global_step14778
```
{% enddetails %}

{% details Click here to show the training command for stage 5. %}
```
# Stage 5
# 013-STDiT3-XL-2
# load from stage 4
# 1CycleWarmUp
# No mask
# midres config
# OpenVid-Miradata-mix.csv
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage5.py \
--data-path OpenVid-Miradata-mix.csv \
--ckpt-path ./outputs_speedrun/012-STDiT3-XL-2/epoch3-global_step2100
```
{% enddetails %}

{% details Click here to show the training command for stage 6. %}
```
# Stage 6
# 015-STDiT3-XL-2
# load from stage 5
# 1CycleWarmUp
# No mask
# midres config
# OpenVid-Miradata-mix.csv
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage6.py \
--data-path OpenVid-Miradata-mix.csv \
--ckpt-path ./outputs_speedrun/013-STDiT3-XL-2/epoch1-global_step3300
```
{% enddetails %}

{% details Click here to show the training command for stage 7. %}
```
# Stage 7 (continue Stage 6 by replacing node012 by node031)
# 019-STDiT3-XL-2
# load from stage 8
# 1CycleWarmUp
# No mask
# midres config
# OpenVid-Miradata-mix.csv
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile hostfile_calvin_31 \
scripts/train.py configs/opensora-v1-2/train/stage6.py \
--data-path OpenVid-Miradata-mix.csv \
--ckpt-path ./outputs_speedrun/015-STDiT3-XL-2/epoch0-global_step2900
```
{% enddetails %}

## Monitoring Model Quality
We monitor loss curves in [weights and biases](https://wandb.com). Importantly, we need to evaluate the model beyond its training loss by the quality of videos generated from a set of validation prompts. This requires constanly run run model inference on a separate inference server so to unblock the training.

TODO: how to start the inference server. 
{: .todo}

## Monitoring Cluster Health
We use an internal tool to monitor the cluster health. It logs metrics such as x, y, z, and these are some screenshots.
Talk to Landon for screen shots.
{: .todo}

As mentioned in this [paper](https://scontent-sjc3-1.xx.fbcdn.net/v/t39.2365-6/453304228_1160109801904614_7143520450792086005_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=D3eIShkp1TcQ7kNvgE-JJzm&_nc_ht=scontent-sjc3-1.xx&_nc_gid=ALn92M1rjShxW06X9RGrGBT&oh=00_AYBxxn4Xx7EA2XpjH0xWOopzeB3-ZtDTZUZRkQXl6322RA&oe=6705DE47), large scale distributed training will face downtime. And indeed we exprienced that. To know more, checkout the later tourbleshooting section.

REMINDER: add some wandb screenshot for loss etc, and GPU powerdraw etc.
