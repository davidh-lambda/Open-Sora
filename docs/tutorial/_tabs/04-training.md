---
layout: post
icon: fas fa-play-circle
title: Training
date: 2024-10-02
toc: true
---

> While the reading time of this section may be low, note that running the commands on this page may take a bit longer. For us the total time was `~5` days on a cluster with `192` H100 GPUs.
{: .prompt-tip}


# Get the Ball Rolling
We will primarily adhere to the multi-stage training recipe of OpenSora 1.2.
The original recipe involves three stages of training, with the primary distinction of an increase of resolution of the input videos used for training the model from one stage to the next.

**Original OpenSora 1.2 training configuration**:  
In detail, the training stages are defined as follows:
1. **Preparatory Stage**: This "zero-th" stage gradually adapts the T2I (Text-to-Image) model's weights, specifically from [PixArt-Σ 2K](https://pixart-alpha.github.io/PixArt-sigma-project/), to the proposed T2V (Text-to-Video) architecture named [STDiT](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md#efficiency-in-choosing-the-architecture) (an acronym for **S**patio-**T**emporal **Di**ffusion **T**ransformer).
2. **Stage One** ([link to config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage1.py)): The focus here is on `240p` and `360p` video resolutions, with video lengths ranging from 2 to 16 seconds.
3. **Stage Two** ([link to config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage2.py)): This stage emphasizes `360p` and `480p` video resolutions.
4. **Stage Three** ([link to config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage3.py)): The final stage concentrates on `720p` and `1080p` video resolutions.

Unfortunately, the conversion process used to transform Pixart-Σ 2K weights to STDiT weights lacks both a config and specific commits to reproduce the original results. Although the team outlines the process of the required model code adaptions in their [report notes](https://github.com/hpcaitech/Open-Sora/blob/476b6dc79720e5d9ddfb3cd589680b2308871926/docs/report_03.md#rectified-flow-and-model-adaptation), we will adopt a simpler approach for this tutorial at the cost of a decrease in quality and knowledge about typical topics.

## Speed Run
Thus, instead of Open-Sora's 35k H100 GPU hour training run (not counting the weight conversion training time), our approach for this tutorial involves training for approximately half that duration and skipping the preparation stage to assess the model's capabilities within this reduced budget. Subsequently, we will invest an additional 7K GPU hours to evaluate whether the model's performance improves. Furthermore, we will disclose the intermediate and final outcomes of our runs and examine the two configurations we have experimented with.


### Configuration
- we train on a cluster consisting of 192 H100 GPUs
- our version the three stages described above matches mostly the recipe of Open-Sora's original training approach
- we adapted the learning rate according to counter the increased batch size and with that increased implicit step size
- additionally, we decreased the number of warmup steps from 1000 to 400 (by observing the loss, we did not feel that warmup requires that many steps)
- we added weight decay according to the recomendations in [this paper](https://arxiv.org/abs/2407.15811), setting to a high weight decay value of `0.01`.
    - upon inspection of model outputs, we felt that adding weight decay has lead to bigger changes to occur when stuck in local minima
- thus our tutorial, trying to mimick a typical research-oriented foundation model training, involves two parts:
    - a 18k gpu hour run using only small changes that we are more sure improve training (weight decay adaption) - we don't expect great results yet here, since this is only half of the training time of open-sora, and basically training from scratch, as the weights are not converted "correctly" from pixart to STDiT.
    - an additional 7k gpu hour run where we try to improve the base-line performance of the 18k gpu hour run. here, we test to apply a different learning rate scheduling with a small "warmup-bump" to enable more drastic changes at the start of training, and re-apply the same three stage training recipe to test if training longer (and reiterating on lower resolutions improves training, too)
        - we additionally remove masking here. while masking increases performance of training, has a exponential negative impact on output quality, see [this paper](https://arxiv.org/abs/2407.15811)). let's see if their observation holds up.


### Commands and Helpers 

**Training Command**
this is how a training command looks like:
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage1.py \
--data-path OpenVid-1M.csv \
--ckpt-path pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth
```

let's deconstruct what happens:
- starts with cluster-specific NCCL settings
- TODO: remove unneeded settings, that are not required for production
- TODO: explain the remaining environment variables and why they are chosen that way
- TODO: abstract away specific paths and replace that with placeholders
- `--ckpt-path` vs `--load` vs `--start-from-scratch`
    - all ask for a model checkpoint folder
    - use `--load` if you want to resume training (i.e. when training crashes or is stopped manually)
    - use `--ckpt-path` if you start a new training stage, or if you apply changes to the data set or model, as this resets the data loader and optimizer state
        - note: if you would use `--load` while changing the data set size or number of nodes, this could result in bugs in the data-loader since it's save/load mechanism expects equal settings when saving and when loading
    - use `--load` with the boolean flag `--start-from-scratch` if you want to keep the optimizer state, but if you have applied data set changes or changes in the number of nodes of your nodes. this means that the lr scheduler and optimizer will be continued, but the dataloader will be re-initialized


**Managing Jobs on Bare Metal**  
- **check number of running gpu processes before (re-)starting training**
   the repository contains a small tool running `nvidia-smi` on a list of nodes:

   ```bash
   > python nvtop_all.py nodes.txt
   Node  GPU Processes  Mean Power Consumption (W)
   0  ml-64-node-008              8                   697.76625
   1  ml-64-node-003              8                   698.57875
   2  ml-64-node-005              8                   697.44250
   3  ml-64-node-004              8                   696.90750
   4  ml-64-node-006              8                   696.62000
   5  ml-64-node-007              8                   695.61625
   6  ml-64-node-002              8                   696.28000
   7  ml-64-node-001              8                   695.35500
   ...
   ```
   Oops, seems like an old job is still running.
- **stop all training processes on the cluster**.
    This tool kills all processes on the cluster matching the regex `python.*train\.py` (since this is what our training will use)
    ```bash
    > ./kill_process.sh nodes.txt
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-001
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-002
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-003
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-004
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-005
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-006
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-007
    Sending 'sudo pkill -f python.*train\.py' to ml-64-node-008
    ...
    ```


**Starting the inference server**
When training is running, make sure to start an inference server to monitor the model quality.
We chose to evaluate model performance on a separate machine.
See below for details and how to do start the inference server to log into the same weights and bias run.



### 18k Hour Training

Let's start with the first part, the **18k GPU hour run**.

Key changes to the original three stages were.
- we start to train with OpenVid-1M
- adapt weight decay 
- short learning rate warmup (400 steps)
- we train for 5 epochs per stage (adapt the config if needed)

#### Stage 1
- 5 epochs with [`lambda/stage1.py`{: .filepath}](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage1.py), training mainly on lower resolutions
- we load pixart sigma weights here. note that we did not apply model conversion training here (stage 0), so only the spatial parts of the model are pre-trained. the temporal branches will be initialized randomly. same applies to the other differnces that STDiT has in contrast to PixArt-Sigma (other than that the coarse structure is very similar, which is the reason why we still apply the pre-trained weights from the T2I model)
- to look into the details of the w&b run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/gt4gjgwm)

{% details Click here to show the training command for stage 1. %}
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage1.py \
--data-path OpenVid-1M.csv \
--ckpt-path pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth
```
{% enddetails %}

#### Stage 2

- 5 epochs with [`lambda/stage2.py`{: .filepath}](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage1.py), training mainly on mid resolutions
- we load the checkpoint from Stage 1 (using `--ckpt-path`)
- to look into the details of the w&b run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/7xw6fx7o)

{% details Click here to show the training command for stage 2. %}
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage2.py \
--data-path OpenVid-1M.csv \
--ckpt-path ./outputs_speedrun/008-STDiT3-XL-2/epoch4-global_step2210/
```
{% enddetails %}

#### Stage 3

- 5 epochs with [`lambda/stage3.py`{: .filepath}](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage1.py), training mainly on higher resolutions
- additionally MiraData-330k has been added (for more HQ content)
- we load the checkpoint from Stage 2 (using `--ckpt-path`)
- increased learning rate a bit to 2e-4
- to look into the details of the w&b run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/mxd1zk0o)

{% details Click here to show the training command for stage 3. %}
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage3.py \
--data-path OpenVid-Miradata.csv \
--ckpt-path ./outputs_speedrun/009-STDiT3-XL-2/epoch4-global_step7099
```
{% enddetails %}


### Additional 7k GPU Hours
additional 7k GPU hours to further improve quality
- for an additional improvement, we define 3 more training stages (5, 6, 7)
    - which are mostly copies of the original 3 by Open-Sora team
    - we test if TODO
    - all stages use both datasets (OpenVid and MiraData) at the same time
        - a total of 1.3M video clips
- to summarize the more detailed explanation above, differences of stages 5, 6, 7 are:
    - employ short 1-cycle schedule warmup phase, increases learning rate significantly shortly to allow more significant changes during warmup
    - stage 6 (reduces the learning rate, since we experienced unstable loss curves, see the respective w&b run)
    - in our case the last stage, stage6 crashed, thus for completenes' sake (of this tutorial), we add an additional stage, stage7, to finish training wich is an exact copy of stage6


#### Stage 4
- 3 epochs with [`lambda/stage4.py`{: .filepath}](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage4.py): again training mainly in lower resolutions
- ckpt from stage 3
- short 1cycle Warm-up "bump"
- no masking
- to look into the details of the w&b run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/92dwzcpr)

{% details Click here to show the training command for stage 4. %}
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage4.py \
--data-path OpenVid-Miradata.csv \
--ckpt-path ./outputs_speedrun/010-STDiT3-XL-2/epoch4-global_step14778
```
{% enddetails %}

#### Stage 5
- 1 epoch with [`lambda/stage5.py`{: .filepath}](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage5.py): training on mid resolutions
- ckpt from stage4
- short 1cycle Warm-up "bump"
- no masking
- to look into the details of the w&b run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/8flh231q)

{% details Click here to show the training command for stage 5. %}
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage5.py \
--data-path OpenVid-Miradata.csv \
--ckpt-path ./outputs_speedrun/012-STDiT3-XL-2/epoch3-global_step2100
```
{% enddetails %}



#### Stage 6
- 1 epoch with [`lambda/stage6.py`{: .filepath}](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage6.py): training on mid resolution
- ckpt from stage5
- short 1cycle Warm-up "bump" with reduced learning rate
- no masking
- to look into the details of the w&b run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/w388gmol)

{% details Click here to show the training command for stage 6. %}
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage6.py \
--data-path OpenVid-Miradata.csv \
--ckpt-path ./outputs_speedrun/013-STDiT3-XL-2/epoch1-global_step3300
```
{% enddetails %}



#### Stage 7
- 3 epochs with [`lambda/stage7.py`{: .filepath}](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage7.py): training on high resolutions (this is the same setting as stage6, but we needed to restart due to a crash)
- same config as stage 6
- to look into the details of the w&b run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/5fgm44u5)

Note, we could have used `--load` here as well to "continue" training of stage 6.

{% details Click here to show the training command for stage 7. %}
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
--hostfile hostfile_calvin_31 \
scripts/train.py configs/opensora-v1-2/train/stage7.py \
--data-path OpenVid-Miradata.csv \
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
