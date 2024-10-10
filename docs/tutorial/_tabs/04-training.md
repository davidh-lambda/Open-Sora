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
In this tutorial, we'll primarily follow the multi-stage training recipe of **OpenSora 1.2**. The original approach involves three training stages, with each stage focusing on higher video resolutions than the previous one.
The original recipe involves three stages of training, with the primary distinction of an increase of resolution of the input videos used for training the model from one stage to the next.

**Original OpenSora 1.2 training configuration**:  
In detail, the training stages are defined as follows:
1. **Preparatory Stage**: This initial stage gradually adapts the T2I (Text-to-Image) model's weights from [PixArt-Σ 2K](https://pixart-alpha.github.io/PixArt-sigma-project/) to the proposed Text-to-Video (T2V) architecture named [STDiT](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md#efficiency-in-choosing-the-architecture) (**S**patio-**T**emporal **Di**ffusion **T**ransformer).
2. **Stage One** ([config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage1.py)): Focuses on `240p` and `360p` video resolutions, with video lengths ranging from 2 to 16 seconds.
3. **Stage Two** ([config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage2.py)): Emphasizes `360p` and `480p` video resolutions.
4. **Stage Three** ([config file](https://github.com/hpcaitech/Open-Sora/blob/main/configs/opensora-v1-2/train/stage3.py)): Concentrates on `720p` and `1080p` video resolutions.


Unfortunately, the conversion process used to transform Pixart-Σ 2K weights to STDiT weights lacks both a config and specific commits to reproduce the original results. Although the team outlines the required model code adaptions in their [report notes](https://github.com/hpcaitech/Open-Sora/blob/476b6dc79720e5d9ddfb3cd589680b2308871926/docs/report_03.md#rectified-flow-and-model-adaptation), we will adopt a simpler approach for this tutorial at the cost of a decrease in quality.

## Speed Run
Instead of Open-Sora's 35k H100 GPU hour training run (not counting the weight conversion training time), our approach for this tutorial involves training for approximately half that duration and skipping the preparation stage. We aim to assess the model's capabilities within this reduced budget. Subsequently, we'll invest an additional 7k GPU hours to evaluate whether the model's performance improves. We'll share the intermediate and final outcomes of our runs and examine the two configurations we've experimented with.



### Configuration
For our training setup, we utilized a cluster of 192 H100 GPUs, aiming to closely follow Open-Sora's original approach across the three training stages. However, we made some key adjustments to adapt to our specific requirements.

Firstly, we modified the learning rate to counter the increased batch size, which effectively increases the implicit step size in training. Observing the loss curves, we decided to also reduce the number of warmup steps from `1000` to 400, as we felt that an extensive warmup wasn't necessary.

Additionally, we incorporated weight decay into our training, setting it to a relatively high value of 0.01 based on recommendations from [this paper](https://arxiv.org/abs/2407.15811). This adjustment was made after noticing that adding weight decay led to more significant changes when the model seemed stuck in local minima.


Our tutorial, aiming to mimic a typical research-oriented foundation model training, involves two main parts: a base-line model and then tests to further improve the quality.
- **18k GPU hour run**: Using only small changes to the original three stages that we're confident will improve training (like weight decay adaptation). Since this base-line training is using only half of Open-Sora's training time, and we're basically training from scratch, as the weights aren't converted "correctly" from PixArt to STDiT, we don't expect great results yet here.
- **Additional 7k GPU hour run**: To further improve performance of the base model, we tested a different learning rate schedule with a small "warmup bump" to allow for more drastic changes at the start of training, and re-apply the same three-stage training recipe as before to see if training longer (and reiterating on lower resolutions again) improves training. We remove masking here. While masking increases training performance, it has an exponential negative impact on output quality, as noted in [this paper](https://arxiv.org/abs/2407.15811). We'll see if their observation holds up and the quality increases faster with masking turned off.


### Commands and Helpers 

#### Training Command
Here's how a typical training command looks:
```
NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 \
NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_ENABLE_MONITORING=1 \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=16 \
colossalai run --nproc_per_node 8 \
--hostfile nodes.txt \
scripts/train.py configs/opensora-v1-2/lambda/stage1.py \
--data-path OpenVid-1M.csv \
--ckpt-path pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth
```

TODO: remove NCCL settings or explain?  
TODO: do the same for the other commands below  
{: .todo}

**Breaking Down the Command**
- **Environment Variables**:
    - `TOKENIZERS_PARALLELISM=false`: Disables parallelism in tokenizers to prevent potential deadlocks.
    - `OMP_NUM_THREADS=16`: Sets the number of OpenMP threads to use, which can improve performance by limiting the number of threads per process.
- **Colossal-AI Command**:
    - `colossalai run`: Command to run the training script with [Colossal-AI's distributed training](https://colossalai.org/docs/concepts/colossalai_overview), essentially a wrapper around `torch.distributed` with improved parallelization and memory management.
    - `--nproc_per_node 8`: Specifies the number of processes (GPUs) to run per node. Adjust this according to your cluster's configuration.
    - `--hostfile <HOSTFILE>`: Specifies the file containing the list of hostnames or IP addresses of the nodes in the cluster. In the [setup section](../02-setup.md) we have named the hostfile `nodes.txt`{: .filepath}.
- **Training Script Arguments**:
    - `scripts/train.py`: The training script to run.
    - `<CONFIG_PATH>`: Path to the configuration file for the current training stage.
    - `--data-path <DATA_PATH>`: Path to the dataset CSV file.
    - `--ckpt-path <CHECKPOINT_PATH>`: Path to the checkpoint file from which to load model weights.
- **Checkpoint Loading Comes in Three Variants**:
    - `--ckpt-path`: This loads only the model weights, so effectively it resets the data loader and optimizer state. Use this when you want to start a new training stage or if you've applied changes to the dataset or model. 
    - `--load`: This loads all the data loader state, the learning rate scheduler state, the model weights and the optimizer state. Use this to resume training, for instance after a crash or manual stop has happened. 
    - `--load` with `--start-from-scratch`: This re-initializes the data loader while resuming the model, optimizer and scheduler states. Use this if you want to keep the optimizer state but have applied changes to the dataset or number of nodes. 


#### Managing Jobs on Bare Metal
When training on a bare-metal cluster, it's essential to manage jobs effectively.

- **Check Running GPU Processes Before Starting Training**:  
  The repository contains a small tool, `nvtop_all.py`, that runs `nvidia-smi` all nodes contained in the hostfile.  
  ```bash
  python nvtop_all.py <HOSTFILE>
  ```
  This helps you ensure that no previous jobs are still running.

  This is how its output looks like.  
  (Seems like an old job is still running on our cluster.)
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
- **Stop All Training Processes on the Cluster**:  
  To stop all training processes matching the regex `python.*train\.py`, you can use the `kill_process.sh` script:
  ```bash
  ./kill_process.sh <HOSTFILE>
  ```

  Its output looks as follows:
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



#### Launching the Inference Server
To effectively monitor a diffusion model during training, it is essential inspect the model quality recurrently. We recommend assessing the model's performance on a separate machine. While typical training scripts perform evaluation concurrently with training, such as every few hundred steps, it is more practical to delegate this task to a dedicated process for Text-to-Video (T2V) models. This approach ensures that the relatively slow evaluation process does not hinder the training progress of the whole cluster.
Read the details below on how to start the inference server and log into the same Weights & Biases (W&B) run.




### 18k Hour Training
With our initial budget, we aim to replicate the core aspects of Open-Sora's training recipe at about half the training time of the original model.  
Let's start training with the first part, the **18k GPU hour run**, with only minor adjustments:
**Key Changes to the Original Three Stages**
- **Dataset**: We start training with **OpenVid-1M**.
- **Weight Decay**: Adapted weight decay to `0.01`.
- **Warmup Steps**: Shortened learning rate warmup to 400 steps.
- **Epochs**: We train for 5 epochs per stage (adjust the config if needed).


#### Stage 1
- **Config**: [`lambda/stage1.py`](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage1.py)
- **Details**: 5 epochs, mainly on lower resolutions.
    - We load PixArt Sigma weights here. Note that we didn't apply model conversion training (Stage 0), so only the spatial parts of the model are pre-trained and the temporal branches are initialized randomly. However, since the overall structure of the model is similar, we still use the pre-trained weights from the T2I model.
    - To view the details of the W&B run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/gt4gjgwm).
- **Training Command**:
    ```bash
    NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
    --hostfile nodes.txt \
    scripts/train.py configs/opensora-v1-2/lambda/stage1.py \
    --data-path OpenVid-1M.csv \
    --ckpt-path pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth
    ```

TODO: share checkpoints after every stage?
{: .todo}

**Results**


<div id="iframe1-button" style="width: 100%; height: 150px; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; cursor: pointer;">
    <p>
        <a href="https://wandb.ai/lambdalabs/sora_speedrun/reports/Speed-Run-Stage-1--Vmlldzo5NjcyMjQ0" target="_blank">Click to load the Weights & Bias report</a>
    </p>
</div>
<iframe
  id="iframe1"
  width="100%" 
  height="0" 
  frameborder="0"
  allowfullscreen>
</iframe>

#### Stage 2
- **Config**: [`lambda/stage2.py`](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage2.py)
- **Details**: 5 epochs, mainly on mid resolutions.
  - We load the checkpoint from Stage 1 using `--ckpt-path`.
  - To view the details of the W&B run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/7xw6fx7o).
- **Training Command**:
    ```bash
    NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
    --hostfile nodes.txt \
    scripts/train.py configs/opensora-v1-2/lambda/stage2.py \
    --data-path OpenVid-1M.csv \
    --ckpt-path ./outputs_speedrun/008-STDiT3-XL-2/epoch4-global_step2210/
    ```

**Results**
<div id="iframe2-button" style="width: 100%; height: 150px; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; cursor: pointer;">
    <p>
        <a href="https://wandb.ai/lambdalabs/sora_speedrun/reports/Speed-Run-Stage-2--Vmlldzo5NjcyNDQ2" target="_blank">Click to load the Weights & Bias report</a>
    </p>
</div>
<iframe
  id="iframe2"
  width="100%" 
  height="0" 
  frameborder="0"
  allowfullscreen>
</iframe>

#### Stage 3
- **Config**: [`lambda/stage3.py`](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage3.py)
- **Details**: 5 epochs, mainly on higher resolutions.
  - Adds the **MiraData-330k** dataset to provide a larger selection of high-resolution video clips.
  - We load the checkpoint from Stage 2 using `--ckpt-path`.
  - Increased the learning rate slightly, to `2e-4`.
  - To view the details of the W&B run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/mxd1zk0o).
- **Training Command**:
    ```bash
    NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
    --hostfile nodes.txt \
    scripts/train.py configs/opensora-v1-2/lambda/stage3.py \
    --data-path OpenVid-Miradata.csv \
    --ckpt-path ./outputs_speedrun/009-STDiT3-XL-2/epoch4-global_step7099
    ```

**Results**
<div id="iframe3-button" style="width: 100%; height: 150px; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; cursor: pointer;">
    <p>
        <a href="https://wandb.ai/lambdalabs/sora_speedrun/reports/Speed-Run-Stage-3--Vmlldzo5NjcyNDUw" target="_blank">Click to load the Weights & Bias report</a>
    </p>
</div>
<iframe
  id="iframe3"
  width="100%" 
  height="0" 
  frameborder="0"
  allowfullscreen>
</iframe>

### Additional 7k GPU Hours
To further enhance quality, we invested an additional **7k GPU hours** and defined three more training stages (Stages 4, 5, and 6). These stages are mostly copies of the original three used above for the 18k GPU hours run.

**Key Differences in Additional Stages**
In this phase, we decided to experiment with several key modifications to explore their impact on our results. We used both datasets, OpenVid and MiraData, simultaneously, totaling 1.3 million video clips, to provide the model with more diverse data. We adopted a different learning rate schedule with a small "warmup bump" to encourage more significant changes at the start of training. Additionally, we removed masking, as it's been shown to negatively affect output quality according to [this paper](https://arxiv.org/abs/2407.15811). By eliminating masking, we wanted to see if the trade-off between faster training times and potential improvements in output quality would benefit our application.


#### Stage 4
- **Config**: [`lambda/stage4.py`](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage4.py)
- **Details**: 3 epochs, mainly on lower resolutions.
  - We load the checkpoint from Stage 3 using `--ckpt-path`.
  - Warm-Up using a short 1-cycle warmup "bump".
  - Disabled masking
  - To view the details of the W&B run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/92dwzcpr).
- **Training Command for Stage 4**
    ```bash
    NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
    --hostfile nodes.txt \
    scripts/train.py configs/opensora-v1-2/lambda/stage4.py \
    --data-path OpenVid-Miradata.csv \
    --ckpt-path ./outputs_speedrun/010-STDiT3-XL-2/epoch4-global_step14778
    ```

**Results**
<div id="iframe4-button" style="width: 100%; height: 150px; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; cursor: pointer;">
    <p>
        <a href="https://wandb.ai/lambdalabs/sora_speedrun/reports/Speed-Run-Stage-4--Vmlldzo5NjcyNDcx" target="_blank">Click to load the Weights & Bias report</a>
    </p>
</div>
<iframe
  id="iframe4"
  srcThen=""
  width="100%" 
  height="1200" 
  frameborder="0"
  allowfullscreen>
</iframe>

#### Stage 5
- **Config**: [`lambda/stage5.py`](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage5.py)
- **Details**: 1 epoch, on mid resolutions.
  - We load the checkpoint from Stage 4 using `--ckpt-path`.
  - Warm-Up using a short 1-cycle warmup "bump".
  - Disabled masking
  - To view the details of the W&B run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/8flh231q).
- **Training Command for Stage 5**
    ```bash
    NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
    --hostfile nodes.txt \
    scripts/train.py configs/opensora-v1-2/lambda/stage5.py \
    --data-path OpenVid-Miradata.csv \
    --ckpt-path ./outputs_speedrun/012-STDiT3-XL-2/epoch3-global_step2100
    ```

**Results**
<div id="iframe5-button" style="width: 100%; height: 150px; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; cursor: pointer;">
    <p>
        <a href="https://wandb.ai/lambdalabs/sora_speedrun/reports/Speed-Run-Stage-5--Vmlldzo5NjcyNDc4" target="_blank">Click to load the Weights & Bias report</a>
    </p>
</div>
<iframe
  id="iframe5"
  width="100%" 
  height="0" 
  frameborder="0"
  allowfullscreen>
</iframe>


#### Stage 6

- **Config**: [`lambda/stage6.py`](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage6.py)
- **Details**: 1 epoch, on high resolutions.
  - We load the checkpoint from Stage 4 using `--ckpt-path`.
  - Warm-Up using a short 1-cycle warmup "bump".
  - Disabled masking
  - **Note**: Stage 6 crashed. For completeness, we added Stage 7 to finish training.
  - To view the details of the W&B run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/w388gmol).
- **Training Command for Stage 6**
    ```bash
    NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
    --hostfile nodes.txt \
    scripts/train.py configs/opensora-v1-2/lambda/stage6.py \
    --data-path OpenVid-Miradata.csv \
    --ckpt-path ./outputs_speedrun/013-STDiT3-XL-2/epoch1-global_step3300
    ```

**Results**
<div id="iframe6-button" style="width: 100%; height: 150px; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; cursor: pointer;">
    <p>
        <a href="https://wandb.ai/lambdalabs/sora_speedrun/reports/Speed-Run-Stage-6--Vmlldzo5NjcyNTk3" target="_blank">Click to load the Weights & Bias report</a>
    </p>
</div>
<iframe
  id="iframe6"
  width="100%" 
  height="0" 
  frameborder="0"
  allowfullscreen>
</iframe>


#### Stage 7
- **Config**: [`lambda/stage7.py`](https://github.com/LambdaLabsML/Open-Sora/blob/main/configs/opensora-v1-2/lambda/stage7.py)
- **Details**: 3 epochs, on high resolutions.
  - Same config as Stage 6.
    > Note, we could have used `--load` here as well to "continue" training of stage 6 without requiring to re-warmup.
    {: .prompt-tip}
  - To view the details of the W&B run, click [here](https://wandb.ai/lambdalabs/sora_speedrun/runs/5fgm44u5).
- **Training Command for Stage 7**
    ```bash
    NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 \
    --hostfile hostfile_calvin_31 \
    scripts/train.py configs/opensora-v1-2/train/stage7.py \
    --data-path OpenVid-Miradata.csv \
    --ckpt-path ./outputs_speedrun/015-STDiT3-XL-2/epoch0-global_step2900
    ```

**Results**
<div id="iframe7-button" style="width: 100%; height: 150px; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; cursor: pointer;">
    <p>
        <a href="https://wandb.ai/lambdalabs/sora_speedrun/reports/Speed-Run-Stage-7--Vmlldzo5NjcyNjEy" target="_blank">Click to load the Weights & Bias report</a>
    </p>
</div>
<iframe
  id="iframe7"
  width="100%" 
  height="0" 
  frameborder="0"
  allowfullscreen>
</iframe>






## Monitoring Model Quality

While tracking loss curves in [Weights & Biases](https://wandb.com) provides valuable insights during training, the loss values often plateau after the initial few epochs. This makes it essential to evaluate the model beyond numerical metrics by assessing the quality of videos generated from a set of validation prompts. To do this, we need to run the most recent model weights in inference mode regularly.

However, running inference on a Text-to-Video (T2V) model is computationally expensive. For example, generating videos at 720p resolution, 4 seconds in duration, with a batch size of 2 can takes already X minutes on a H100 — we don't want the entire cluster to idle while waiting for some nodes to finish evaluation!

To address this, we set up a separate, smaller server to handle inference asynchronously. This allows the main training process to continue uninterrupted, maximizing our computational resources. The inference server runs the latest model checkpoints, generates sample videos, and saves the outputs to the same Weights & Biase runs that training is logging to, as you've seen in the result sections above.


### Setting Up the Inference Server

The inference server in this repository is designed to work asynchronously and supports several modes. Here's how you can set it up:
1. **Synchronize Checkpoints**  
   First, we need to synchronize the latest checkpoints from the training cluster to the inference server. If you don't have acccess to your shared storage on this dedicated inference machine, you can use `rsync` for this purpose to query the checkpoints regularly.

   ```bash
   watch -n 100 rsync -avzruP --exclude='*/optimizer' training-cluster:/path/to/your/training/outputs/* .
   ```

   This command runs every 100 seconds and synchronizes new checkpoints, excluding optimizer states to save bandwidth and storage.
2. **Initialize the Node & Log In into W&B**:  
    Ensure that both W&B and Open-Sora are properly initialized and functioning on this node. If the node is not included in the cluster where you have previously completed the setup, please refer to the instructions provided in the [Setup](../02-setup.md) section.
3. **Run the Inference Server**  
   Next, we start the inference server using the desired preset and experiment numbers:

   ```bash
   python scripts/inference-server.py --preset low4s --expnums 656
   ```

   - `--preset`: Specifies the inference settings. Available options include `low4s`, `high4s`, and `low16s`, which correspond to different resolutions and durations.
   - `--expnums`: Specifies the experiment numbers or checkpoint directories to monitor and execute inference on. The experiment number is used to automatically extract the W&B run-id, enabling the inference server to identify the location to push the results to.

   You can explore additional options and features by running:

   ```bash
   python scripts/inference-server.py --help
   ```

   For instance, you can run a second server that computes the `720p` results.
   ```bash
   python scripts/inference-server.py --preset high4s --expnums 656
   ```

By setting up the inference server this way, we can continuously monitor the model`s output quality without interrupting the training process. This approach ensures that our valuable training resources remain focused on model optimization, while inference and evaluation happen in parallel.




## Monitoring Cluster Health

While your training is running, it's crucial to keep an eye on the health of your cluster. We use an internal tool to monitor cluster performance, regularly checking for any signs of degrading performance. This tool logs metrics such as power draw across the entire cluster and InfiniBand or Ethernet speeds.

As highlighted in [the LLama 3 Paper](https://arxiv.org/abs/2407.21783), large-scale distributed training can often face downtime. We too experienced this firsthand during our training runs. If you're interested in learning more about what we learned when our training failed recurrently, be sure to check out the [Lessons](../05-lessions.md) Section later in this tutorial.


add screenshots if possible
{: .todo}



<br/>

---

**What Next?**:  
Now that we've covered the training process and how to monitor both model quality and cluster health, it's time to dive into the lessons we learned during our reproduction experiment. In the [next section](../05-lessons.md), we'll share insights on various challenges we faced—from finding data loader bugs that led to diverging training, to debugging issues in worker code that appeared randomly across the cluster, and tackling low-level problems with NCCL on a bare-metal setup.

By exploring these experiences, you'll be better prepared to address similar challenges in your own work.



---



<script>
  function loadIframe(buttonId, iframeId) {
    const button = document.getElementById(buttonId);
    const iframe = document.getElementById(iframeId);
    const link = button.querySelector('a');

    button.addEventListener('click', function() {
      iframe.src = link.getAttribute('href');
      iframe.style.height = '1200px';
      button.style.display = 'none';
    });
  }
  function disableLinks(buttonId) {
    const button = document.getElementById(buttonId);
    const link = button.querySelector('a');
    
    if (link) {
      link.style.pointerEvents = 'none';
      link.style.color = 'inherit';
      link.style.textDecoration = 'none';
    }
  }

  for (let i = 1; i <= 7; i++) {
    const buttonId = `iframe${i}-button`;
    const iframeId = `iframe${i}`;
    loadIframe(buttonId, iframeId);
    disableLinks(buttonId);
  }
</script>