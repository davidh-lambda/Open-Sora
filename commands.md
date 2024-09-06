
Datasets:
Stage 1: `data_csvs/webvid10m.csv`
Stage 2: `data_csvs/OpenVid-1M-osora_le50MB.csv`
Stage 3: `data_csvs/OpenVid-Miradata-mix.csv`

Speed Improvements:
- deferred masking https://arxiv.org/pdf/2407.15811 
- precompute T5 + VAE and augment later
- LongLLAVA for true video captioning

Quality improvements of inference:
- EMA reweighing: https://arxiv.org/abs/2312.02696

Re-training v2:
- weight decay 0.01
- large LR (current setup assumes 96 GPUs. scale linearly to match the new number)
- warm-up for first stage (`warmup_steps`), maybe next too if training fails
- stage 4: reduce masking even more

Notes:
- `hostfile_calvin` has to start with the main node (017) !
- `conda activate osora-12b`

Commands:

Install new conda env:
(adapt env-var in the script and then run)
> bash install.sh

Install new nodes:
# 1) add bashrc link shared drv if conda not installed, yet
> `for i in {017..028}; do ssh calvin-training-node-$i bash -c "echo $i; cat ~/ml-1cc/dave/bashrc.condainit >> ~/.bashrc ; ln -s ~/ml-1cc ~/ml-Illinois;" ; done`
# 2) also make sure ~/.cache/huggingface/hub is filled for each new node
> `parallel -j 0 ssh -t calvin-training-node-{} cp -r ~/ml-1cc/dave/opensora_hub_ckpts/* ~/.cache/huggingface/hub/ ::: {005..016}`

Check if a job is running:
> `python nvtop_all.py hostfile_calvin`

Kill current job:
> `bash kill_process.sh hostfile_calvin python`


First Launch (pixart init)
> `NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 --hostfile hostfile_calvin scripts/train.py configs/opensora-v1-2/train/stage1.py --data-path /home/ubuntu/ml-1cc/yunpeng/openvid_1m/data/train/OpenVid-1M-osora_le50MB.csv --ckpt-path pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth`

Typical Lanuch (new stage)
> `NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 --hostfile hostfile_calvin scripts/train.py configs/opensora-v1-2/train/stage2.py --data-path /home/ubuntu/ml-1cc/yunpeng/openvid_1m/data/train/OpenVid-1M-osora_le50MB.csv --ckpt-path ./outputs/592-STDiT3-XL-2/epoch37-global_step33300/`

Typical Launch (while in a stage)
> `NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 --hostfile hostfile_calvin scripts/train.py configs/opensora-v1-2/train/stage2.py --data-path /home/ubuntu/ml-1cc/yunpeng/openvid_1m/data/train/OpenVid-1M-osora_le50MB.csv --load ./outputs/592-STDiT3-XL-2/epoch37-global_step33300/`

In case number of nodes or GPUS change
This resets only the dataloader. W&B reports will start at step0 again.
> `NCCL_P2P_LEVEL=NVL NCCL_NET_GDR_LEVEL=PIX NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_SOCKET_IFNAME=eno1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 colossalai run --nproc_per_node 8 --hostfile hostfile_calvin scripts/train.py configs/opensora-v1-2/train/stage2.py --data-path /home/ubuntu/ml-1cc/yunpeng/openvid_1m/data/train/OpenVid-1M-osora_le50MB.csv --load ./outputs/592-STDiT3-XL-2/epoch37-global_step33300/ --start-from-scratch`

Inference server
1. `watch -n 100 rsync -avzruP --exclude='*/optimizer' cluster-node-017:/home/ubuntu/ml-1cc/dave/Open-Sora-12/outputs/6\* .`
2. `CUDA_VISIBLE_DEVICES=1 python scripts/inference-server.py --preset low4s --expnums 656`
3. (features explained under `--help` argument)


New Dataset:
Filter out FFMPEG errors.
See the example under: `~/ml-1cc/yunpeng/MiraData/clip_folders`
1. create an `all.txt` with all file names
2. assuming we have 24 nodes available for checking, `split -n l/24 all.txt` to create the sub-lists xa ... xz
3. adapt the parameters in `process_parallel.sh` and run it on all nodes
4. this will create .err files along all files in all.txt. if the file is empty, it's good, if it's non-empty there was an error
5. we will filter out all non-zero err files from all.txt using `filter_without_errors.py`

