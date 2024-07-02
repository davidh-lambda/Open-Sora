from copy import deepcopy
from datetime import timedelta
from pprint import pprint
import time
import numpy as np
import os
import random
from einops import rearrange
import torch.nn.functional as F

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader, save_sample
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, update_ema

import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from diffusion import IDDPM, DPMS


class HuggingFaceColossalAIWrapper:
    def __init__(self, boosted_model, original_model):
        self.boosted_model = boosted_model
        self.original_model = original_model
        self._copy_attributes()
        # TODO: remove modules / weights from original_model to save the space

    def _copy_attributes(self):
        # Copy attributes from the original model that are required by diffusers
        required_attrs = ['config', 'dtype', 'device']
        for attr in required_attrs:
            if hasattr(self.original_model, attr):
                setattr(self, attr, getattr(self.original_model, attr))

    def __getattr__(self, name):
        # Delegate attribute access to the boosted model
        if hasattr(self.boosted_model, name):
            return getattr(self.boosted_model, name)
        elif hasattr(self.original_model, name):
            return getattr(self.original_model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        return self.boosted_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # Delegate forward pass to the boosted model
        return self.boosted_model(*args, **kwargs)



from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from typing import List
class WarmupScheduler(_LRScheduler):
    """Starts with a log space warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to warmup lr in log space until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After warmup_epochs, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs: int, after_scheduler: _LRScheduler, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        self.min_lr  = 1e-7
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                self.finished = True
            return self.after_scheduler.get_lr()

        # log linear
        #return [self.min_lr * ((lr / self.min_lr) ** ((self.last_epoch + 1) / self.warmup_epochs)) for lr in self.base_lrs]

        # cosine warmup
        return [self.min_lr + (lr - self.min_lr) * 0.5 * (1 - torch.cos(torch.tensor((self.last_epoch + 1) / self.warmup_epochs * torch.pi))) for lr in self.base_lrs]

    def step(self, epoch: int = None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super().step(epoch)


class ConstantWarmupLR(WarmupScheduler):
    """Multistep learning rate scheduler with warmup.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        gamma (float, optional): Multiplicative factor of learning rate decay, defaults to 0.1.
        num_steps_per_epoch (int, optional): Number of steps per epoch, defaults to -1.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        factor: float,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        base_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=-1)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List

class OneCycleScheduler(_LRScheduler):
    """Implements the 1-cycle learning rate policy with warmup and cooldown.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle for each parameter group.
        total_steps (int): The total number of steps in the cycle.
        warmup_steps (int): Number of steps to warm up the learning rate.
        cooldown_steps (int): Number of steps to cool down the learning rate.
        final_lr (float): The final learning rate at the end of the cooldown.
        min_lr (float): The minimum learning rate to start with.
        anneal_strategy (str): {'cos', 'linear'} Learning rate annealing strategy.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_lr, warmup_steps, cooldown_steps, final_lr=0.001, min_lr=1e-7, anneal_strategy='cos', last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = 1e6
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.final_lr = final_lr
        self.min_lr = min_lr
        self.anneal_strategy = anneal_strategy

        self.step_size_up = self.warmup_steps
        self.step_size_down = self.cooldown_steps

        self.anneal_func = self._cosine_annealing if self.anneal_strategy == 'cos' else self._linear_annealing

        super(OneCycleScheduler, self).__init__(optimizer, last_epoch)

    def _cosine_annealing(self, step, start_lr, end_lr, step_size):
        cos_out = torch.cos(torch.tensor(math.pi * step / step_size)) + 1
        return end_lr + (start_lr - end_lr) / 2.0 * cos_out

    def _linear_annealing(self, step, start_lr, end_lr, step_size):
        return end_lr + (start_lr - end_lr) * (step / step_size)

    def get_lr(self):
        if self.last_epoch < self.step_size_up:
            # Warm-up phase
            lr = [self.anneal_func(self.last_epoch, self.min_lr, self.max_lr, self.step_size_up) for _ in self.base_lrs]
        elif self.last_epoch < self.step_size_up + self.step_size_down:
            # Cooldown phase
            step = self.last_epoch - self.step_size_up
            lr = [self.anneal_func(step, self.max_lr, self.final_lr, self.step_size_down) for _ in self.base_lrs]
        else:
            # Constant phase
            lr = [self.final_lr for _ in self.base_lrs]
        return lr

    def step(self, epoch=None):
        if self.last_epoch == -1:
            if epoch is None:
                self.last_epoch = 0
            else:
                self.last_epoch = epoch
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




def save_rng_state():
    rng_state = {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }
    return rng_state


def load_rng_state(rng_state):
    torch.set_rng_state(rng_state['torch'])
    if rng_state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
    np.random.set_state(rng_state['numpy'])
    random.setstate(rng_state['random'])


#from mmengine.runner import set_random_seed
def set_seed_custom(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #set_random_seed(seed=seed)


def calculate_weight_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def ensure_parent_directory_exists(file_path):
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


z_log = None
def write_sample(pipe, pipe2, cfg, epoch, exp_dir, global_step, dtype, device):
    prompts = cfg.eval_prompts[dist.get_rank()::dist.get_world_size()]
    if prompts:
        global z_log   
        rng_state = save_rng_state()
        save_dir = os.path.join(
            exp_dir, f"epoch{epoch}-global_step{global_step + 1}"
        )

        with torch.no_grad():
            image_size = cfg.eval_image_size
            num_frames = cfg.eval_num_frames
            fps = cfg.eval_fps
            eval_batch_size = cfg.eval_batch_size

            input_size = (num_frames, *image_size)
            #latent_size = vae.get_latent_size(input_size)
            if z_log is None:
                rng = np.random.default_rng(seed=42)
                #z_log = rng.normal(size=(len(prompts), vae.out_channels, *latent_size))
                z_log = rng.normal(size=(len(prompts), 4, 44, 80)) # TODO
            z = torch.tensor(z_log, device=device, dtype=float).to(dtype=dtype)
            set_seed_custom(42)

            samples = []
            num_timesteps = pipe.scheduler.num_train_timesteps


            # null caption
            null_caption_token = pipe.tokenizer("", max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
            null_caption_embs = pipe.text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]
            null_y = null_caption_embs.repeat(min(eval_batch_size, len(prompts)), 1, 1)[:, None]


            for i in range(0, len(prompts), eval_batch_size):
                batch_z = z[i:i + eval_batch_size]
                shape = batch_z.shape
                batch_prompts = prompts[i:i + eval_batch_size]

                # global noise (image seed)
                noise_vid = torch.randn(shape, device=device, dtype=dtype)

                # image de-noise (animation)
                # 1. perform one step of image diffusion
                # 2. perform video diffusion on the first step (as many frames as needed)
                # 3. complete image diffusion process of all frames / "enhance" each frame using rest of image diffusion
                num_frames = 2

                frame_latents = []
                kwargs = {"height": 360, "width": 600}
                def inject_first_frame(step, timestep, latents):
                    if step != 1:
                        return
                    device = latents.device
                    dtype = latents.dtype
                    # print("inject extract", step, timestep)
                    for _ in range(num_frames):
                        latents = pipe2(batch_prompts, latents=latents.clone().to(device), output_type="latent", return_dict=False, **kwargs)[0]
                        frame_latents.append(latents)
                        latents = torch.tensor(latents).to(device=device, dtype=dtype)

                def inject_continue_frame(frame):
                    frame_latent = frame_latents[frame]
                    def inject_continue(step, timestep, latents):
                        if step != 1: # yeah ... it's a convenience hack
                            return
                        # print("inject continue", step, timestep)
                        latents *= 0
                        latents += frame_latent.to(device=latents.device,dtype=latents.dtype)
                    return inject_continue

                frames = []
                frames.append(pipe(batch_prompts, callback=inject_first_frame, **kwargs))
                for i in range(num_frames):
                    frames.append(pipe(batch_prompts, callback=inject_continue_frame(i), **kwargs))

                # convert PIL.Images back to video numpy arrays [C, T, W, H]
                images = [np.stack([np.array(frame.images[i]) for frame in frames], axis=0).transpose(3,0,1,2) for i in range(len(batch_z))]
                
                # FOR DEBUG ONLY
                # for i in range(num_frames):
                #     frames[i].images[0].save("test%i.png" % i)

                samples += images


            # 4.4. save samples
            # if coordinator.is_master():
            for sample_idx, sample in enumerate(samples):
                id = sample_idx * dist.get_world_size() + dist.get_rank()
                save_path = os.path.join(
                    save_dir, f"sample_{id}"
                )
                ensure_parent_directory_exists(save_path)

                save_sample(
                    sample,
                    fps=fps,
                    save_path=save_path,
                )


        #if back_to_train_model:
        #    model = model.train()
        #if back_to_train_vae:
        #    vae = vae.train()
        #text_encoder.y_embedder = None
        load_rng_state(rng_state)

def is_file_complete(file_path, interval=1, timeout=60):
    previous_size = -1
    elapsed_time = 0
    
    while elapsed_time < timeout:
        if os.path.isfile(file_path):
            current_size = os.path.getsize(file_path)
            if current_size == previous_size:
                return True  # File size hasn't changed, assuming file is complete
            previous_size = current_size
        
        time.sleep(interval)
        elapsed_time += interval
    
    return False

def log_sample(is_master, cfg, epoch, exp_dir, global_step, check_interval=1, size_stable_interval=1):
    if cfg.wandb:
        for sample_idx, prompt in enumerate(cfg.eval_prompts):
            save_dir = os.path.join(
                exp_dir, f"epoch{epoch}-global_step{global_step + 1}"
            )
            save_path = os.path.join(
                save_dir, f"sample_{sample_idx}"
            )
            file_path = os.path.abspath(save_path + ".mp4")
            while not os.path.isfile(file_path):
                time.sleep(check_interval)

            # File exists, now check if it is complete
            if is_file_complete(file_path, interval=size_stable_interval):
                if is_master:
                    wandb.log(
                        {
                            f"eval/prompt_{sample_idx}": wandb.Video(
                                file_path,
                                caption=prompt,
                                format="mp4",
                                fps=cfg.eval_fps,
                            )
                        },
                        step=global_step,
                    )
                    print(f"{file_path} logged")
            else:
                print(f"{file_path} not found, skip logging.")            






def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        print("Training configuration:")
        pprint(cfg._cfg_dict)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            PROJECT=cfg.wandb_project_name
            wandb.init(project=PROJECT, entity=cfg.wandb_project_entity, name=exp_name, config=cfg._cfg_dict)

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info(f"Dataset contains {len(dataset)} samples.")
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    # TODO: use plugin's prepare dataloader
    if cfg.bucket_config is None:
        dataloader = prepare_dataloader(**dataloader_args)
    else:
        dataloader = prepare_variable_dataloader(
            bucket_config=cfg.bucket_config,
            num_bucket_build_workers=cfg.num_bucket_build_workers,
            **dataloader_args,
        )
    if cfg.dataset.type == "VideoTextDataset":
        total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
        logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model

    # You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
    transformer_orig = Transformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
        subfolder='transformer', 
        torch_dtype=dtype,
        use_safetensors=True,
        use_additional_conditions=False,
    )
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        transformer=transformer_orig,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    pipe.transformer.eval()

    # time dit
    # model = build_module(
    #     cfg.model,
    #     MODELS,
    #     # input_size=[None, None],
    #     in_channels=pipe.vae.config.out_channels,
    #     caption_channels=pipe.text_encoder.config.d_model,
    #     model_max_length=pipe.tokenizer.model_max_length
    # )
    transformer = Transformer2DModel.from_config(pipe.transformer.config)
    transformer.to(device)
    transformer.train()
    pipe2 = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        transformer=transformer_orig,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe2.transformer = transformer
    pipe2.to(device)



    scheduler = IDDPM(str(20), learn_sigma=True, pred_sigma=True, snr=False)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        # filter(lambda p: p.requires_grad, pipe.transformer.parameters()),
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    if cfg.load is not None:
        lr_scheduler = None
    else:
        #lr_scheduler = ConstantWarmupLR(optimizer, factor=1, warmup_steps=500, last_epoch=-1)
        lr_scheduler = ConstantWarmupLR(optimizer, factor=1, warmup_steps=1500, last_epoch=-1)
        #lr_scheduler = OneCycleScheduler(optimizer, min_lr=1e-7, max_lr=1e-4, final_lr=1e-5, warmup_steps=1500, cooldown_steps=2500, anneal_strategy='cos')
    
    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(transformer)
        # set_grad_checkpoint(pipe.transformer)
    if cfg.mask_ratios is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    #model, optimizer, _, dataloader, lr_scheduler = booster.boost(
    #    model=model,
    #    optimizer=optimizer,
    #    lr_scheduler=lr_scheduler,
    #    dataloader=dataloader,
    #)
    boosted_transformer, optimizer, _, dataloader, lr_scheduler = booster.boost(
         model=transformer,
         optimizer=optimizer,
         lr_scheduler=lr_scheduler,
         dataloader=dataloader,
    )
    transformer = HuggingFaceColossalAIWrapper(boosted_transformer, transformer)
    # pipe.transformer = HuggingFaceColossalAIWrapper(boosted_transformer, pipe.transformer)
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")
    if cfg.dataset.type == "VariableVideoTextDataset":
        num_steps_per_epoch = dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
    else:
        num_steps_per_epoch = len(dataloader)

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = 0.0
    sampler_to_io = dataloader.batch_sampler if cfg.dataset.type == "VariableVideoTextDataset" else None
    # 6.1. resume training
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            pipe,
            None,
            optimizer,
            None,# lr_scheduler,
            cfg.load,
            sampler=sampler_to_io if not cfg.start_from_scratch else None,
        )
        if not cfg.start_from_scratch:
            start_epoch, start_step, sampler_start_idx = ret
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")

        
        optim_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Overwriting loaded learning rate from {optim_lr} to config lr={cfg.lr}")
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    if cfg.dataset.type == "VideoTextDataset":
        dataloader.sampler.set_start_index(sampler_start_idx)

    # log prompts for pre-training ckpt
    first_global_step = start_epoch * num_steps_per_epoch + start_step

    write_sample(pipe, pipe2, cfg, start_epoch, exp_dir, first_global_step, dtype, device)
    log_sample(coordinator.is_master(), cfg, start_epoch, exp_dir, first_global_step)
    

    # 6.2. training loop
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            iteration_times = []
            for step, batch in pbar:
                start_time = time.time()
                x = batch.pop("video")  # [B, C, T, H, W]
                y = batch.pop("text")
                # Visual and text encoding
                with torch.no_grad():

                    # for debugging: prepare visual inputs
                    #tsize = x.shape[2]
                    #x = rearrange(x, "b c t w h -> (b t) c w h")

                    # choose time frames to optimize
                    num_frames = 2
                    start_index = random.randint(0, x.shape[2] - num_frames)
                    x = x[:, :, start_index:start_index + num_frames, :, :]
                    tsize = x.shape[2]
                    x = rearrange(x, "b c t w h -> (b t) c w h").to(device, dtype)


                    # ----------------------------------------- #
                    # The code snippets below are copied from : # https://github.com/PixArt-alpha/PixArt-sigma/blob/master/train_scripts/train.py
                    # ----------------------------------------- #

                    # encode
                    posterior = pipe.vae.encode(x).latent_dist  # [B, C, H/P, W/P]
                    if True:#config.sample_posterior:
                        z = posterior.sample()
                    else:
                        z = posterior.mode()

                    # TODO: pixart creates only [B, C, 2*h, 2*w] images due to P==2
                    if z.shape[-2] % 2 == 1:
                        z = z[..., :-1, :]
                    if z.shape[-1] % 2 == 1:
                        z = z[..., :, :-1]

                    # t5 features
                    txt_tokens = pipe.tokenizer(
                        y, max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(device)
                    y_embed = pipe.text_encoder(
                        txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                    y_mask = txt_tokens.attention_mask[:, None, None].to(torch.bool)

                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

                # re-arrange and extract frames
                # z = rearrange(z, "(b t) c w h -> b c t w h ", t=tsize)
                z = rearrange(z, "(b t) c w h -> t b c w h ", t=tsize)
                z = z.to(device, dtype)
                shape = z[0].shape

                # 6.1 Prepare micro-conditions. (from https://github.com/huggingface/diffusers/blob/d457beed92e768af6090238962a93c4cf4792e8f/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py#L882)
                if pipe.transformer.config.sample_size == 128:
                    resolution = torch.stack([batch["height"],batch["width"]], 1)#.repeat(tsize, 1, 1).reshape([-1])
                    aspect_ratio = batch["ar"]#.repeat(tsize)
                    added_cond_kwargs["resolution"] = resolution.to(dtype=dtype, device=device)
                    added_cond_kwargs["aspect_ratio"] = aspect_ratio.to(dtype=dtype, device=device)

                # helpers
                model_kwargs_diffusers = dict(encoder_hidden_states=y_embed, encoder_attention_mask=y_mask, added_cond_kwargs=added_cond_kwargs)
                model_kwargs = dict(encoder_attention_mask=y_mask, added_cond_kwargs=added_cond_kwargs)
                model_converter = lambda model: lambda z, timestep, **kwargs: model(z, y_embed, timestep=timestep, return_dict=False, **kwargs)[0]
                t2i_model = model_converter(pipe.transformer)
                t2v_model = model_converter(transformer)

                # diffusion timesteps
                t_vid = torch.randint(0, scheduler.num_timesteps, (shape[0],), device=device)
                t_img = torch.tensor(scheduler.num_timesteps - 1).repeat(shape[0]).to(device)
                #t_img = torch.randint(1, scheduler.num_timesteps, (shape[0],), device=device)

                # global noise (image seed)
                noise_vid = torch.randn(shape, device=device, dtype=dtype)

                # pipe.transformer.train()

                # image noise (animation)
                loss = 0
                # frames_pos = []
                # frames_neg = []
                frame_pos_1 = None
                frame_pos_2 = None
                frame_neg_1 = None
                frame_neg_2 = None
                last_clean_images = None
                clean_images = None
                pos_and_neg = True
                for frame in z:
                    #last_clean_images = clean_images
                    clean_images = frame * pipe.vae_scale_factor
                    noise_frame = torch.randn(shape, device=device, dtype=dtype)
                    for losstype in (["pos", "neg"] if pos_and_neg else ["pos"]):
                        #full_noise_images = (noise_frame + noise_vid)/math.sqrt(2) if losstype == "pos" else noise_vid
                        full_noise_images = noise_frame if losstype == "pos" else noise_vid
                        noisy_images = scheduler.q_sample(clean_images, t_img, noise=full_noise_images).to(device, dtype)
                        #frame_pred = scheduler.q_sample(clean_images, t_img - 1, noise=full_noise_images).to(device, dtype)
                        with torch.no_grad():
                            frame_pred = scheduler.p_sample(t2i_model, noisy_images, t_img, model_kwargs = model_kwargs)["sample"].to(dtype)
                        if losstype == "pos":
                            frame_pos_1 = frame_pos_2
                            frame_pos_2 = frame_pred
                        else:
                            frame_neg_1 = frame_neg_2
                            frame_neg_2 = frame_pred

                    # if frame_pos_1 is not None and frame_pos_2 is not None:
                    #     # TODO: t_img or t_img-1 ?
                    #     t_vid = torch.randint(0, scheduler.num_timesteps, (shape[0],), device=device)
                    #     noisy_video = scheduler.q_sample(frame_pos_1, t_vid, noise=frame_pos_2).to(device, dtype)
                    #     frame_pred_vid = scheduler.p_sample(t2v_model, noisy_video, t_vid, model_kwargs = model_kwargs)["sample"].to(dtype)
                    #     alpha_vid = ((1.0 * t_vid) / (1.0 * scheduler.num_timesteps)).reshape(-1, 1, 1, 1).to(device, dtype)
                    #     frame_mix = last_clean_images * (1-alpha_vid) + (alpha_vid) * clean_images
                    #     loss +=   scheduler.training_losses(t2i_model, frame_mix, t_img - 1, model_kwargs, noise=frame_pred_vid)["loss"].mean()

                    # compute loss
                    if frame_pos_1 is not None and frame_pos_2 is not None:
                        loss +=   scheduler.training_losses(t2v_model, frame_pos_2, t_vid, model_kwargs, noise=frame_pos_1)["loss"].mean()
                    if frame_neg_1 is not None and frame_neg_2 is not None:
                        loss += (-scheduler.training_losses(t2v_model, frame_neg_2, t_vid, model_kwargs, noise=frame_neg_1)["loss"].mean()).abs()

                # Diffusion using pip-diffusers
                #t = torch.randint(0, pipe.scheduler.num_train_timesteps, (z.shape[0],), device=device)
                #noise = torch.randn(z.shape, device=z.device, dtype=z.dtype)
                #noisy_x = pipe.scheduler.add_noise(z, noise, t)
                #noise_pred = pipe.transformer(noisy_x, y_embed, timestep=t, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
                #loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Diffusion using Open-Sora code
                #t = torch.randint(0, scheduler.num_timesteps, (z.shape[0],), device=device)
                #loss_dict = scheduler.training_losses(lambda z,t: pipe.transformer(z, y_embed, timestep=t, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0], z, t)
                #loss = loss_dict["loss"].mean()

                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1
                iteration_times.append(time.time() - start_time)


                # Log to tensorboard
                if coordinator.is_master() and global_step % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    running_loss = 0
                    log_step = 0
                    writer.add_scalar("loss", loss.item(), global_step)

                    weight_norm = calculate_weight_norm(pipe.transformer)

                    if cfg.wandb:
                        wandb.log(
                            {
                                "avg_iteration_time": sum(iteration_times) / len(iteration_times),
                                "iter": global_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "acc_step": acc_step,
                                "lr": optimizer.param_groups[0]["lr"],
                                "weight_norm": weight_norm,
                            },
                            step=global_step,
                        )
                        iteration_times = []

                # Save checkpoint
                if cfg.ckpt_every > 0 and global_step % cfg.ckpt_every == 0 and global_step != 0:
                    save(
                        booster,
                        transformer.boosted_model,
                        None,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        cfg.batch_size,
                        coordinator,
                        exp_dir,
                        None,
                        sampler=sampler_to_io,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )

                    # log prompts for each checkpoints
                if global_step % cfg.eval_steps == 0:
                    write_sample(pipe, pipe2, cfg, epoch, exp_dir, global_step, dtype, device)
                    log_sample(coordinator.is_master(), cfg, epoch, exp_dir, global_step)

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_start_index(0)
        if cfg.dataset.type == "VariableVideoTextDataset":
            dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        start_step = 0


if __name__ == "__main__":
    main()
