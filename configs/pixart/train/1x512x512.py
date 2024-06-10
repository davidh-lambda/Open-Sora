# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=1,
    frame_interval=3,
    image_size=(512, 512),
)

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="PixArt-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    no_temporal_pos_emb=True,
    from_pretrained="PixArt-XL-2-512x512.pth",
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)
scheduler_inference = dict(
    type="iddpm",
    num_sampling_steps=20,
    cfg_scale=7.0,
    cfg_channel=3,  # or None
)


# Others
seed = 42
outputs = "outputs"
wandb = True
wandb_project_name = "qss_timedit"
wandb_project_entity = "lambdalabs"

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size = 32
lr = 2e-5
grad_clip = 1.0
exp_id = "0_pixart"


eval_prompts = [
        "People eating ice cream and drinkin espresso outside of a cafe on a narrow street in Rome. There are stores along the street selling a variety of wares. One shop sells fruits. Another shop sells vegetables. A third shop sells christmas ornaments. Many people walk along the street.",
]

eval_image_size = (360, 640)
eval_num_frames = 1
eval_fps = 8
eval_batch_size = 2
eval_steps = ckpt_every