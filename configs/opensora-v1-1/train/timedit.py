# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=4,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {  # 13s/it
    "360p": {4: (1.0, 16)},
    #"360p": {4: (1.0, 9)},


    #"360p": {1: (1.0, 128), 16: (1.0, 8), 32: (1.0, 4)},
    # "144p": {1: (1.0, 200), 16: (1.0, 36), 32: (1.0, 18), 64: (1.0, 9), 128: (1.0, 4)},
    # "256": {1: (0.8, 200), 16: (0.5, 22), 32: (0.5, 11), 64: (0.5, 6), 128: (0.8, 4)},
    # "240p": {1: (0.8, 200), 16: (0.5, 22), 32: (0.5, 10), 64: (0.5, 6), 128: (0.5, 3)},
    # "360p": {1: (0.5, 120), 16: (0.5, 9), 32: (0.5, 4), 64: (0.5, 2), 128: (0.5, 1)},
    #"512": {1: (0.5, 120), 16: (0.5, 9), 32: (0.5, 4), 64: (0.5, 2), 128: (0.8, 1)},
    # "512": {16: (1.0, 8), 32: (1.0, 4)},
    # "480p": {1: (0.4, 80), 16: (0.6, 6), 32: (0.6, 3), 64: (0.6, 1), 128: (0.0, None)},
    # "720p": {1: (0.4, 40), 16: (0.6, 3), 32: (0.6, 1), 96: (0.0, None)},
    # "720p": {16: (1.0, 3), 32: (1.0, 1)},
    # "1024": {1: (0.3, 40)},
}
# mask_ratios = {
#     "mask_no": 0.75,
#     "mask_quarter_random": 0.025,
#     "mask_quarter_head": 0.025,
#     "mask_quarter_tail": 0.025,
#     "mask_quarter_head_tail": 0.05,
#     "mask_image_random": 0.025,
#     "mask_image_head": 0.025,
#     "mask_image_tail": 0.025,
#     "mask_image_head_tail": 0.05,
# }

# Define acceleration
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    #from_pretrained="./pretrained_models/PixArt-XL-2-512x512.pth",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    in_channels=4
)
# model = dict(
#     type="PixArt-XL/2",
#     space_scale=1.0,
#     time_scale=1.0,
#     no_temporal_pos_emb=True,
#     from_pretrained="PixArt-XL-2-512x512.pth",
#     enable_flash_attn=True,
#     enable_layernorm_kernel=True,
# )
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)
scheduler_inference = dict(
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=7.0,
)

# Others
seed = 42
outputs = "outputs"
wandb = True

epochs = 1000
log_every = 10
ckpt_every = 500
load = None

batch_size = None
lr = 1e-5
grad_clip = 1.0

eval_prompts = [
        "People eating ice cream and drinkin espresso outside of a cafe on a narrow street in Rome. There are stores along the street selling a variety of wares. One shop sells fruits. Another shop sells vegetables. A third shop sells christmas ornaments. Many people walk along the street.",
        "a serene mountainous landscape under a cloudy sky. A small village with traditional European architecture is nestled in a valley, surrounded by lush green fields. The buildings have pitched roofs and are predominantly white, with some featuring darker roof tiles. The village is situated at the base of steep, forested mountains that rise dramatically from the valley floor. The mountains are partially shrouded in low-lying clouds or mist, adding a sense of mystery to the scene. The overall style of the scene is realistic with a focus on natural scenery, capturing the tranquility and beauty of a rural mountainous region.",
        "an aerial view of a serene coastal scene. Dominating the foreground is a large body of water, its surface a deep blue-green, reflecting the clear sky above. The water is calm, with only a few small waves visible, suggesting a peaceful day. In the middle of the water, there are two boats. One is closer to the viewer, appearing larger and more detailed. It's a white boat, possibly a yacht or a motorboat, with a sleek design. The other boat is smaller and further away, its details less distinct due to the distance. The coastline is rugged and rocky, with patches of greenery interspersed, adding a touch of life to the otherwise stark landscape. The rocks are a mix of brown and gray, contrasting with the vibrant blue of the water. The sky above is a light blue, with no clouds visible, indicating a sunny day.",
        "a stunning scene showing an iconic european castle on a hill. The ancient stone structures rises against the backdrop of a lush green forest. The camera circles the site, ascending gradually, revealing the intricate layout of the ruins. The shot highlights the contrast between the historical architecture and the surrounding natural beauty, moving smoothly and steadily to provide a comprehensive, panoramic view of the castle from every direction."
]

eval_image_size = (360, 600)
eval_num_frames = 4
eval_fps = 8
eval_batch_size = 2
eval_steps = ckpt_every

wandb_project_name = "qss_timedit"
wandb_project_entity = "lambdalabs"

exp_id = "1_timedit"
