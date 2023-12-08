from pathlib import Path

from config import *


class Args(BasicArgs):
    root_dir = BasicArgs.root_dir
    if not Path(BasicArgs.root_dir).exists():
        root_dir = "./blob_dir/debug_output/video_sythesis"

    task_name, method_name = BasicArgs.parse_config_name(__file__)
    log_dir = os.path.join(root_dir, task_name, method_name)

    # data
    dataset_cf = "dataset/tiktok_video_dataset.py"
    max_train_samples = None
    max_eval_samples = None
    max_video_len = 1  # L=16
    debug_max_video_len = 1
    img_full_size = (256, 256)
    img_size = (256, 256)
    fps = 5
    data_dir = "./blob_dir/debug_output/video_sythesis/dataset"
    debug_train_yaml = "./blob_dir/debug_output/video_sythesis/dataset/composite/val_TiktokDance-poses.yaml"
    debug_val_yaml = "./blob_dir/debug_output/video_sythesis/dataset/composite/val_TiktokDance-poses.yaml"
    train_yaml = "./blob_dir/debug_output/video_sythesis/dataset/composite_offset/train_TiktokDance-poses-masks-depth.yaml"
    val_yaml = "./blob_dir/debug_output/video_sythesis/dataset/composite_offset/new10val_TiktokDance-poses-masks-depth.yaml"

    tiktok_data_root = "./tiktok_datasets"

    eval_before_train = True
    eval_enc_dec_only = False

    # training
    local_train_batch_size = 8
    local_eval_batch_size = 8
    learning_rate = 2e-5
    null_caption = False
    refer_sdvae = True
    combine_clip_local = True
    combine_use_mask = True
    gradient_accumulate_steps = 1
    unet_unfreeze_type = "all"
    ref_null_caption = False

    # output except for the synthesized image
    outs = [""]

    # max_norm = 1.0
    epochs = 50
    num_workers = 4
    eval_step = 5
    save_step = 5
    drop_text = 1.0  # drop text only activate in args.null_caption, default=1.0
    scale_factor = 0.18215

    pretrained_model_path = os.path.join(
        root_dir, "diffusers/sd-image-variations-diffusers"
    )
    sd15_path = os.path.join(root_dir, "diffusers/stable-diffusion-v1-5-2")
    gradient_checkpointing = True
    enable_xformers_memory_efficient_attention = True
    freeze_unet = True

    # sample
    num_inf_images_per_prompt = 1
    num_inference_steps = 50
    guidance_scale = 7.5

    # others
    seed = 42
    deepspeed = True
    fix_dist_seed = True


args = Args()
