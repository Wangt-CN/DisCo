from utils.lib import *
from utils.dist import dist_init
from config import BasicArgs
from utils.wutils_ldm import import_filename


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object,
    # enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def parse_with_cf(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object,
    # enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if os.path.exists(parsed_args.cf):
        cf = import_filename(parsed_args.cf)
        config_args = edict(vars(cf.Args))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    else:
        raise NotImplementedError('Config filename %s does not exist.' % args.cf)
    return args


def update_args(parsed_args, args):
    for key in vars(parsed_args):
        val = getattr(parsed_args, key)
        if key in ["epochs", "train_batch_size", "eval_batch_size", "train_yaml", "val_yaml", "learning_rate"]:
            if val is None:
                continue
        if key == "log_dir":
            if val is None:
                continue
            else:
                val = os.path.join(BasicArgs.root_dir, parsed_args.log_dir)
        if key == "root_dir":
            continue
        setattr(args, key, val)
    return args


class Args(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self, desc="shared config"):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('--root_dir', default=None, type=str)
        parser.add_argument('--cf', default=None, type=str, required=True)
        parser.add_argument('--pretrained_model',  default=None, type=str)
        parser.add_argument('--pretrained_model_lora',  default=None, type=str)
        parser.add_argument('--pretrained_model_controlnet', default=None, type=str)
        parser.add_argument('--debug', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--debug_seed', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--debug_dataloader', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--log_dir', default=None, type=str)
        parser.add_argument('--deepspeed',
                            help="use deepspeed",  type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--use_amp', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization.")
        parser.add_argument('--fix_dist_seed', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--tiktok_data_root', default=None, type=str)
        # parser.add_argument('--tiktok_ann_root', default=None, type=str)

        # input
        parser.add_argument("--img_size", default=256, type=int,
                            help="image input size")
        parser.add_argument("--max_video_len", default=4, type=int,
                            help="frame input length")
        parser.add_argument("--debug_max_video_len", default=4, type=int,
                            help="debug frame input length")
        parser.add_argument("--conds", default=[], #
                            nargs='+', type=str,
                            choices=["poses", "masks", "densepose", "hed", "canny_100_200", "midas", "mlsd_0.1_0.1", "uniformer"],
                            help="used in uni-controlnet/tsv datasets")

        # Model setting
        parser.add_argument('--gradient_checkpointing', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--find_unused_parameters', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--enable_xformers_memory_efficient_attention', type=str_to_bool,
                            nargs='?', const=True, default=False)

        # SD related
        parser.add_argument(
            "--trainable_modules", type=str, nargs='+', default=None)
        parser.add_argument("--scale_factor", default=0.18215, type=float,
                            help="scale factor in SD")
        parser.add_argument("--loss_target", default="noise",  type=str,
                            choices=["noise", "x0", "mixed"],
                            help="unet_loss_target")
        parser.add_argument("--x0_steps", default=200,  type=int,
                            help="steps to calc x0 loss")
        parser.add_argument('--pretrained_model_path', default='diffusers/stable-diffusion-v1-4', type=str)

        # training configs
        parser.add_argument("--num_workers", default=4,  type=int,
                            help="number of  workers")
        parser.add_argument('--node_split_sampler',
                            help="use node_split_sampler",  type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--gradient_accumulate_steps', default=1, type=int)
        parser.add_argument('--max_grad_norm', default=-1, type=float)
        parser.add_argument('--learning_rate',  default=5e-6, type=float)
        parser.add_argument("--decay", default=1e-3, type=float,
                            help="Weight deay.")
        parser.add_argument("--warmup_ratio", default=0.1, type=float,
                            help="warm up ratio of lr")
        parser.add_argument("--max_train_samples", default=None, type=int,
                            help="number train samples")
        parser.add_argument("--debug_max_train_samples", default=100, type=int,
                            help="number train samples in debug mode")
        parser.add_argument("--drop_text", default=1.0, type=float, # drop text only activate in args.null_caption
                            help="prob to drop text")
        # Note that tbs/ebs is the Global batch size = GPU_PER_NODE * NODE_COUNT * LOCAL_BATCH_SIZE
        parser.add_argument('--local_train_batch_size', default=1, type=int)
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--eval_step', default=5, type=float)
        parser.add_argument('--save_step', default=5, type=float)
        parser.add_argument('--do_train', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--train_yaml', default=None, type=str)
        parser.add_argument('--resume',  type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--null_caption',  type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--refer_sdvae',  type=str_to_bool, # use sd vae to process the reference image
                            nargs='?', const=True, default=False)
        parser.add_argument('--controlnet_conditioning_scale_cond', default=1.0, type=float)
        parser.add_argument('--controlnet_conditioning_scale_ref', default=1.0, type=float)

        ### for temporal disco
        parser.add_argument("--nframes", type=int, default=8, help="the number of frames for synthesis")
        parser.add_argument("--frame_interval", type=int, default=1, help="frame interval for synthesis")
        parser.add_argument("--eval_sample_interval", type=float, default=1, help="eval interval for fast evalaution, only valid for video datasets")
        parser.add_argument("--train_sample_interval", type=float, default=1, help="train interval for fast training") 
        
        ### reference image attention
        parser.add_argument("--unet_unfreeze_type", default=None,  type=str, # if set --freeze_unet=False, will ft all the unet
                            choices=["crossattn-kv", "crossattn", "transblocks", "all", "null"],
                            help="which structure of unet will be unfreezed")
        parser.add_argument('--controlnet_attn', type=str_to_bool,
                            nargs='?', const=True, default=False, help="if use ref image to replace the text prompt in controlnet")
        parser.add_argument('--use_cfg', type=str_to_bool,
                            nargs='?', const=True, default=False, help="if use classifier free guidance for image")

        ### reference image attention with clip
        parser.add_argument('--refer_clip_preprocess',  type=str_to_bool,
                            nargs='?', const=True, default=False, help='if use clip preprocess in diffusers to process the reference image')
        parser.add_argument('--refer_clip_proj',  type=str_to_bool,
                            nargs='?', const=True, default=False, help='if use pretrained clip visual projection layer for the reference image')

        ### reference image attention with clip + combine controlnet path
        parser.add_argument('--ref_null_caption',  type=str_to_bool, # use null caption for the reference controlnet path
                            nargs='?', const=True, default=False)
        parser.add_argument('--combine_clip_local',  type=str_to_bool,
                            nargs='?', const=True, default=False, help='if use local clip feature in combine controlnet path (a bit messey here)')
        parser.add_argument('--combine_use_mask',  type=str_to_bool,
                            nargs='?', const=True, default=False, help='if add mask annotation to the (attn + controlnet) structure; default: attn human pose; controlnet background')
        parser.add_argument("--drop_ref", default=0., type=float, # drop the reference image during trianing?
                            help="prob to drop reference image")
        parser.add_argument('--my_adapter', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if use my adapter?')

        ### no crop in training, resize after generation (post resize)
        parser.add_argument('--pos_resize_img', type=str_to_bool,
                            nargs='?', const=True, default=False, help='resize img to the targe size after generation?')
        parser.add_argument("--fg_variation", default=0., type=float,
                            help="if add foreground variation during training")
        parser.add_argument('--strong_aug_stage2', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if use strong aug in stage 1 for warm up?')
        parser.add_argument('--strong_rand_stage2', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if use strong rand in stage 2 for warm up?')

        ### stage 1, warm up with training attribute
        parser.add_argument('--strong_aug_stage1', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if use strong aug in stage 1 for warm up?')
        parser.add_argument('--stage1_pretrain_path', default=None, type=str,
                            help='if use stage 1 attribute pretraining to initialize the model (unet + ref controlnet')
        parser.add_argument('--stage2_only_pose', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if only train pose controlnet in the 2nd stage')
        parser.add_argument('--constant_lr', type=str_to_bool,
                            nargs='?', const=True, default=False, help='no linear lr decay?')


        ### SD-2-1 args
        parser.add_argument('--SD2_not_add_image_emb_noise', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if not add noise to image embedding? default add noise')
        # evaluation configs
        parser.add_argument('--val_yaml', default=None, type=str)
        parser.add_argument("--max_eval_samples", default=None, type=int,
                            help="number eval samples")
        parser.add_argument("--debug_max_eval_samples", default=20, type=int,
                            help="number eval samples")
        parser.add_argument('--pose_normalize', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--normalize_by_1st_frm', type=str_to_bool,
                            nargs='?', const=True, default=False)
        # Note that tbs/ebs is the Global batch size = GPU_PER_NODE * NODE_COUNT * LOCAL_BATCH_SIZE
        parser.add_argument('--local_eval_batch_size', default=1, type=int)
        parser.add_argument('--eval_visu', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--eval_visu_trainsample', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--eval_visu_imagefolder', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--eval_visu_changepose', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if change pose from the other data source?')
        parser.add_argument('--eval_visu_changefore', type=str_to_bool,
                            nargs='?', const=True, default=False, help='if change foreground from the other data source?')

        parser.add_argument('--eval_save_filename', default='eval_visu', type=str)
        parser.add_argument('--eval_before_train', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--eval_scheduler', default="ddim",  type=str,
                            choices=["pndms", "ddpm", 'ddim'],
                            help="frame interpolation mode")
        parser.add_argument('--eval_enc_dec_only', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--num_inf_videos_per_prompt', default=1, type=int)
        parser.add_argument('--num_inference_steps', default=50, type=int)
        parser.add_argument('--guidance_scale', default=3, type=float)
        parser.add_argument('--stepwise_sample_depth',default=-1, type=int)
        parser.add_argument('--interpolation', default=None,  type=str,
                            choices=["copy", "average", 'interpolate', 
                                    'average_noise', None],
                            help="frame interpolation mode")
        parser.add_argument('--interpolate_mode',default=None,  type=str,
                            choices=["nearest", "bilinear", 'trilinear', 'area', 'nearest-exact', None],
                            help="frame interpolation mode")

        ### save the visualization file?
        parser.add_argument('--visu_save', type=str_to_bool, # if True, means save the visualization file with each filename
                            nargs='?', const=True, default=False)

        ### stage3 ft on specific images
        parser.add_argument('--freeze_pose', type=str_to_bool,
                            nargs='?', const=True, default=False, help='freeze the pose path?')
        parser.add_argument('--freeze_background', type=str_to_bool,
                            nargs='?', const=True, default=False, help='freeze the background path?')
        parser.add_argument('--ft_img_num', default=0, type=int, help='use how many frame as the training sample? 0 denote all')
        parser.add_argument('--ft_one_ref_image', type=str_to_bool,
                            nargs='?', const=True, default=True, help='only use first frame as the ref image?')
        parser.add_argument('--ft_iters', default=None, type=int)


        ### dreampose baseline ###
        parser.add_argument("--s1", default=1.0, type=float, # the param s1 for dreampose
                            help="the param s1 for dreampose")
        parser.add_argument("--s2", default=1.0, type=float, # the param s2 for dreampose
                            help="the param s2 for dreampose")

        ### stage 3 ft ###
        parser.add_argument("--ft_idx", default=None,  type=str, # if set --freeze_unet=False, will ft all the unet
                            help="ft video idx from web folder")

        ### stage 2 ref mode ###
        parser.add_argument("--ref_mode", default='first',  type=str,
                            help="ref mode")
        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        if parsed_args.root_dir:
            BasicArgs.root_dir = parsed_args.root_dir
        else:
            parsed_args.root_dir = BasicArgs.root_dir
        parsed_args.pretrained_model_path = os.path.join(parsed_args.root_dir, parsed_args.pretrained_model_path)
        args = parse_with_cf(parsed_args)
        if args.debug:
            args.max_video_len = getattr(args, 'debug_max_video_len', 1)
            args.max_train_samples = getattr(args, 'debug_max_train_samples', 100)
            args.max_eval_samples = getattr(args, 'debug_max_eval_samples', 20)
            args.num_workers = 0
            args.epochs = 2
            args.eval_step = 1
            args.save_step = 1
            if hasattr(args, 'debug_train_yaml'):
                args.train_yaml = args.debug_train_yaml
            if hasattr(args, 'debug_val_yaml'):
                args.val_yaml = args.debug_val_yaml
        
        args.n_gpu = T.cuda.device_count() # local size
        args.local_size = args.n_gpu
        if args.root_dir not in args.log_dir:
            args.log_dir = os.path.join(args.root_dir, args.log_dir)

        # if len(args.conds) > 0:
        #     args.log_dir += "_".join(args.conds)
        
        # args.wandb_project = [
        #     p_.replace(".py", "")
        #     for p_ in args.cf.split("/")
        #     if p_ != "config" and p_ != "."]
        # args.wandb_project = [args.method_name] + args.log_dir.split('/')[-2:]
        args.wandb_project = args.log_dir.split('/')[-2:]
        # args.project_name = "/".join(
        #     args.wandb_project)
        args.project_name = args.wandb_project[1]
        # args.wandb_project = args.task_name #args.wandb_project[0]
        args.wandb_project = args.wandb_project[0]

        if args.stepwise_sample_depth == -1:
            args.interpolation = None
            args.interpolate_mode = None
        if args.interpolation != "interpolate":
            args.interpolate_mode = None
        
        assert args.eval_step > 0, "eval_step must be positive"
        assert args.save_step > 0, "save_step must be positive"

        dist_init(args)
        args.dist = args.distributed
        args.nodes = args.num_nodes
        args.world_size = args.num_gpus
        args.train_batch_size = args.local_train_batch_size * args.world_size
        args.eval_batch_size = args.local_eval_batch_size * args.world_size
        return args


sharedArgs = Args()


# def get_args(distributed=True):
#     args = sharedArgs.parse_args()
#     dist_init(args, distributed)

#     if not args.distributed:
#         args.deepspeed = False

#     args.effective_batch_size = args.size_batch * args.num_gpus
#     if os.path.exists(args.path_ckpt):
#         path_ckpt_dir = os.path.dirname(args.path_ckpt)
#         training_args = f"{path_ckpt_dir}/args.json"
#         if os.path.exists(training_args):
#             args = update_args(args)
#     return args


# def update_args(args):
#     path_ckpt_dir = os.path.dirname(args.path_ckpt)
#     training_args = edict(json.load(open(f"{path_ckpt_dir}/args.json", "r")))

#     print("===============Loaded model training args=================")
#     print(f"\t\t{json.dumps(training_args)}")
#     print("===============Default args=================")
#     print(f"\t\t{json.dumps(args)}")
#     toUpdate = [
#         "vis_backbone", "vis_backbone_size", "temporal_fusion",
#         "imagenet", "kinetics", "swinbert",
#         "txt_backbone", "fusion_encoder",
#         "txt_backbone_embed_only", "tokenizer", "mask_pos", "fuse_type",
#         "num_fuse_block_t2i", "num_fuse_block_i2t"]
#     if args.size_epoch == 0:
#         toUpdate += ['size_frame', 'size_txt', 'size_img', 'img_transform']
#     args.imagenet_norm = False
#     for key in training_args:
#         if key == "imagenet_norm":
#             args.imagenet_norm = training_args.imagenet_norm
#         if key in toUpdate:
#             args[key] = training_args[key]
#         if "vidswin" in key:
#             new_key = key.replace("vidswin", "vis_backbone")
#             print(f"Make old key compatible, old: {key}, new {new_key}")
#             args[new_key] = training_args[key]
#         if "backbone" in key and not (
#                 'vis_backbone' in key or 'txt_backbone' in key):
#             new_key = key.replace("backbone", "vis_backbone")
#             print(f"Make old key compatible, old: {key}, new {new_key}")
#             if new_key in toUpdate:
#                 args[new_key] = training_args[key]
#     if "vis_backbone" not in training_args and "backbone" not in training_args:
#         print(f"Evaluating models without specific backbone,"
#               f"revert to default: vidswin")
#         args.vis_backbone = "vidswin"
#     return args
