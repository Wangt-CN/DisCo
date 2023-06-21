from .logger import LOGGER as logger
from pprint import pformat
import torch
from .lib import WANDB_ENABLE


def get_deepspeed_config(args):
        use_fp16 = args.deepspeed  # args.deepspeed_fp16
        use_amp = not args.deepspeed  # not args.deepspeed_fp16 
        # use_amp = True
        # use_fp16 = False
        # by default, if not use deepspeed fp16, will enable deepspeed amp 
        gradient_accumulate_steps = getattr(args, 'gradient_accumulate_steps', 1)
        config_params = {
            'train_micro_batch_size_per_gpu': args.local_train_batch_size, # batch size per GPU without grandient accumulation
            "gradient_accumulation_steps": gradient_accumulate_steps
        }
        if use_amp:
            config_params['amp'] = {
                'enabled': True,
                'opt_level': 'O2',
            }

        if use_fp16:
            config_params['fp16'] = {
                'enabled': True,
                # "auto_cast": True,
            }
            if args.do_train:
                config_params['zero_optimization'] = {
                    'stage': 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True},
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                    "reduce_scatter": True, # default
                    "reduce_bucket_size": 5e8, # default
                    "allgather_bucket_size": 5e8, # default
                    "round_robin_gradients": True
                    }
                config_params['zero_allow_untested_optimizer'] = True
        if args.do_train:
            gradient_clip = getattr(args, 'max_grad_norm', -1)
            if gradient_clip > 0:
                config_params['gradient_clipping'] = gradient_clip

        config_params['flops_profiler'] = {
            'enabled': False,
            'profile_step': 1,
            'module_depth': -1,
            'top_modules': 3,
            'detailed': True,
        }
        # if WANDB_ENABLE:
        #     config_params['wandb'] = {
        #         "enabled": True,
        #         "group": args.wandb_project,
        #         "project": args.project_name
        #     }
        config_params['tensorboard'] = {
            "enabled": True,
            "output_path": args.log_dir,
            "job_name": "tensorboard_log"
        }

        # if hasattr(args, "zero_opt_stage") and args.zero_opt_stage > 0:
        
        # if args.zero_opt_stage > 0:
        #     config_params['fp16'] = {
        #         'enabled': True
        #     }

        logger.info(pformat(config_params))
        return config_params


def fp32_to_fp16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.half)
    elif isinstance(batch, list):
        new_batch = [fp32_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch
