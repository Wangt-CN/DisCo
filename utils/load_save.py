"""
saving utilities
"""
import json
import os
from os.path import dirname, exists, join, realpath
from apex import amp
from easydict import EasyDict as edict
from .basic_utils import is_jsonable, save_json, make_zipfile

import torch
from .logger import LOGGER


def save_training_meta(args):
    # args is an EasyDict object, treat it the same as a normal dict
    os.makedirs(join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(join(args.output_dir, 'ckpt'), exist_ok=True)

    # training args
    save_args_path = join(args.output_dir, 'log', 'args.json')
    save_json(args, save_args_path, save_pretty=True)

    # model args
    model_config = json.load(open(args.model_config))
    save_model_config_path = join(args.output_dir, 'log', 'model_config.json')
    save_json(model_config, save_model_config_path, save_pretty=True)

    # save a copy of the codebase. !!!Do not store heavy file in your codebase when using it.
    code_dir = dirname(dirname(dirname(realpath(__file__))))
    code_zip_filename = os.path.join(args.output_dir, "code.zip")
    LOGGER.info(f"Saving code from {code_dir} to {code_zip_filename}...")
    make_zipfile(code_dir, code_zip_filename,
                 enclosing_dir="code",
                 exclude_dirs_substring="results",
                 exclude_dirs=["results", "debug_results", "__pycache__", "linjli"],
                 exclude_extensions=[".pyc", ".ipynb", ".swap"])
    LOGGER.info(f"Saving code done.")


class TrainingSaver(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.max_save_load_trial = 10
    
    def save_tokenizer(self, tokenizer):
        tokenizer_dir = join(self.output_dir, 'tokenizer')
        os.makedirs(tokenizer_dir, exist_ok=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(tokenizer_dir)

    def save_args(self, args):
        arg_dir = join(self.output_dir, 'log')
        os.makedirs(arg_dir, exist_ok=True)
        save_args_path = join(arg_dir, 'args.json')
        LOGGER.info(f"Training/evaluation parameters: {args}")
        LOGGER.info(f"saving args to {save_args_path}")
        temp_args = edict(vars(args))
        for key, value in temp_args.items():
            if not is_jsonable(value):
                value = f'{value}'
                temp_args[key] = value
        save_json(temp_args, save_args_path, save_pretty=True, sort_keys=True)

    def save_model(self, checkpoint_dir, step, model, optimizer=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = join(checkpoint_dir, 'model.bin')
        model_to_save = model.module if hasattr(model, 'module') else model
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model_to_save.state_dict().items()}
        # with retrial, as azure blob fails occasionally.
        save_trial = 0
        while save_trial < self.max_save_load_trial:
            exception_msg = ''
            try:
                LOGGER.info(f"ModelSaver save trial NO. {save_trial}")
                torch.save(state_dict, model_path)
                if optimizer is not None:
                    optimizer_state_dict = {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in optimizer.state_dict().items()}
                    dump = {'step': step, 'optimizer': optimizer_state_dict}
                    torch.save(
                        dump,
                        f'{checkpoint_dir}/optmizer_state.bin')
                LOGGER.info(f"Save checkpoint to {checkpoint_dir}")
                break
            except Exception as e:
                exception_msg = e
                save_trial += 1
        else:
            LOGGER.info(
                f"Failed to save checkpoint after {self.max_save_load_trial} trails, "
                f"exception msg: {exception_msg}.")
        return


def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
    """operated in-place, no need to return `model`"""

    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(
            loaded_state_dict_or_path, map_location="cpu")
    else:
        loaded_state_dict = loaded_state_dict_or_path
    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]

    LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
    LOGGER.info("Keys in loaded but not in model:")
    diff_keys = load_keys.difference(model_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model but not in loaded:")
    diff_keys = model_keys.difference(load_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model and loaded, but shape mismatched:")
    LOGGER.info(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
    model.load_state_dict(toload, strict=False)


def compare_dict_difference(dict1, dict2, dict1_name="dict1",
                            dict2_name="dict2",
                            print_value_diff=True, verbose=False,
                            exclude_keys=()):
    """
    Args:
        dict1:
        dict2:
        dict1_name:
        dict2_name:
        print_value_diff: bool, output dict value difference within shared keys
            for dict1 and dict2. In effect only when verbose == True
        verbose:
    """
    exclude_keys = set(exclude_keys)
    keys1 = set(dict1.keys()).difference(exclude_keys)
    keys2 = set(dict2.keys()).difference(exclude_keys)
    shared_keys = keys1.intersection(keys2)
    keys1_unique = keys1.difference(shared_keys)
    keys2_unique = keys2.difference(shared_keys)
    key_diff_list = list(keys1_unique) + list(keys2_unique)

    # value difference in the shared keys in dict1 and dict2
    value_diff_dict = {}
    for k in shared_keys:
        if dict1[k] != dict2[k]:
            value_diff_dict[k] = [(dict1_name, dict1[k]), (dict2_name, dict2[k])]

    if len(value_diff_dict) == 0 and len(key_diff_list) == 0:
        return True

    def print_value_diff():
        if verbose and print_value_diff:
            LOGGER.info("=" * 30 + "value difference")
            LOGGER.info(f"{json.dumps(value_diff_dict, indent=4)}")

    if len(value_diff_dict) > 0  and len(key_diff_list) == 0:
        # OK
        print_value_diff()
        return True
    
    if verbose:
        LOGGER.info("=" * 30 + "key difference")
        LOGGER.info(f"keys in {dict1_name} but not in {dict2_name}: "
                    f"total {len(keys1_unique)}, {sorted(keys1_unique)}")
        LOGGER.info(f"keys in {dict2_name} but not in {dict1_name}: "
                    f"total {len(keys2_unique)}, {sorted(keys2_unique)}")
    return False


def _to_cuda(state):
    """ usually load from cpu checkpoint but need to load to cuda """
    if isinstance(state, torch.Tensor):
        ret = state.cuda()  # assume propoerly set py torch.cuda.set_device
        if 'Half' in state.type():
            ret = ret.float()  # apex O2 requires it
        return ret
    elif isinstance(state, list):
        new_state = [_to_cuda(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cuda(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cuda(t) for n, t in state.items()}
    else:
        return state
    return new_state


def _to_cpu(state):
    """ store in cpu to avoid GPU0 device, fp16 to save space """
    if isinstance(state, torch.Tensor):
        ret = state.cpu()
        if 'Float' in state.type():
            ret = ret.half()
        return ret
    elif isinstance(state, list):
        new_state = [_to_cpu(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cpu(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cpu(t) for n, t in state.items()}
    else:
        return state
    return new_state
                


class TrainingRestorer(object):
    def __init__(self, args, model, optimizer):
        if exists(f"{args.output_dir}/log/args.json"):
            restore_args = json.load(
                open(f'{args.output_dir}/log/args.json', 'r'))
            restore_args_path = join(
                    args.output_dir, 'log',
                    'restore_args.json')
            temp_args = edict(vars(restore_args))
            for key, value in temp_args.items():
                if not is_jsonable(value):
                    value = f'{value}'
                    temp_args[key] = value
            save_json(
                temp_args, restore_args_path,
                save_pretty=True, sort_keys=True)
            assert compare_dict_difference(
                args, restore_args, dict1_name="current_args",
                dict2_name="restore_args",
                print_value_diff=True, verbose=True,
                exclude_keys=('local_rank'))
        # keep 2 checkpoints in case of corrupted
        self.save_path = f'{args.output_dir}/restore.pt'
        self.backup_path = f'{args.output_dir}/restore_backup.pt'
        self.model = model
        self.optimizer = optimizer
        self.min_restore_steps = 20
        self.restorer_save_step = max(
            self.min_restore_steps, int(args.restore_ratio * args.max_global_step))
        # since saving to or loading from azure blob fails sometimes
        self.max_save_load_trial = 10
        self.amp = args.mixed_precision_method == "apex"
        self.deepspeed = args.mixed_precision_method == "deepspeed" and args.restore_deepspeed_ckpt
        if self.deepspeed:
            self.save_path = f'{args.output_dir}/deepspeed_restore'
            os.makedirs(self.save_path, exist_ok=True)
            self.backup_path = f'{args.output_dir}/deepspeed_restore_backup'
            os.makedirs(self.backup_path, exist_ok=True)
        self.restore_at_init()
    
    def restore_at_init(self):
        if self.save_path.endswith(".pt"):
            save_path = self.save_path
            backup_path = self.backup_path
        else:
            # deepspeed
            save_path = join(self.save_path, "restore_ckpt.pt")
            backup_path = join(self.backup_path, "restore_ckpt.pt")
        if exists(save_path) or exists(backup_path):
            LOGGER.info('found previous checkpoint. try to resume...')
            exception_msg = ''
            # with retrial, as azure blob fails occasionally.
            restore_trial = 0
            while restore_trial < self.max_save_load_trial:
                LOGGER.info(f"TrainingRestorer restore trial NO. {restore_trial}")
                # try:
                self.restore()
                LOGGER.info(f"TrainingRestorer restore from global_step {self.global_step}")
                break
            #     except Exception as e: 
            #         exception_msg = e
            #         restore_trial += 1
            # else:
            #     LOGGER.info(
            #         f"TrainingRestorer restore failed after {self.max_save_load_trial} trails, "
            #         f"exception msg: {exception_msg}.")
        else:
            self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step % self.restorer_save_step == 0:
            # with retrial, as azure blob fails occasionally.
            save_trial = 0
            while save_trial < self.max_save_load_trial:
                LOGGER.info(f"TrainingRestorer save trial NO. {save_trial}")
                try:
                    self.save()
                    break
                except Exception as e:
                    save_trial += 1

    def save(self):
        checkpoint = {'global_step': self.global_step}
        if not self.deepspeed:
            # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            checkpoint['model_state_dict'] = _to_cpu(self.model.state_dict())
            checkpoint['optim_state_dict'] =  _to_cpu(self.optimizer.state_dict())
            if self.amp:
                checkpoint['amp_state_dict'] = amp.state_dict()
            if exists(self.save_path):
                os.rename(self.save_path, self.backup_path)
            torch.save(checkpoint, self.save_path)
        else:
            # deepspeed, not efficient
            if exists(self.save_path):
                os.rename(self.save_path, self.backup_path)
            else:
                self.model.save_checkpoint(self.save_path)
                torch.save(checkpoint, join(self.save_path, "restore_ckpt.pt"))

    def restore(self):
        if not self.deepspeed:
            try:
                checkpoint = torch.load(self.save_path)
            except Exception:
                checkpoint = torch.load(self.backup_path)
            self.model.load_state_dict(_to_cuda(checkpoint['model_state_dict']))
            self.optimizer.load_state_dict(
                _to_cuda(checkpoint['optim_state_dict']))
            if self.amp:
                amp.load_state_dict(checkpoint['amp_state_dict'])
        else:
            # deepspeed, not efficient
            try:
                checkpoint = torch.load(join(self.save_path, "restore_ckpt.pt"))
                self.model.load_checkpoint(self.save_path)
            except Exception:
                checkpoint = torch.load(join(self.backup_path, "restore_ckpt.pt"))
                self.model.load_checkpoint(self.backup_path)
        self.global_step = checkpoint['global_step']
        LOGGER.info(f'resume training from step {self.global_step}')
