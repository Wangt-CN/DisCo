# -*- coding:utf-8 -*-
import os
import sys
import shutil
import subprocess
import logging
import colorlog
import argparse
import copy
import pathlib
import shlex
import deepdish
from tqdm import tqdm
import time
import platform
import pickle
import yaml
import glob
import random
import msgpack
import importlib
import traceback
from PIL import Image
import functools
from functools import partial
import urllib.request
from warnings import simplefilter
from datetime import timedelta
from timeit import default_timer
from configobj import ConfigObj
import requests
import psutil
import hashlib
import imageio
import math
import h5py
import csv
import collections
import json
import json_lines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
import torch.distributed as dist
from torchvision import datasets, transforms, utils
from torch import autocast

# Disable transformers outputs weights.
logging.getLogger().setLevel(logging.WARNING)
simplefilter(action='ignore', category=FutureWarning)

def ldm_tensor2img_wt(input):
    assert len(input.shape) == 3
    return Image.fromarray(input.mul(255).add_(0.5).clamp_(0.5, 255).permute(1,2,0).to("cpu", torch.uint8).numpy())

def ldm_tensor2img(input, preprocess=False):
    if preprocess:
        input = (input / 2 + 0.5).clamp(0, 1)
    assert len(input.shape) == 3
    image = input.cpu().permute(1, 2, 0).float().numpy()
    image = (image * 255).round().astype("uint8")
    if image.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = Image.fromarray(image.squeeze(), mode="L")
    else:
        pil_images = Image.fromarray(image)

    return pil_images


def get_logger(filename=None):
    """
    examples:
        logger = get_logger('try_logging.txt')

        logger.debug("Do something.")
        logger.info("Start print log.")
        logger.warning("Something maybe fail.")
        try:
            raise ValueError()
        except ValueError:
            logger.error("Error", exc_info=True)

        tips:
        DO NOT logger.inf(some big tensors since color may not helpful.)
    """
    logger = logging.getLogger('utils')
    level = logging.DEBUG
    logger.setLevel(level=level)
    # Use propagate to avoid multiple loggings.
    logger.propagate = False
    # Remove %(levelname)s since we have colorlog to represent levelname.
    format_str = '[%(asctime)s <%(filename)s:%(lineno)d> %(funcName)s] %(message)s'

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    coloredFormatter = colorlog.ColoredFormatter(
        '%(log_color)s' + format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            # 'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'reg,bg_white',
        }
    )

    streamHandler.setFormatter(coloredFormatter)
    logger.addHandler(streamHandler)

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(level)
        formatter = logging.Formatter(format_str)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Fix multiple logging for torch.distributed
    try:
        class UniqueLogger:
            def __init__(self, logger):
                self.logger = logger
                self.local_rank = torch.distributed.get_rank()

            def info(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.info(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.warning(msg, *args, **kwargs)

        logger = UniqueLogger(logger)
    # AssertionError for gpu with no distributed
    # AttributeError for no gpu.
    except Exception:
        pass
    return logger


logger = get_logger()
logger.info("<utils.py>: Deep Learning Utils @ Chenfei Wu")


def path_join(path, *paths):
    output = os.path.join(path, *paths).replace('\\', '/')
    return output


class Timer:
    def __init__(self):
        '''
        t = Timer()
        time.sleep(1)
        print(t.elapse())
        '''
        self.start = default_timer()

    def elapse(self, readable=False):
        seconds = default_timer() - self.start
        if readable:
            seconds = str(timedelta(seconds=seconds))
        return seconds


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        logger.info('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def identity(x):
    return x


def groupby(l, key=lambda x: x):
    d = collections.defaultdict(list)
    for item in l:
        d[key(item)].append(item)
    return dict(d.items())


def list_filenames(dirname, filter_fn=None, sort_fn=None, printable=True):
    dirname = os.path.abspath(dirname)
    filenames = os.listdir(dirname)
    filenames = [os.path.join(dirname, filename) for filename in filenames]
    if filter_fn:
        tmp = len(filenames)
        if printable:
            logger.info('Start filtering files in %s by %s.' % (dirname, filter_fn))
        filenames = [e for e in filenames if filter_fn(e)]
        if printable: logger.info(
            'Detected %s files/dirs in %s, filtering to %s files.' % (tmp, dirname, len(filenames)))
    else:
        if printable: logger.info('Detected %s files/dirs in %s, No filtering.' % (len(filenames), dirname))
    if sort_fn:
        filenames = sorted(filenames, key=sort_fn)

    return filenames


def listdict2dict2list(listdict, printable=True):
    tmp_dict = collections.defaultdict(list)
    for example_dict in listdict:
        for k, v in example_dict.items():
            tmp_dict[k].append(v)
    if printable: logger.info('%s' % tmp_dict.keys())
    return dict(tmp_dict)


def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname

def get_suffix(file_path):
    try:
        return os.path.splitext(file_path)[-1]
    except:
        raise ValueError(f"file_path:{file_path} error!")

def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        elif extname == 'msg':
            with open(filename, 'wb') as f:
                msgpack.dump(data, f)
        elif extname == 'h5':
            if kwargs is None:
                params = {}
            split_num = kwargs.get('split_num')

            if split_num:
                if not isinstance(data, list):
                    raise ValueError(
                        '[error] utils.data2file: data must have type of list when use split_num, but got %s' % (
                            type(data)))

                if not split_num <= len(data):
                    raise ValueError(
                        '[error] utils.data2file: split_num(%s) must <= data(%s)' % (len(split_num), len(data)))

                print_save_flag = False
                print_did_not_save_flag = False
                pre_define_filenames = ["%s_%d" % (filename, i) for i in range(split_num)]
                pre_search_filenames = glob.glob("%s*" % filename)

                strict_existed = (set(pre_define_filenames) == set(pre_search_filenames) and len(
                    set([os.path.exists(e) for e in pre_define_filenames])) == 1)
                common_existed = len(set([os.path.exists(e) for e in pre_search_filenames])) == 1

                def rewrite():
                    logger.info('Spliting data to %s parts before saving...' % split_num)
                    data_splits = np.array_split(data, indices_or_sections=split_num)
                    for i, e in enumerate(data_splits):
                        deepdish.io.save("%s_%d" % (filename, i), list(e))
                    logger.info('Saved data to %s_(0~%d)' % (
                        os.path.abspath(filename), len(data_splits) - 1))

                if strict_existed and not override:
                    logger.info(
                        'Did not save data to %s_(0~%d) because the files strictly exist and override is False' % (
                            os.path.abspath(filename), len(pre_search_filenames) - 1))
                elif common_existed:
                    logger.warning('Old wrong files (maybe a differnt split) exist, auto delete them.')
                    for e in pre_search_filenames:
                        os.remove(e)
                    rewrite()
                else:
                    rewrite()
            else:
                deepdish.io.save(filename, data)
        elif extname == 'hy':
            # hy support 2 params: key and max_step
            # if key, then create group using key, else create group using index
            # if max_step, then the loop may early stopping, used for debug
            # Remove filename since h5py may corrupt.
            if override:
                remove_filename(filename)
            key_str = kwargs.pop('key_str', None)
            topk = kwargs.pop('topk', None)

            with h5py.File(filename, 'w') as f:
                for i, datum in enumerate(tqdm(data)):
                    if key_str:
                        grp = f.create_group(name=datum[key_str])
                    else:
                        grp = f.create_group(name=str(i))
                    for k in datum.keys():
                        grp[k] = datum[k]
                    if topk is not None and i + 1 == topk:
                        break
        elif extname == 'csv':
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif extname == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f)
        elif extname == 'npy':
            np.save(filename, data)
        elif extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            imageio.mimsave(filename, data, format='GIF', duration=kwargs.get('duration'))
        elif extname == 'pth':
            torch.save(data, filename)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('type can only support h5, csv, json, sess')
        if printable: logger.info('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: logger.info(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    if extname == 'pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif extname == 'msg':
        with open(filename, 'rb') as f:
            data = msgpack.load(f, encoding="utf-8")
    elif extname == 'h5':
        split_num = kwargs.get('split_num')
        if split_num:
            print_load_flag = False
            if isinstance(split_num, int):
                filenames = ["%s_%i" % (filename, i) for i in range(split_num)]
                if split_num != len(glob.glob("%s*" % filename)):
                    logger.warning('Maybe you are giving a wrong split_num(%d) != seached num (%d)' % (
                        split_num, len(glob.glob("%s*" % filename))))

            elif split_num == 'auto':
                filenames = glob.glob("%s*" % filename)
                logger.info('Auto located %d splits linked to %s' % (len(filenames), filename))
            else:
                raise ValueError("params['split_num'] got unexpected value: %s, which is not supported." % split_num)
            data = []
            for e in filenames:
                data.extend(deepdish.io.load(e))
            logger.info('Loaded data from %s_(%s)' % (
                os.path.abspath(filename), ','.join(sorted([e.split('_')[-1] for e in filenames]))))
        else:
            data = deepdish.io.load(filename)
    elif extname == 'csv':
        data = pd.read_csv(filename)
    elif extname == 'tsv':  # Returns generator since tsv file is large.
        if not kwargs.get('delimiter'):  # Set default delimiter
            kwargs['delimiter'] = '\t'
        if not kwargs.get('fieldnames'):  # Check field names
            raise ValueError('You must specify fieldnames when load tsv data.')
        # Required args.
        key_str = kwargs.pop('key_str')
        decode_fn = kwargs.pop('decode_fn')
        # Optimal args.
        topk = kwargs.pop('topk', None)
        redis = kwargs.pop('redis', None)
        if not redis:
            data = dict()
        else:
            data = redis
        if not redis or not redis.check():
            with open(filename) as f:
                reader = csv.DictReader(f, **kwargs)
                for i, item in enumerate(tqdm(reader)):
                    if not redis:  # if memory way
                        decode_fn(item)
                    data[item[key_str]] = item
                    if topk is not None and i + 1 == topk:
                        break
        else:
            logger.warning('check_str %s in redis, skip loading.' % data.check_str)
    elif extname == 'hy':
        data = h5py.File(filename, 'r')
    elif extname in ['npy', 'npz']:
        try:
            data = np.load(filename, allow_pickle=True)
        except UnicodeError:
            logger.warning('%s is python2 format, auto use latin1 encoding.' % os.path.abspath(filename))
            data = np.load(filename, encoding='latin1', allow_pickle=True)
    elif extname == 'json':
        with open(filename) as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError('[error] utils.file2data: failed to load json file %s' % filename)
    elif extname == 'jsonl':
        with open(filename, 'rb') as f:
            data = [e for e in json_lines.reader(f)]
    elif extname == 'ini':
        data = ConfigObj(filename, encoding='utf-8')
    elif extname in ['pth', 'ckpt', 'pt']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            logger.info('Loaded data from %s' % os.path.abspath(filename))
    return data


def download_file(fileurl, filedir=None, progress_bar=True, override=False, fast=False, printable=True):
    if filedir:
        ensure_dirname(filedir)
        assert os.path.isdir(filedir)
    else:
        filedir = ''
    filename = os.path.abspath(os.path.join(filedir, fileurl.split('/')[-1]))
    # print(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.info("%s not exist, automatic makedir." % dirname)
    if not os.path.exists(filename) or override:
        if fast:
            p = subprocess.Popen('axel -n 10 -o {0} {1}'.format(filename, fileurl), shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(p.stdout.readline, ''):
                if line:
                    logger.info(line.decode('utf-8').replace('\n', ''))
                else:
                    p.kill()
                    break
        else:
            if progress_bar:
                def my_hook(t):
                    last_b = [0]

                    def inner(b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            t.total = tsize
                        t.update((b - last_b[0]) * bsize)
                        last_b[0] = b

                    return inner

                with tqdm(unit='B', unit_scale=True, miniters=1,
                          desc=fileurl.split('/')[-1]) as t:
                    urllib.request.urlretrieve(fileurl, filename=filename,
                                               reporthook=my_hook(t), data=None)
            else:
                urllib.request.urlretrieve(fileurl, filename=filename)
        if printable: logger.info("%s downloaded sucessfully." % filename)
    else:
        if printable: logger.info("%s already existed" % filename)
    return filename


def copy_file(filename, targetname, override=False, printable=True):
    filename = os.path.abspath(filename)
    targetname = os.path.abspath(targetname)
    if not os.path.exists(targetname) or override:
        shutil.copy2(filename, targetname)
        if printable:
            logger.info('Copied %s to %s.' % (filename, targetname))
    else:
        if printable:
            logger.info('Did not copy because %s exists.' % targetname)


def videofile2videometa(input_video):
    out = execute_cmd('ffprobe -i %s -print_format json -show_streams' % input_video)
    meta = json.loads(out.decode('utf-8'))

    if 'duration' in meta['streams'][0]:
        duration = float(meta['streams'][0]['duration'])
    else:  # Fix Duration for webm format.
        duration_str = meta['streams'][0]['tags']['DURATION']
        h, m, s = duration_str.split(':')
        duration = float(h) * 3600 + float(m) * 60 + float(s)

    res = {'width': meta['streams'][0]['width'],
           'height': meta['streams'][0]['height'],
           'duration': duration,
           'fps': eval(meta['streams'][0]['r_frame_rate'])}
    return res


def videofile2videoarr(input_file, seek_start=None, seek_duration=None, seek_fps=None):
    ffprob_out = execute_cmd(f'ffprobe -i {input_file} -print_format json -show_streams')
    meta = json.loads(ffprob_out.decode('utf-8'))
    width = meta['streams'][0]['width']
    height = meta['streams'][0]['height']
    cmd = f'ffmpeg -y -i {input_file} '
    if seek_start:
        cmd += f'-ss {seek_start} '
    if seek_duration:
        cmd += f'-t {seek_duration} '
    if seek_fps:
        cmd += f'-filter_complex [0]fps=fps={seek_fps}[s0] -map [s0] '
    cmd += '-f rawvideo -pix_fmt rgb24 pipe:'
    #     assert cmd == 'ffmpeg -y -i pipe: -ss 2 -t 4 -filter_complex [0]fps=fps=0.5[s0] -map [s0] -f rawvideo -pix_fmt rgb24 pipe:'
    ffmpeg_out = execute_cmd(cmd)
    video = np.frombuffer(ffmpeg_out, np.uint8)
    video = video.reshape([-1, height, width, 3])
    return video


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        logger.info('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        logger.info('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)


def ensure_filename(filename, override=False):
    dirname, rootname, extname = split_filename(filename)
    ensure_dirname(dirname, override=False)
    if os.path.exists(filename) and override:
        os.remove(filename)
        logger.info('Deleted filename %s' % filename)


def remove_filename(filename, printable=False):
    if os.path.isfile(filename) or os.path.islink(filename):
        os.remove(filename)
        if printable:
            logger.info('Deleted file %s.' % filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)
        if printable:
            logger.info('Deleted dir %s.' % filename)
    else:
        raise ValueError("%s is not a file or dir." % filename)


def execute(cmd, wait=True, printable=True):
    if wait:
        if printable: logger.warning('Executing: '"%s"', waiting...' % cmd)
        try:
            output = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            logger.warning(e.output)
            output = None
            # sys.exit(-1)

        return output
    else:
        if platform.system() == 'Windows':
            black_hole = 'NUL'
        elif platform.system() == 'Linux':
            black_hole = '/dev/null'
        else:
            raise ValueError('Unsupported system %s' % platform.system())
        cmd = cmd + ' 1>%s 2>&1' % black_hole
        if printable: logger.info('Executing: '"%s"', not wait.' % cmd)
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def execute_cmd(cmd, input_data=None):
    process = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=input_data)
    retcode = process.poll()
    if retcode:
        raise ValueError(err.decode('utf-8'))
    return out


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pname2pid(str_proc_name):
    map_proc_info = {}
    for proc in psutil.process_iter():
        if proc.name() == str_proc_name:
            map_proc_info[proc.pid] = str_proc_name

    return map_proc_info


def get_parameters(net: torch.nn.Module):
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    fp32_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and p.requires_grad)
    fp16_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and p.requires_grad)
    fp32_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and not p.requires_grad)
    fp16_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and not p.requires_grad)
    return {'trainable': trainable_params, 'frozen': frozen_params,
            'trainable_fp32': fp32_trainable_params,
            'trainalbe_fp16': fp16_trainable_params,
            'frozen_fp32': fp32_frozen_params, 'frozen_fp16': fp16_frozen_params}


def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    common_dict = {}
    mismatched_keys, unexpected_keys = [], []
    for k, v in state_dict.items():
        if k in target_dict:
            try:
                if v.size() != target_dict[k].size():
                    mismatched_keys.append(k)
                else:
                    common_dict[k] = v
            except Exception as e:
                logger.warning(f'load error for {k} {e}')
                common_dict[k] = v
        else:
            unexpected_keys.append(k)


    # try:
    #     common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
    # except Exception as e:
    #     logger.warning('load error %s', e)
    #     common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        logger.warning('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)

    missing_keys = [k for k in target_dict.keys() if k not in common_dict and k not in  mismatched_keys]
    # unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        logger.warning(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(mismatched_keys) != 0:
        logger.warning(
            f"Mismatched shape of weights loaded from state_dict: {mismatched_keys}"
        )
    if len(missing_keys) != 0:
        logger.warning(
            f"Some weights of target are missing in state_dict: {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0 and len(mismatched_keys) == 0:
        logger.warning("Strictly Loaded state_dict.")


class Meter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if isinstance(val, (int, float)):
            self.val = val
            if self.sum:
                self.sum += val * n
            else:
                self.sum = val * n
            if self.count:
                self.count += n
            else:
                self.count = n
            self.avg = self.sum / self.count
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    val[k] = v.item()
            # if 'loss_total' in val.keys():
            #     if not math.isnan(val['loss_total']):
            #         # print('it is not nan!!!')
            if self.val:
                for k in val.keys():
                    self.val[k] = val[k]
            else:
                self.val = val
            if self.sum:
                for k in val.keys():
                    if k in self.sum:
                        self.sum[k] = self.sum[k] + val[k] * n
                    else:
                        self.sum[k] = val[k] * n
            else:
                self.sum = {k: val[k] * n for k in val.keys()}
            if self.count:
                for k in val.keys():
                    if k in self.count:
                        self.count[k] = self.count[k] + n
                    else:
                        self.count[k] = n
            else:
                self.count = {k: n for k in val.keys()}
            self.avg = {k: self.sum[k] / self.count[k] for k in self.count.keys()}
        else:
            raise ValueError('Not supported type %s' % type(val))

    def __str__(self):
        if isinstance(self.avg, dict):
            return str({k: "%.4f" % v for k, v in self.avg.items()})
        else:
            return 'Nan'


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Trainer:
    """
        Trainer
    """

    def __init__(self, args, model, optimizers=None, scheduler=None, pretrained_model=None, use_amp=True,
                 find_unused_parameters=True):
        # Basic Params
        self.args = args
        self.log_dir = args.log_dir
        self.model = model
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.pretrained_model = pretrained_model
        self.use_amp = use_amp
        self.find_unused_parameters = find_unused_parameters
        # Load Pretrained Models.
        if pretrained_model:
            self.from_pretrained(pretrained_model)
        # Get Variables from ENV
        self.rank = int(os.getenv('RANK', '-1'))
        self.local_rank = int(os.getenv('LOCAL_RANK', '-1'))
        # Define Running mode.
        if self.local_rank == -1:
            self.mode = 'common'
            self.enable_write_model = True
            self.enable_collect = True
            self.enable_write_metric = True
        else:
            self.mode = 'dist'
            self.enable_write_model = (self.rank == 0)
            self.enable_collect = True
            self.enable_write_metric = (self.rank == 0)

        if self.enable_write_metric:
            ensure_dirname(self.log_dir, override=False)
        self.metric_filename = os.path.join(self.log_dir, 'metric.json')
        self.last_checkpoint_filename = os.path.join(self.log_dir, 'last.pth')
        self.best_checkpoint_filename = os.path.join(self.log_dir, 'best.pth')
        self.each_checkpoint_filename = os.path.join(self.log_dir, 'epoch%s.pth')
        self.epoch = -1

        # Get device and number of GPUs
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu >= 1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if (self.use_amp or self.args.deepspeed) and self.n_gpu < 1:
            raise ValueError('AMP/Deepspeed Does not support CPU!')
        if self.use_amp and self.mode == 'common':
            logger.warning('In common mode, remember to @autocast before forward function.')
        if self.use_amp:
            logger.warning('Attntion!!! you are using amp')
        if self.args.deepspeed:
            logger.warning('Attntion!!! you are using deepspeed')

        self.scalar = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # TODO
        if hasattr(args, 'iterative_model_class'):
            self.iterative_model = args.iterative_model_class(args=args)
        else:
            self.iterative_model = None

    def reduce_mean(self, tensor):
        rt = tensor.clone()
        size = int(os.environ['WORLD_SIZE'])
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt = rt / size
        return rt

    def wrap_model(self):
        if hasattr(self.model, 'module'):
            raise ValueError('You do not need to wrap a models with modules.')

        if self.mode == 'common':
            logger.info('Wrapped models to common %s.' % self.device)

            self.model.to(self.device)
            if self.n_gpu > 1:
                logger.warning('Detected %s gpus, auto using DataParallel.' % self.n_gpu)
                self.model = torch.nn.DataParallel(self.model)
        elif self.mode == 'dist':
            logger.info('Wrapped models to distributed %s.' % self.device)

            self.device = torch.device("cuda", self.local_rank)
            self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.find_unused_parameters)
        else:
            raise ValueError
        # wrap_optimizers
        if self.optimizers:
            for i in range(len(self.optimizers)):
                self.optimizers[i].load_state_dict(
                    complex_to_device(self.optimizers[i].state_dict(), device=self.device))

    def check_outputs(self, outputs):
        error_message = 'Model output must be a dict. The key must be "class_subclass" format.' \
                        ' "class" can only be loss, metric, or logits. "subclass" should be a string.' \
                        ' But got an unexpected key %s'
        loss_total_list = [e for e in outputs.keys() if e.startswith('loss_total')]
        if not loss_total_list:
            raise ValueError('Model output must contain a key startswith "loss_total"!')

        for k, v in outputs.items():
            split_res = k.split('_')
            if len(split_res) < 2:
                raise ValueError(error_message % k)
            if k.split('_')[0] not in ['loss', 'metric', 'logits']:
                raise ValueError(error_message % k)

    def train(self, train_loader, eval_loader=None, epochs=5, resume=True, eval_step=10,
              save_step=None, use_tqdm=None, max_norm=None, gradient_accumulate_steps=1,
              inner_collect_fn=None, best_metric_fn=lambda x: x['train']['loss_total']):
        if not save_step:
            save_step = eval_step
        best_eval_metric = np.Infinity
        if resume:
            if os.path.exists(self.last_checkpoint_filename):
                self.load_checkpoint(self.last_checkpoint_filename)
                #self.load_checkpoint('/workspace/f_ndata_tmp/last.pth')
        else:
            if self.enable_write_metric:
                logger.warning('Dangerous! You set resume=False. Auto cleaning all the logs under %s' % self.log_dir)
                ensure_dirname(self.log_dir, override=True)

        self.wrap_model()

        epoch_iter = range(self.epoch + 1, epochs, 1)
        if len(epoch_iter):
            logger.warning('Start train & val phase...')
        else:
            logger.warning('Skip train & val phase...')
        logger.warning(f'Train examples: {len(train_loader.dataset)}, epochs: {epochs}, '
                       f'global_batch_size: {self.args.train_batch_size}, local_batch_size: {train_loader.batch_size}.')

        # Train & Eval phase
        for epoch in epoch_iter:
            self.epoch = epoch
            # Train phase
            train_meter, train_time = self.train_fn(train_loader,
                                                    max_norm=max_norm,
                                                    gradient_accumulate_steps=gradient_accumulate_steps,
                                                    use_tqdm=use_tqdm)
            logger.info('[Rank %s] Train Epoch: %d/%d, Time: %s\n %s' %
                        (self.rank, epoch + 1, epochs, train_time, train_meter.avg))
            if not isinstance(train_meter.avg, dict):
                raise ValueError(type(train_meter.avg))
            metric = {'Epoch%s' % (epoch + 1): {'train': {**train_meter.avg, **{'time': train_time}}}}

            if self.enable_write_metric:
                self.update_metric_file(metric)
            if (epoch + 1) % save_step == 0:
                if self.enable_write_model:
                    self.save_checkpoint(self.last_checkpoint_filename)
                    # copy_file(self.last_checkpoint_filename, self.each_checkpoint_filename % str(epoch + 1),
                    #           override=True)  # TODO sometimes we need to copy file

            if (epoch + 1) % eval_step == 0:
                if eval_loader:
                    eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                         use_tqdm=use_tqdm)
                    logger.info('[Rank %s] Valid Epoch: %d/%d, Time: %s\n %s' %
                                (self.rank, epoch + 1, epochs, eval_time, eval_meter.avg))

                    # Update metric with eval metrics
                    # metric['Epoch%s' % (epoch + 1)].update({'eval': {**eval_meter.avg, **{'time': eval_time}}})

                    # Save metric file
                    # if self.enable_write_metric:
                    #     self.update_metric_file(metric)

                    # If the best models, save another checkpoint.
                    # if best_metric_fn(metric['Epoch%s' % (epoch + 1)]) < best_eval_metric and self.enable_write_model:
                    #     best_eval_metric = best_metric_fn(metric['Epoch%s' % (epoch + 1)])
                    #     if os.path.exists(self.last_checkpoint_filename):
                    #         copy_file(self.last_checkpoint_filename, self.best_checkpoint_filename, override=True)
                    #     else:
                    #         logger.warning('No checkpoint_file %s' % self.last_checkpoint_filename)

    def eval(self, eval_loader, inner_collect_fn=None, use_tqdm=True):
        # This function is used to do evaluating after training.
        if not self.pretrained_model:
            logger.warning('You should create a new config file and specify pretrained_model in Args when using eval.')
        # Wrap models before evaluating. This will support ddp evaluating.
        self.wrap_model()
        eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn, use_tqdm=use_tqdm)
        logger.info('[Rank %s] Valid Time: %s\n %s' % (self.rank, eval_time, eval_meter.avg))

    def update_metric_file(self, metric):
        if os.path.exists(self.metric_filename):
            r = file2data(self.metric_filename, printable=False)
            data2file(dict(r, **metric), self.metric_filename, override=True)
        else:
            data2file(metric, self.metric_filename)

    def train_fn(self, train_loader, max_norm, gradient_accumulate_steps=1, use_tqdm=True):
        self.model.train()
        train_meter = Meter()
        train_timer = Timer()
        train_iter = tqdm(train_loader, total=len(train_loader), disable=not use_tqdm)
        for step, inputs in enumerate(train_iter):
            # torch.cuda.empty_cache()
            for optimizer_idx in range(len(self.optimizers)):
                # if not getattr(self.optimizers[optimizer_idx], 'is_enabled', lambda x: True)(self.epoch):
                #     continue
                if not getattr(self.optimizers[optimizer_idx], 'is_enabled', lambda x: True)(self.epoch * len(train_loader) + step):
                    continue # adjust to sdm KL-VAE
                inputs = complex_to_device(inputs, self.device)

                inputs['epoch'] = self.epoch
                inputs['global_step'] = self.epoch * len(train_loader) + step
                inputs['optimizer_idx'] = optimizer_idx

                # for outputs in self.models(inputs):
                # for outputs in self.iterative_model.forward(self.model, inputs) \
                #         if self.iterative_model else [self.model(inputs)]:
                #     self.check_outputs(outputs)
                #     # If we use nn.Parallel, we will get a list of metric or losses from different GPUs, we need to mean them.
                #     if self.mode == 'common' and self.n_gpu > 1:
                #         for k, v in outputs.items():
                #             if k.split('_')[0] in ['metric', 'loss']:
                #                 outputs[k] = v.mean()
                #
                #     if optimizer_idx == 0:
                #         outputs['loss_total'].backward()
                #     else:
                #         outputs['loss_total_%s' % optimizer_idx].backward()
                #
                #     if (step + 1) % gradient_accumulate_steps == 0 and outputs.get('logits_last', True):
                #         if max_norm:
                #             nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                #         self.optimizers[optimizer_idx].step()
                #         self.optimizers[optimizer_idx].zero_grad()
                #
                #     metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                #     if self.mode != 'common':
                #         for k, v in metric_and_loss.items():
                #             metric_and_loss[k] = self.reduce_mean(v)
                #     train_meter.update(metric_and_loss)
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = self.model(inputs)

                    self.check_outputs(outputs)
                    # If we use nn.Parallel, we will get a list of metric or losses from different GPUs, we need to mean them.
                    if self.mode == 'common' and self.n_gpu > 1:
                        for k, v in outputs.items():
                            if k.split('_')[0] in ['metric', 'loss']:
                                outputs[k] = v.mean()

                    if optimizer_idx == 0:
                        # outputs['loss_total'].backward()
                        self.scalar.scale(outputs['loss_total']).backward()
                    else:
                        # outputs['loss_total_%s' % optimizer_idx].backward()
                        self.scalar.scale(outputs['loss_total_%s' % optimizer_idx]).backward()

                    if (step + 1) % gradient_accumulate_steps == 0 and outputs.get('logits_last', True):
                        if max_norm:
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                        # self.optimizers[optimizer_idx].step()
                        self.scalar.step(self.optimizers[optimizer_idx])
                        self.scalar.update()
                        self.optimizers[optimizer_idx].zero_grad()

                    metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                    if self.mode != 'common':
                        for k, v in metric_and_loss.items():
                            metric_and_loss[k] = self.reduce_mean(v)
                    train_meter.update(metric_and_loss)

                    if self.scheduler:
                        self.scheduler.step()
                else:
                    outputs = self.model(inputs)
                    self.check_outputs(outputs)
                    # If we use nn.Parallel, we will get a list of metric or losses from different GPUs, we need to mean them.
                    if self.mode == 'common' and self.n_gpu > 1:
                        for k, v in outputs.items():
                            if k.split('_')[0] in ['metric', 'loss']:
                                outputs[k] = v.mean()

                    if optimizer_idx == 0:
                        outputs['loss_total'].backward()
                    else:
                        outputs['loss_total_%s' % optimizer_idx].backward()

                    if (step + 1) % gradient_accumulate_steps == 0 and outputs.get('logits_last', True):
                        if max_norm:
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                        self.optimizers[optimizer_idx].step()
                        self.optimizers[optimizer_idx].zero_grad()

                    metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                    if self.mode != 'common':
                        for k, v in metric_and_loss.items():
                            metric_and_loss[k] = self.reduce_mean(v)
                    train_meter.update(metric_and_loss)

                    if self.scheduler:
                        self.scheduler.step()

            if (inputs['global_step'] + 1) % (int(getattr(self.args, 'save_setp', 8000)) + 1) == 0:
                if self.enable_write_model:
                    self.save_checkpoint(os.path.join(self.log_dir, str(inputs['global_step']) + '.pth'))
            train_iter.set_description("Metering:" + str(train_meter))
        train_time = train_timer.elapse(True)
        return train_meter, train_time

    def eval_fn(self, eval_loader, inner_collect_fn=None, use_tqdm=True):
        # TODO Note that eval_fn supports ddp. So we do not need to unwrap things here.
        model_to_eval = self.model
        model_to_eval.eval()
        eval_meter = Meter()
        eval_timer = Timer()
        with torch.no_grad():
            eval_loader = tqdm(eval_loader, total=len(eval_loader)) if use_tqdm else eval_loader
            for inputs in eval_loader:
                inputs = complex_to_device(inputs, self.device)
                outputs = model_to_eval(inputs)
                if self.mode == 'common' and self.n_gpu > 1:
                    for k, v in outputs.items():
                        if k.split('_')[0] in ['metric', 'loss']:
                            outputs[k] = v.mean()
                metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                if self.mode != 'common':
                    for k, v in metric_and_loss.items():
                        metric_and_loss[k] = self.reduce_mean(v)
                eval_meter.update(metric_and_loss)

                if inner_collect_fn and self.enable_collect:
                    inner_collect_fn(self.args, inputs, outputs, self.log_dir, self.epoch, self.args.eval_save_filename)

        eval_time = eval_timer.elapse(True)
        return eval_meter, eval_time

    def load_checkpoint(self, checkpoint_filename):
        if hasattr(self.model, "module"):
            raise ValueError("Please do not load checkpoint into wrapped models, ensure self.models is CPU.")
        checkpoint = file2data(checkpoint_filename, map_location='cpu')
        adaptively_load_state_dict(self.model, checkpoint['models'])
        if self.optimizers:
            if len(self.optimizers) > 1:
                for i, optimizer in enumerate(self.optimizers):
                    adaptively_load_state_dict(self.optimizers[i], checkpoint['optimizer'][i])

            elif len(self.optimizers) == 1:
                adaptively_load_state_dict(self.optimizers[0], checkpoint['optimizer'])

            else:
                raise ValueError
        if self.scheduler:
            adaptively_load_state_dict(self.scheduler, checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] - 1

        # IMPORTANT! The models will be wrapped automatically.
        logger.warning('Loaded checkpoint %s of epoch %s' % (checkpoint_filename, checkpoint['epoch']))

    def save_checkpoint(self, checkpoint_filename):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if len(self.optimizers) > 1:
            optimizer_to_save = [optimizer.state_dict() for optimizer in self.optimizers]
        elif len(self.optimizers) == 1:
            optimizer_to_save = self.optimizers[0].state_dict()
        else:
            raise ValueError
        checkpoint = {
            'models': model_to_save.state_dict(),
            'optimizer': optimizer_to_save,
            'epoch': self.epoch + 1,
        }
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        data2file(checkpoint, checkpoint_filename, override=True)
        logger.warning('Saved epoch %s to %s.' % (checkpoint['epoch'], checkpoint_filename))

    def from_pretrained(self, pretrained_model):
        if hasattr(self.model, "module"):
            raise ValueError("Please do not load pretrained models into wrapped models, ensure self.models is CPU.")
        if isinstance(pretrained_model, str):
            logger.warning('Loading Pretrained Model Path: %s...' % pretrained_model)
            pretrained_dict = file2data(pretrained_model, map_location='cpu')
            if 'models' in pretrained_dict:
                pretrained_dict = pretrained_dict['models']
        else:
            logger.warning('Loading Given Pretrained Dict...')
            pretrained_dict = pretrained_model
        adaptively_load_state_dict(self.model, pretrained_dict)


def dl2ld(dl):
    return [dict(zip(dl, e)) for e in zip(*dl.values())]


def ld2dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def complex_to_device(complex, device, non_blocking=False):
    # added by Linjie
    if complex is None:
        return complex
    if isinstance(complex, torch.Tensor):
        return complex.to(device, non_blocking=non_blocking)
    elif isinstance(complex, dict):
        return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
    elif isinstance(complex, list) or isinstance(complex, tuple):
        return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
    elif isinstance(complex, str) or isinstance(complex, bytes) or \
            isinstance(complex, int) or isinstance(complex, float):
        return complex
    else:
        raise ValueError('Unsupported complex', complex)
