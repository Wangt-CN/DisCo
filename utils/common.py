#import matplotlib
#matplotlib.use('Agg')
import contextlib
from PIL import Image
from io import BytesIO
import yaml
from collections import OrderedDict
#import progressbar
import json
import traceback
import random
import sys
import os
import math
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Event
import logging
import functools
import numpy as np
import glob
import re
from tqdm import tqdm
try:
    from itertools import izip as zip
except ImportError:
    # in python3, we don't need itertools.izip since zip is izip
    pass
import time
#import matplotlib.pyplot as plt
from pprint import pprint
from pprint import pformat
import numpy as np
import os.path as op
import re
import base64
import cv2
#import psutil
import shutil
import argparse
import subprocess as sp
from datetime import datetime
from future.utils import viewitems
#from ete3 import Tree
try:
    # py3
    from urllib.request import urlopen, Request
    from urllib.request import HTTPError
except ImportError:
    # py2
    from urllib2 import urlopen, Request
    from urllib2 import HTTPError
import copy
#from deprecated import deprecated
import io

from PIL import ImageFile
#https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_sys_memory_usage_info():
    out = cmd_run(['free'], return_output=True)
    import ipdb;ipdb.set_trace(context=15)
    lines = out.split('\n')
    headers = lines[0].strip().split(' ')
    headers = [h for h in headers if len(h) > 0]
    mem = lines[1]
    x1, x2 = mem.split(':')
    assert 'Mem' == x1
    values = [int(i) for i in x2.split(' ') if len(i) > 0]
    assert len(headers) == len(values)
    return dict(zip(headers, values))

def get_mem_usage_in_bytes():
    import os, psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # in bytes

def print_type_memory_usage():
    from pympler import muppy, summary
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)

def encode_np(x):
    compressed_array = io.BytesIO()
    np.savez_compressed(compressed_array, x)
    return base64.b64encode(compressed_array.getvalue())

def decode_np(s):
    s = base64.b64decode(s)
    return np.load(io.BytesIO(s))['arr_0']

def print_trace():
    import traceback
    traceback.print_exc()

def get_trace():
    import traceback
    return traceback.format_exc()

def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
            print_trace()
    return func_wrapper

def master_process_run(func):
    def func_wrapper(*args, **kwargs):
        if get_mpi_rank() == 0:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.info('ignore error \n{}'.format(str(e)))
                print_trace()
    return func_wrapper

@try_once
def try_delete(f):
    os.remove(f)

def list_to_nested_dict(xs, idxes):
    rest_idxes = set(range(len(xs[0]))).difference(idxes)
    result = {}
    for r in xs:
        curr_result = result
        for i in idxes[:-1]:
            if r[i] not in curr_result:
                curr_result[r[i]] = {}
            curr_result = curr_result[r[i]]
        key = r[idxes[-1]]
        if key not in curr_result:
            curr_result[key] = []
        value = [r[i] for i in rest_idxes]
        if len(value) == 1:
            value = value[0]
        curr_result[key].append(value)
    return result

def make_by_pattern_result(data, pattern_results):
    for p, result in pattern_results:
        match_result = re.match(p, data)
        if match_result is not None:
            return result

def make_by_pattern_maker(data, pattern_makers):
    for p, maker in pattern_makers:
        match_result = re.match(p, data)
        if match_result is not None:
            return maker()

def is_positive_uhrs_verified(r):
    uhrs = r['uhrs']
    y, n = uhrs.get('1', 0), uhrs.get('2', 0)
    return y > n

def is_negative_uhrs_verified(r):
    uhrs = r['uhrs']
    y, n = uhrs.get('1', 0), uhrs.get('2', 0)
    return n > y

def find_float_tolorance_unequal(d1, d2):
    # return a list of string. Each string means a path where the value is
    # different
    from past.builtins import basestring
    if all(isinstance(x, basestring) for x in [d1, d2]) or \
            all(type(x) is bool for x in [d1, d2]):
        if d1 != d2:
            return ['0']
        else:
            return []
    if type(d1) is int and type(d2) is int:
        if d1 == d2:
            return []
        else:
            return ['0']
    if type(d1) in [int, float] and type(d2) in [int, float]:
        equal = abs(d1 - d2) <= 0.00001 * abs(d1)
        if equal:
            return []
        else:
            return ['0']
    if isinstance(d1, (dict, OrderedDict)) and isinstance(d2, (dict, OrderedDict)):
        if len(d1) != len(d2):
            return ['0']
        path_d1 = dict_get_all_path(d1, with_type=True)
        result = []
        for p in path_d1:
            v1 = dict_get_path_value(d1, p, with_type=True)
            if not dict_has_path(d2, p, with_type=True):
                result.append(p)
            else:
                v2 = dict_get_path_value(d2, p, with_type=True)
                curr_result = find_float_tolorance_unequal(v1, v2)
                for r in curr_result:
                    result.append(p + '$' + r)
        return result
    if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
        diff = np.absolute((d1 - d2)).sum()
        s = np.absolute(d1).sum()
        equal = diff < 1e-5 * s
        if equal:
            return []
        else:
            return ['0']
    if type(d1) in [tuple, list] and type(d2) in [tuple, list]:
        if len(d1) != len(d2):
            return ['-1']
        result = []
        for i, (x1, x2) in enumerate(zip(d1, d2)):
            curr_result = find_float_tolorance_unequal(x1, x2)
            for r in curr_result:
                result.append('{}${}'.format(i, r))
        return result
    if type(d1) != type(d2):
        return ['0']
    else:
        import torch
        if isinstance(d1, torch.Tensor):
            diff = (d1 - d2).float().abs().sum()
            s = d1.float().abs().sum()
            if float(s) < 1e-5:
                equal = diff < 1e-5
            else:
                equal = float(diff / s) < 1e-5
            if equal:
                return []
            else:
                return ['0']
        else:
            raise Exception('unknown type')

def float_tolorance_equal(d1, d2, check_order=True):
    from past.builtins import basestring
    if isinstance(d1, basestring) and isinstance(d2, basestring):
        return d1 == d2
    if type(d1) in [int, float] and type(d2) in [int, float]:
        return abs(d1 - d2) <= 0.00001 * abs(d1)
    if type(d1) != type(d2) and \
            (not (type(d1) in [tuple, list] and
                type(d2) in [tuple, list])):
        return False
    if type(d1) in [dict, OrderedDict]:
        if len(d1) != len(d2):
            return False
        for k in d1:
            if k not in d2:
                return False
            v1, v2 = d1[k], d2[k]
            if not float_tolorance_equal(v1, v2):
                return False
        return True
    elif type(d1) in [tuple, list]:
        if len(d1) != len(d2):
            return False
        if not check_order:
            d1 = sorted(d1, key=lambda x: pformat(x))
            d2 = sorted(d2, key=lambda x: pformat(x))
        for x1, x2 in zip(d1, d2):
            if not float_tolorance_equal(x1, x2, check_order):
                return False
        return True
    elif type(d1) is bool:
        return d1 == d2
    elif d1 is None:
        return d1 == d2
    elif type(d1) is datetime:
        if d1.tzinfo != d2.tzinfo:
            return d1.replace(tzinfo=d2.tzinfo) == d2
        else:
            return d1 == d2
    elif type(d1) is np.ndarray:
        if not float_tolorance_equal(d1.shape, d2.shape, check_order=True):
            return False
        return np.absolute(d1 - d2).sum() <= 1e-5 * np.absolute(d1).sum()
    elif type(d1) in [np.float64]:
        return np.absolute(d1 - d2).sum() <= 1e-5 * np.absolute(d1).sum()
    else:
        import torch
        if type(d1) is torch.Tensor:
            diff = (d1 - d2).abs().sum()
            s = d1.abs().sum()
            if s < 1e-5:
                return diff < 1e-5
            else:
                return diff / s < 1e-5
        else:
            raise Exception('unknown type')

def case_incensitive_overlap(all_terms):
    all_lower_to_term = [{t.lower(): t for t in terms} for terms in all_terms]
    all_lowers = [set(l.keys()) for l in all_lower_to_term]
    anchor = all_lowers[0].intersection(*all_lowers[1:])

    return [[lower_to_term[l] for l in anchor]
        for lower_to_term in all_lower_to_term]

def get_executable():
    return sys.executable

def collect_process_info():
    result = {}
    for process in psutil.process_iter():
        result[process.pid] = {}
        result[process.pid]['username'] = process.username()
        result[process.pid]['time_spent_in_hour'] = (int(time.time()) -
                process.create_time()) / 3600.0
        result[process.pid]['cmdline'] = ' '.join(process.cmdline())
    return result

def remote_run(str_cmd, ssh_info, return_output=False):
    cmd = ['ssh', '-t', '-t', '-o', 'StrictHostKeyChecking no']
    for key in ssh_info:
        if len(key) > 0 and key[0] == '-':
            cmd.append(key)
            cmd.append(str(ssh_info[key]))
    cmd.append('{}@{}'.format(ssh_info['username'], ssh_info['ip']))
    if is_cluster(ssh_info):
        prefix = 'source ~/.bashrc && export PATH=/usr/local/nvidia/bin:$PATH && '
    else:
        cs = []
        # don't use anaconda since caffe is slower under anaconda because of the
        # data preprocessing. not i/o
        cs.append('source ~/.bashrc')
        if 'conda' in get_executable():
            cs.append('export PATH=$HOME/anaconda3/bin:\$PATH')
            cs.append('export LD_LIBRARY_PATH=$HOME/anaconda3/lib:\$LD_LIBRARY_PATH')
        cs.append('export PATH=/usr/local/nvidia/bin:\$PATH')
        cs.append('export OMP_NUM_THREADS=1')
        prefix = ' && '.join(cs) + ' && '

    suffix = ' && hostname'
    ssh_command = '{}{}{}'.format(prefix, str_cmd, suffix)
    # this will use the environment variable like what you have after ssh
    ssh_command = 'bash -i -c "{}"'.format(ssh_command)
    cmd.append(ssh_command)
    return cmd_run(cmd, return_output)

def compile_by_docker(src_zip, docker_image, dest_zip):
    # compile the qd zip file and generate another one by compiling. so that
    # there is no need to compile it again.
    src_fname = op.basename(src_zip)
    src_folder = op.dirname(src_zip)

    docker_src_folder = '/tmpwork'
    docker_src_zip = op.join(docker_src_folder, src_fname)
    docker_out_src_fname = src_fname + '.out.zip'
    docker_out_zip = op.join(docker_src_folder, docker_out_src_fname)
    out_zip = op.join(src_folder, docker_out_src_fname)
    docker_compile_folder = '/tmpcompile'
    cmd = ['docker', 'run',
            '-v', '{}:{}'.format(src_folder, docker_src_folder),
            docker_image,
            ]
    cmd.append('/bin/bash')
    cmd.append('-c')
    compile_cmd = [
            'mkdir -p {}'.format(docker_compile_folder),
            'cd {}'.format(docker_compile_folder),
            'unzip {}'.format(docker_src_zip),
            'bash compile.aml.sh',
            'zip -yrv x.zip *',
            'cp x.zip {}'.format(docker_out_zip),
            'chmod a+rw {}'.format(docker_out_zip),
            ]
    cmd.append(' && '.join(compile_cmd))
    cmd_run(cmd)
    ensure_directory(op.dirname(dest_zip))
    copy_file(out_zip, dest_zip)

def zip_qd(out_zip, options=None):
    ensure_directory(op.dirname(out_zip))
    cmd = [
        'zip',
        '-uyrv',
        out_zip,
        '*',
    ]
    if options:
        cmd.extend(options)
    else:
        cmd.extend([
            '-x',
            '\*src/CCSCaffe/\*',
            '-x',
            '\*src/build/lib.linux-x86_64-2.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-2.7/\*',
            '-x',
            '\*build/temp.linux-x86_64-2.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.5/\*',
            '-x',
            '\*build/temp.linux-x86_64-3.5/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.7/\*',
            '-x',
            'assets\*',
            '-x',
            '\*build/temp.linux-x86_64-3.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.6/\*',
            '-x',
            '\*build/temp.linux-x86_64-3.6/\*',
            '-x',
            '\*src/detectron2/datasets/\*',
            '-x',
            '\*src/CCSCaffe/models/\*',
            '-x',
            '\*src/CCSCaffe/data/\*',
            '-x',
            '\*src/CCSCaffe/examples/\*',
            '-x',
            '\*src/detectron2/output\*',
            '-x',
            'aux_data/yolo9k/\*',
            '-x',
            'visualization\*',
            '-x',
            'output\*',
            '-x',
            'data\*',
            '-x',
            '\*.build_release\*',
            '-x',
            '\*.build_debug\*',
            '-x',
            '\*.build\*',
            '-x',
            '\*tmp_run\*',
            '-x',
            '\*src/CCSCaffe/MSVC/\*',
            '-x',
            '\*.pyc',
            '-x',
            '\*.so',
            '-x',
            '\*.o',
            '-x',
            '\*src/CCSCaffe/docs/tutorial/\*',
            '-x',
            '\*src/CCSCaffe/matlab/\*',
            '-x',
            '\*.git\*',
            '-x',
            '\*src/qd_classifier/.cache/\*',
            '\*wandb\*',
        ])
    cmd_run(cmd, working_dir=os.getcwd(), shell=True)

def func_retry_agent(info, func, *args, **kwargs):
    i = 0
    num = info.get('retry_times', -1)
    throw_if_fail = info.get('throw_if_fail')
    while True:
        try:
            return func(*args, **kwargs)
        except Exception:
            logging.info('fails: try {}-th time'.format(i))
            print_trace()
            i = i + 1
            if num > 0 and i >= num:
                if throw_if_fail:
                    raise
                else:
                    break
            import time
            time.sleep(random.random() * 5)

def limited_retry_agent(num, func, *args, **kwargs):
    for i in range(num):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning('fails with \n{}: tried {}/{}-th time'.format(
                e,
                i + 1,
                num,
            ))
            import time
            print_trace()
            if i == num - 1:
                #raise
                pass
            else:
                t = random.random() * 5
                time.sleep(t)

def retry_agent(func, *args, **kwargs):
    return func_retry_agent(
        {'retry_times': -1},
        func, *args, **kwargs,
    )

def ensure_copy_folder(src_folder, dst_folder, dst_host=None):
    if op.isfile(src_folder):
        opt = '-vz'
    else:
        opt = '-ravz'
        src_folder = src_folder + '/'
    if dst_host:
        cmd_run(['ssh', dst_host, 'mkdir', '-p', op.dirname(dst_folder)])
    if not dst_host:
        ensure_directory(dst_folder)
        cmd_run('rsync {} {} {} --progress'.format(
            opt,
            src_folder, dst_folder).split(' '))
    else:
        cmd_run('rsync {} {} {}:{} --progress'.format(
            opt,
            src_folder, dst_host, dst_folder).split(' '))

def get_current_time_as_str():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def iter_swap_param_simple(swap_params):
    if isinstance(swap_params, dict):
        swap_params = [[k, v] for k, v in swap_params.items()]
    num = len(swap_params)
    for p in swap_params:
        if type(p[1]) is not list and type(p[1]) is not tuple:
            p[1] = [p[1]]
    counts = [len(p[1]) for p in swap_params]
    assert all(c > 0 for c in counts)
    idx = [0] * num

    while True:
        result = {}
        for p, i in zip(swap_params, idx):
            result[p[0]] = p[1][i]
        yield result

        for i in range(num - 1, -1, -1):
            idx[i] = idx[i] + 1
            if idx[i] < counts[i]:
                break
            else:
                idx[i] = 0
                if i == 0:
                    return

def iter_swap_param(swap_params):
    if isinstance(swap_params, dict):
        swap_params = [(k, v) for k, v in swap_params.items()]
    num = len(swap_params)
    for p in swap_params:
        if type(p[1]) is not list and type(p[1]) is not tuple:
            p[1] = [p[1]]
    counts = [len(p[1]) for p in swap_params]
    empty_keys = [k for k, vs in swap_params if len(vs) == 0]
    assert len(empty_keys) == 0, empty_keys
    idx = [0] * num

    while True:
        result = {}
        for p, i in zip(swap_params, idx):
            key = p[0]
            value = p[1][i]
            if isinstance(key, tuple):
                for sub_key, sub_value in zip(key, value):
                    dict_update_path_value(result, sub_key, sub_value)
            else:
                dict_update_path_value(result, key, value)
        yield result

        for i in range(num - 1, -1, -1):
            idx[i] = idx[i] + 1
            if idx[i] < counts[i]:
                break
            else:
                idx[i] = 0
                if i == 0:
                    return

def gen_uuid():
    import uuid
    return uuid.uuid4().hex

def remove_dir(d):
    ensure_remove_dir(d)

def ensure_remove_file(d):
    if op.isfile(d) or op.islink(d):
        try:
            os.remove(d)
        except:
            pass

@try_once
def ensure_remove_dir(d):
    is_dir = op.isdir(d)
    is_link = op.islink(d)
    if is_dir:
        if not is_link:
            shutil.rmtree(d)
        else:
            os.unlink(d)

def split_to_chunk_to_range(n, num_chunk=None, num_task_each_chunk=None):
    if num_task_each_chunk is None:
        num_task_each_chunk = (n + num_chunk - 1) // num_chunk
    result = []
    i = 0
    while True:
        start = i * num_task_each_chunk
        end = start + num_task_each_chunk
        if start >= n:
            break
        if end > n:
            end = n
        result.append([start, end])
        i = i + 1
    return result

def split_to_chunk(all_task, num_chunk=None, num_task_each_chunk=None):
    if num_task_each_chunk is None:
        num_task_each_chunk = (len(all_task) + num_chunk - 1) // num_chunk
    result = []
    i = 0
    while True:
        start = i * num_task_each_chunk
        end = start + num_task_each_chunk
        if start >= len(all_task):
            break
        if end > len(all_task):
            end = len(all_task)
        result.append(all_task[start:end])
        i = i + 1
    return result

def hash_sha1(s):
    import hashlib
    if type(s) is not str:
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def copy_file(src, dest):
    tmp = dest + '.tmp'
    # we use rsync because it could output the progress
    cmd_run('rsync {} {} --progress'.format(src, tmp).split(' '))
    os.rename(tmp, dest)

def ensure_copy_file(src, dest):
    ensure_directory(op.dirname(dest))
    if not op.isfile(dest):
        copy_file(src, dest)

def decode_to_str(x):
    try:
        return x.decode('utf-8')
    except UnicodeDecodeError:
        return x.decode('latin-1')

def cmd_run(list_cmd,
            return_output=False,
            env=None,
            working_dir=None,
            stdin=sp.PIPE,
            shell=False,
            dry_run=False,
            silent=False,
            process_input=None,
            stdout=None,
            stderr=None,
            no_commute=False,
            timeout=None,
            ):
    if not silent:
        x = ' '.join(map(str, list_cmd)) if isinstance(list_cmd, list) else list_cmd
        logging.info('start to cmd run: {}'.format(x))
        if working_dir:
            logging.info(working_dir)
    # if we dont' set stdin as sp.PIPE, it will complain the stdin is not a tty
    # device. Maybe, the reson is it is inside another process.
    # if stdout=sp.PIPE, it will not print the result in the screen
    e = os.environ.copy()
    if 'SSH_AUTH_SOCK' in e:
        del e['SSH_AUTH_SOCK']
    if working_dir:
        ensure_directory(working_dir)
    if env:
        for k in env:
            e[k] = env[k]
    if dry_run:
        # we need the log result. Thus, we do not return at teh very beginning
        return
    if not return_output:
        #if env is None:
            #p = sp.Popen(list_cmd, stdin=sp.PIPE, cwd=working_dir)
        #else:
        p = sp.Popen(' '.join(list_cmd) if shell else list_cmd,
                     stdin=stdin,
                     env=e,
                     shell=shell,
                     stdout=stdout,
                     cwd=working_dir,
                     stderr=stderr,
                     )
        if not no_commute:
            message = p.communicate(input=process_input, timeout=timeout)
            if p.returncode != 0:
                message = 'message = {}; cmd = {}'.format(
                    message, ' '.join(list_cmd))
                if stderr == sp.PIPE and not p.stderr:
                    message += '; stderr = {}'.format(p.stderr.read().decode())
                raise ValueError(message)
            return message
        else:
            return p
    else:
        if isinstance(list_cmd, list) and shell:
            list_cmd = ' '.join(list_cmd)
        message = sp.check_output(list_cmd,
                                  env=e,
                                  cwd=working_dir,
                                  shell=shell,
                                  timeout=timeout,
                                  )
        if not silent:
            logging.info('finished the cmd run')
        return decode_to_str(message)

def parallel_imap(func, all_task, num_worker=16):
    if num_worker > 0:
        #from multiprocessing import Pool
        from pathos.multiprocessing import Pool
        m = Pool(num_worker)
        result = []
        for x in qd_tqdm(m.imap(func, all_task), total=len(all_task)):
            result.append(x)
        # there are some error comes out from os.fork() and which says
        # OSError: [Errno 24] Too many open files.
        # self.pid = os.fork()
        # here, we explicitly close the pool and see if it helps. note, this is
        # not verified to work, if we still see that kind of error message, we
        # need other solutions
        m.close()
        m.join()
        return result
    else:
        result = []
        for t in all_task:
            result.append(func(t))
        return result

def get_azfuse_env(v, d=None):
    # this is for back-compatibility only
    qd_k = 'QD_' + v
    if qd_k in os.environ:
        return os.environ[qd_k]
    azfuse_k = 'AZFUSE_' + v
    if azfuse_k in os.environ:
        return os.environ[azfuse_k]
    return d

def get_tmp_folder():
    folder = os.environ.get('QD_TMP_FOLDER', '/raid/tmp')
    return folder

# use parallel_imap if possible
def parallel_map(func, all_task, num_worker=16):
    if num_worker > 0:
        from pathos.multiprocessing import ProcessingPool as Pool
        with Pool(num_worker) as m:
            result = m.map(func, all_task)
        return result
    else:
        result = []
        for t in all_task:
            result.append(func(t))
        return result

def url_to_file_by_wget(url, fname):
    ensure_directory(op.dirname(fname))
    cmd_run(['wget', url, '-O', fname])

@functools.lru_cache(maxsize=1)
def logging_once(s):
    logging.info(s)

# this is specifically for azure blob url, where the last 1k bytes operation is
# not supported. We have to first find the length and then find the start
# point
def get_url_fsize(url):
    result = cmd_run(['curl', '-sI', url], return_output=True)
    for row in result.split('\n'):
        ss = [s.strip() for s in row.split(':')]
        if len(ss) == 2 and ss[0] == 'Content-Length':
            size_in_bytes = int(ss[1])
            return size_in_bytes

def url_to_file_by_curl(url, fname, bytes_start=None, bytes_end=None):
    ensure_directory(op.dirname(fname))
    if bytes_start == 0 and bytes_end == 0:
        cmd_run(['touch', fname])
        return
    if bytes_start is None:
        bytes_start = 0
    elif bytes_start < 0:
        size = get_url_fsize(url)
        bytes_start = size + bytes_start
        if bytes_start < 0:
            bytes_start = 0
    if bytes_end is None:
        # -f: if it fails, no output will be sent to output file
        if bytes_start == 0:
            cmd_run(['curl', '-f',
                url, '--output', fname])
        else:
            cmd_run(['curl', '-f', '-r', '{}-'.format(bytes_start),
                url, '--output', fname])
    else:
        # curl: end is inclusive
        cmd_run(['curl', '-f', '-r', '{}-{}'.format(bytes_start, bytes_end - 1),
            url, '--output', fname])

def url_to_bytes(url):
    try:
        fp = urlopen(url, timeout=30)
        buf = fp.read()
        real_url = fp.geturl()
        if real_url != url and (not real_url.startswith('https') or
                real_url.replace('https', 'http') != url):
            logging.info('new url = {}; old = {}'.format(fp.geturl(), url))
            # the image gets redirected, which means the image is not available
            return None
        return buf
    except HTTPError as err:
        logging.error("url: {}; error code {}; message: {}".format(
            url, err.code, err.msg))
        return None
    except:
        import traceback
        logging.error("url: {}; unknown {}".format(
            url, traceback.format_exc()))
        return None

def url_to_str(url):
    try:
        fp = urlopen(url, timeout=30)
        buf = fp.read()
        real_url = fp.geturl()
        if real_url != url and (not real_url.startswith('https') or
                real_url.replace('https', 'http') != url):
            logging.info('new url = {}; old = {}'.format(fp.geturl(), url))
            # the image gets redirected, which means the image is not available
            return None
        if type(buf) is str:
            # py2
            return buf
        else:
            # py3
            return buf.decode()
    except HTTPError as err:
        logging.error("url: {}; error code {}; message: {}".format(
            url, err.code, err.msg))
        return None
    except:
        logging.error("url: {}; unknown {}".format(
            url, traceback.format_exc()))
        return None

def image_url_to_bytes(url):
    req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
    try:
        response = urlopen(req, None, 10)
        if response.code != 200:
            logging.info("url: {}, error code: {}".format(url, response.code))
            return None
        data = response.read()
        response.close()
        return data
    except Exception as e:
        logging.info("error downloading: {}".format(e))
    return None

def str_to_image(buf):
    image = np.asarray(bytearray(buf), dtype='uint8')
    im = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return im

def bytes_to_image(bs):
    image = np.asarray(bytearray(bs), dtype='uint8')
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def url_to_image(url):
    buf = url_to_bytes(url)
    if buf is None:
        return None
    else:
        image = np.asarray(bytearray(buf), dtype='uint8')
        return cv2.imdecode(image, cv2.IMREAD_COLOR)

def normalize_to_str(s):
    if sys.version_info.major == 3:
        return s.encode("ascii", "ignore").decode()
    else:
        if type(s) is str:
            s = s.decode('unicode_escape')
        import unicodedata
        return unicodedata.normalize('NFKD', s).encode('ascii','ignore')

def query_wiki_info(query_term):
    query_term = '{} site:en.wikipedia.org'.format(query_term)
    rls = limited_retry_agent(10, scrape_bing_general_rich, query_term, 1)
    if not rls:
        return {'query_term': query_term}
    rl = rls[0]
    n_title = normalize_to_str(rl['title'])
    result = re.match('(.*) - Wikipedia', n_title)
    if result:
        best_name = result.groups()[0]
    else:
        best_name = query_term
    result = {
            'query_term': query_term,
            'best_name': best_name,
            'wiki_tile': rl['title'],
            'norm_wiki_title': normalize_to_str(rl['title']),
            'wiki_url': rl['url']}
    #logging.info(pformat(result))
    return result

def scrape_bing_general_rich(query_term, depth):
    '''
    note, the order of url list is not the same as the web query. Even we keep
    the order of how to add it to the result, the order is still not the same.
    '''
    # we might add duplicated terms. need 1) deduplicate, 2) keep the order
    # when it is added
    import requests
    import xml.etree.ElementTree as ET
    format_str = \
            'http://www.bing.com/search?q={}&form=MONITR&qs=n&format=pbxml&first={}&count={}&fdpriority=premium&mkt=en-us'
    start = 0
    all_result = []
    while True:
        count = min(depth - start, 150)
        if count <= 0:
            break
        query_str = format_str.format(query_term, start, count)
        start = start + count
        r = requests.get(query_str, allow_redirects=True)
        content = r.content
        #content = urllib2.urlopen(query_str).read()
        root = ET.fromstring(content)
        for t in root.iter('k_AnswerDataKifResponse'):
            if t.text is None:
                continue
            text_result = json.loads(t.text)
            if 'results' not in text_result:
                continue
            results = text_result['results']
            for r in results:
                rl = {k.lower() : r[k] for k in r}
                if 'url' in rl and 'title' in rl and len(rl['title']) > 0:
                    all_result.append(rl)
    url_to_result = {}
    for rl in all_result:
        url = rl['url']
        if url not in url_to_result:
            url_to_result[url] = rl

    return list(url_to_result.values())

def request_by_browser(url):
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    import bs4
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    soup = bs4.BeautifulSoup(driver.page_source, features='lxml')
    # if we return immediately, the page_source might not be ready
    time.sleep(1)
    soup = bs4.BeautifulSoup(driver.page_source, features='lxml')
    return soup

def iter_bing_visual_search(query_url, origin_url=True):
    format_str = 'http://www.bing.com/images/searchbyimage?FORM=IRSBIQ&cbir=sbi&imgurl={0}'
    # the following two parameters are not valid
    #format_str += '&first=100'
    #format_str += '&count=10'
    bing_url = format_str.format(query_url)
    soup = request_by_browser(bing_url)
    html_keywords = ['richImage relImg', 'richImage relProd flyout']
    alts = ['See related image detail', 'See related product detail']
    caption_classes = ['span', 'a']
    for html_key_word, alt, caption_class in zip(html_keywords, alts, caption_classes):
        for i, container in enumerate(soup.find_all(class_= html_key_word)):
            # one container has one image and one caption container. we will
            # extract the image and the caption, which might be helpful in the
            # future
            info = {'rank': i}
            info['html_keyword'] = html_key_word
            # original url
            if origin_url:
                imgs = container.find_all(class_='richImgLnk')
                if len(imgs) == 1:
                    img = imgs[0]
                    url = 'http://www.bing.com/images/search' + img.attrs['href']
                    result = request_by_browser(url)
                    imgs = result.find_all(alt='See the source image')
                    if len(imgs) == 1:
                        img = imgs[0]
                        url = img.attrs['src']
                        info['url'] = url

            # bing cache image
            imgs = container.find_all('img', alt=alt)
            if len(imgs) == 1:
                bing_cache_url = 'http://www.bing.com' + imgs[0].attrs['src']
                info['bing_cache_url'] = bing_cache_url

            captions = container.find_all(caption_class, class_='tit')
            if len(captions) == 1:
                cap = captions[0]
                info['caption'] = cap.text
            yield info

def scrape_bing_general(query_term, depth):
    '''
    note, the order of url list is not the same as the web query. Even we keep
    the order of how to add it to the result, the order is still not the same.
    '''
    # we might add duplicated terms. need 1) deduplicate, 2) keep the order
    # when it is added
    import requests
    import xml.etree.ElementTree as ET
    format_str = \
            'http://www.bing.com/search?q={}&form=MONITR&qs=n&format=pbxml&first={}&count={}&fdpriority=premium&mkt=en-us'
    start = 0
    all_url = []
    while True:
        count = min(depth - start, 150)
        if count <= 0:
            break
        query_str = format_str.format(query_term, start, count)
        start = start + count
        r = requests.get(query_str, allow_redirects=True)
        content = r.content
        #content = urllib2.urlopen(query_str).read()
        root = ET.fromstring(content)
        for t in root.iter('k_AnswerDataKifResponse'):
            if t.text is None:
                continue
            text_result = json.loads(t.text)
            if 'results' not in text_result:
                continue
            results = text_result['results']
            for r in results:
                rl = {k.lower() : r[k] for k in r}
                if 'url' in rl:
                    url = rl['url']
                    all_url.append(url)
    return list(set(all_url))

def scrape_bing(query_term, depth, trans_bg=False):
    ''' this is for image; for text, use scrape_bing_general
    e.g. scrape_bing('elder person', 300)
    '''
    import requests
    import xml.etree.ElementTree as ET
    format_str = \
            'http://www.bing.com/images/search?q={}&form=MONITR&qs=n&format=pbxml&first={}&count={}&fdpriority=premium&mkt=en-us'
    start = 0
    all_url = []
    while True:
        count = min(depth - start, 150)
        if count <= 0:
            break
        query_str = format_str.format(query_term, start, count)
        if trans_bg:
            query_str += "&&qft=+filterui:photo-transparent"
        start = start + count
        logging.info(query_str)
        r = requests.get(query_str, allow_redirects=True)
        content = r.content
        #content = urllib2.urlopen(query_str).read()
        root = ET.fromstring(content)
        for t in root.iter('k_AnswerDataKifResponse'):
            results = json.loads(t.text)['results']
            for r in results:
                rl = {k.lower() : r[k] for k in r}
                media_url = rl.get('mediaurl', '')
                #url = rl.get('url', '')
                #title = rl.get('title', '')
                all_url.append(media_url)
            break
    return all_url


def calculate_correlation_between_terms(iter1, iter2):
    label_to_num1 = {}
    label_to_num2 = {}
    ll_to_num = {}

    for (k1, str_rects1), (k2, str_rects2) in zip(iter1, iter2):
        assert k1 == k2, 'keys should be aligned ({} != {})'.format(k1, k2)
        rects1 = json.loads(str_rects1)
        rects2 = json.loads(str_rects2)
        for r in rects1:
            c = r['class']
            label_to_num1[c] = label_to_num1.get(c, 0) + 1
        for r in rects2:
            c = r['class']
            label_to_num2[c] = label_to_num2.get(c, 0) + 1
        for r1 in rects1:
            for r2 in rects2:
                i = calculate_iou(r1['rect'], r2['rect'])
                if i > 0.01:
                    k = (r1['class'], r2['class'])
                    ll_to_num[k] = ll_to_num.get(k, 0) + i
    ll_correlation = [(ll[0], ll[1], 1. * ll_to_num[ll] / (label_to_num1[ll[0]]
        + label_to_num2[ll[1]] - ll_to_num[ll]))
        for ll in ll_to_num]
    ll_correlation = [(left, right, c) for left, right, c in ll_correlation
            if left.lower() != right.lower()]
    ll_correlation = sorted(ll_correlation, key=lambda x: -x[2])

    return ll_correlation

def json_dump(obj):
    # order the keys so that each operation is deterministic though it might be
    # slower
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))

def set_if_not_exist(d, key, value):
    if key not in d:
        d[key] = value

def print_as_html(table, html_output):
    from jinja2 import Environment, FileSystemLoader
    j2_env = Environment(loader=FileSystemLoader('./'), trim_blocks=True)
    # find the cols with longest length. If it does not include all cols, then
    # append those not included
    _, cols = max([(len(table[row]), table[row]) for row in table],
            key=lambda x: x[0])
    cols = list(cols)
    for row in table:
        for c in table[row]:
            if c not in cols:
                cols.append(c)
    r = j2_env.get_template('aux_data/html_template/table_viewer.html').render(
        table=table,
        rows=table.keys(),
        cols=cols)
    write_to_file(r, html_output)

def jinja_render(template, **kwargs):
    if len(kwargs) == 0:
        return read_to_buffer(template).decode()
    from jinja2 import Environment, FileSystemLoader
    j2_env = Environment(loader=FileSystemLoader('./'), trim_blocks=True)
    return j2_env.get_template(template).render(
        **kwargs,
    )

def parse_general_args():
    parser = argparse.ArgumentParser(description='General Parser')
    parser.add_argument('-c', '--config_file', help='config file',
            type=str)
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-bp', '--base64_param', help='base64 encoded yaml format',
            type=str)
    args = parser.parse_args()
    kwargs =  {}
    if args.config_file:
        logging.info('loading parameter from {}'.format(args.config_file))
        configs = load_from_yaml_file(args.config_file)
        for k in configs:
            kwargs[k] = configs[k]
    if args.base64_param:
        configs = load_from_yaml_str(base64.b64decode(args.base64_param))
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    if args.param:
        configs = load_from_yaml_str(args.param)
        dict_ensure_path_key_converted(configs)
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    return kwargs

class ProgressBar(object):
    def __init__(self, maxval):
        assert maxval > 0
        self.maxval = maxval

    def __enter__(self):
        self.pbar = progressbar.ProgressBar(maxval=self.maxval).start()
        return self

    def __exit__(self, t, v, traceback):
        self.update(self.maxval)
        sys.stdout.write('\n')

    def update(self, i):
        self.pbar.update(i)

#@deprecated('use tsv_io.concat_files')
def concat_files(ins, out):
    ensure_directory(op.dirname(out))
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)

def get_mpi_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

def get_mpi_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))

def get_mpi_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))

def get_mpi_local_size():
    if 'LOCAL_SIZE' in os.environ:
        return int(os.environ['LOCAL_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))

def load_class_ap(full_expid, predict_file):
    # the normal map
    report_file = op.splitext(predict_file)[0] + '.report'
    fname = op.join('output', full_expid, 'snapshot', report_file +
            '.class_ap.json')
    if op.isfile(fname):
        return json.loads(read_to_buffer(fname))

    glob_pattern = op.splitext(predict_file)[0] + '.neg_aware_gmap*report'
    fnames = glob.glob(op.join('output', full_expid, 'snapshot',
        glob_pattern))
    if len(fnames) > 0 and op.isfile(fnames[0]):
        fname = fnames[0]
        result = load_from_yaml_file(fname)
        return {'overall': {'0.5': {'class_ap': result['class_ap']}}}


def calculate_ap_by_true_list(corrects, total):
    precision = (1. * np.cumsum(corrects)) / np.arange(1, 1 + len(corrects))
    if np.sum(corrects) == 0:
        return 0
    return np.sum(precision * corrects) / total

def calculate_ap_by_true_list_count_num(corrects, total):
    precision = (1. * np.cumsum(corrects)) / np.arange(1, 1 + len(corrects))
    if np.sum(corrects) == 0:
        return 0
    return np.sum(precision) / len(precision) * np.sum(corrects) / total

def calculate_weighted_ap_by_true_list(corrects, weights, total):
    precision = np.cumsum(corrects * weights) / (np.cumsum(weights) + 0.0001)
    if total == 0:
        return 0
    return np.mean(precision) * np.sum(corrects) / total

def calculate_ap_by_true_list_100(corrects, confs, total):
    precision = (1. * np.cumsum(corrects)) / map(lambda x: 100. * (1 - x) + 1, confs)
    return np.sum(precision * corrects) / total

def calculate_image_ap_weighted(predicts, gts, weights):
    corrects, _ = match_prediction_to_gt(predicts, gts)
    return calculate_weighted_ap_by_true_list(corrects, weights, len(gts))

def match_prediction_to_gt(predicts, gts, iou_th=0.5):
    matched = [False] * len(gts)
    corrects = np.zeros(len(predicts))
    match_idx = [-1] * len(predicts)
    for j, p in enumerate(predicts):
        idx_gts = [(i, g) for i, g in enumerate(gts) if not matched[i]]
        if len(idx_gts) == 0:
            # max does not support empty input
            continue
        idx_gt_ious = [(i, g, calculate_iou(p, g)) for i, g in idx_gts]
        max_idx, _, max_iou = max(idx_gt_ious, key=lambda x: x[-1])
        if max_iou > iou_th:
            matched[max_idx] = True
            corrects[j] = 1
            match_idx[j] = max_idx
    return corrects, match_idx

def calculate_image_ap(predicts, gts, count_num=False):
    '''
    a list of rects, use 2 to return more info
    '''
    corrects, _ = match_prediction_to_gt(predicts, gts)
    if not count_num:
        return calculate_ap_by_true_list(corrects, len(gts))
    else:
        return calculate_ap_by_true_list_count_num(corrects, len(gts))


def calculate_image_ap2(predicts, gts):
    '''
    a list of rects
    '''
    corrects, match_idx = match_prediction_to_gt(predicts, gts)
    return calculate_ap_by_true_list(corrects, len(gts)), match_idx

def get_parameters_by_full_expid(full_expid):
    yaml_file = op.join('output', full_expid, 'parameters.yaml')
    if not op.isfile(yaml_file):
        return None
    param = load_from_yaml_file(yaml_file)
    if 'data' not in param:
        param['data'], param['net'] = parse_data_net(full_expid,
                param['expid'])
    return param

def get_all_model_expid():
    names = os.listdir('./output')
    return names

def get_target_images(predicts, gts, cat, threshold):
    image_aps = []
    for key in predicts:
        rects = predicts[key]
        curr_gt = [g for g in gts[key] if cat == 'any' or g['class'] == cat]
        curr_pred = [p for p in predicts[key] if cat == 'any' or (p['class'] == cat and
                p['conf'] > threshold)]
        if len(curr_gt) == 0 and len(curr_pred) == 0:
            continue
        curr_pred = sorted(curr_pred, key=lambda x: -x['conf'])
        ap = calculate_image_ap([p['rect'] for p in curr_pred],
                [g['rect'] for g in curr_gt])
        image_aps.append((key, ap))
    image_aps = sorted(image_aps, key=lambda x: x[1])
    #image_aps = sorted(image_aps, key=lambda x: -x[1])
    target_images = [key for key, ap in image_aps]
    return target_images, image_aps

def readable_confusion_entry(entry):
    '''
    entry: dictionary, key: label, value: count
    '''
    label_count = [(label, entry[label]) for label in entry]
    label_count.sort(key=lambda x: -x[1])
    total = sum([count for label, count in label_count])
    percent = [1. * count / total for label, count in label_count]
    cum_percent = np.cumsum(percent)
    items = []
    for i, (label, count) in enumerate(label_count):
        if i >= 5:
            continue
        items.append((label, '{}'.format(count), '{:.1f}'.format(100. *
            percent[i]),
            '{:.1f}'.format(100. * cum_percent[i])))
    return items

def get_all_tree_data():
    names = sorted(os.listdir('./data'))
    return [name for name in names
        if op.isfile(op.join('data', name, 'root_enriched.yaml'))]

def parse_test_data_with_version_with_more_param(predict_file):
    pattern = \
        'model(?:_iter)?_-?[0-9]*[e]?\.(?:caffemodel|pth\.tar|pth|pt)\.(.*)\.(trainval|train|test|train_[0-9]*_[0-9]*)\..*?(\.v[0-9])?\.(?:predict|report)'
    match_result = re.match(pattern, predict_file)
    if match_result:
        assert match_result
        result = match_result.groups()
        if result[2] is None:
            v = 0
        else:
            v = int(result[2][2])
        return result[0], result[1], v

def parse_test_data_with_version(predict_file):
    # with version
    result = parse_test_data_with_version_with_more_param(predict_file)
    if result is not None:
        return result
    #model_iter_0040760.TaxCaptionBot.trainval.predict.report
    all_pattern = [
        'model(?:_iter)?_-?[0-9]*[e]?\.(?:caffemodel|pth\.tar|pth|pt)\.(.*)\.(trainval|train|test|val)\.(\.v[0-9])?(?:predict|report)',
        'model_iter_[0-9]*\.(.*)\.(trainval|train|test|val)\..*predict\.(?:ir_acc|caption|vqa_acc)\.report',
        'model(?:_iter)?_-?[0-9]*[e]?\.(.*)\.(trainval|train|test|val)\.(\.v[0-9])?.*(?:predict|report|tsv)',
        'model_iter_[0-9]*\.(.*)\.([a-zA-Z0-9]+)\..*predict\.(?:ir_acc|caption|vqa_acc)\.report',
    ]
    for p in all_pattern:
        match_result = re.match(p, predict_file)
        if match_result is not None:
            result = match_result.groups()
            if len(result) == 2 or result[2] is None:
                v = 0
            else:
                v = int(result[2][2:])
            return result[0], result[1], v

    pattern = \
        'model(?:_iter)?_-?[0-9]*[e]?\.(?:caffemodel|pth\.tar|pth|pt)\.([^\.]*).*?(\.v[0-9])?\.(?:predict|report)'
    match_result = re.match(pattern, predict_file)
    assert match_result
    result = match_result.groups()
    if result[1] is None:
        v = 0
    else:
        v = int(result[1][2:])
    return result[0], 'test', v

def parse_test_data(predict_file):
    return parse_test_data_with_version(predict_file)[:2]

def parse_data(full_expid):
    all_data = os.listdir('data/')
    candidates = [data for data in all_data if full_expid.startswith(data)]
    max_length = max([len(c) for c in candidates])
    return [c for c in candidates if len(c) == max_length][0]

def parse_iteration(file_name):
    patterns = [
        '.*model(?:_iter)?_([0-9]*)\..*',
        '.*model(?:_iter)?_([0-9]*)e\..*',
        '.*model(?:_iter)?_([0-9]*)$',
    ]
    for p in patterns:
        r = re.match(p, file_name)
        if r is not None:
            return int(float(r.groups()[0]))
    logging.info('unable to parse the iterations for {}'.format(file_name))
    return -2

def parse_snapshot_rank(predict_file):
    '''
    it could be iteration, or epoch
    '''
    pattern = 'model_iter_([0-9]*)e*\.|model_([0-9]*)e*\.pth'
    match_result = re.match(pattern, predict_file)
    if match_result is None:
        return -1
    else:
        matched_iters = [r for r in match_result.groups() if r is not None]
        assert len(matched_iters) == 1
        return int(matched_iters[0])

def get_all_predict_files(full_expid):
    model_folder = op.join('output', full_expid, 'snapshot')

    predict_files = []

    found = glob.glob(op.join(model_folder, '*.predict'))
    predict_files.extend([op.basename(f) for f in found])

    found = glob.glob(op.join(model_folder, '*.predict.tsv'))
    predict_files.extend([op.basename(f) for f in found])

    iterations = [(parse_snapshot_rank(p), p) for p in predict_files]
    iterations.sort(key=lambda x: -x[0])
    return [p for i, p in iterations]

def dict_to_list(d, idx):
    result = []
    for k in d:
        vs = d[k]
        for v in vs:
            try:
                r = []
                # if v is a list or tuple
                r.extend(v[:idx])
                r.append(k)
                r.extend(v[idx: ])
            except TypeError:
                r = []
                if idx == 0:
                    r.append(k)
                    r.append(v)
                else:
                    assert idx == 1
                    r.append(v)
                    r.append(k)
            result.append(r)
    return result

def list_to_dict_unique(l, idx):
    result = list_to_dict(l, idx)
    for key in result:
        result[key] = list(set(result[key]))
    return result

def list_to_dict(l, idx, keep_one=False):
    result = OrderedDict()
    for x in l:
        if x[idx] not in result:
            result[x[idx]] = []
        y = x[:idx] + x[idx + 1:]
        if not keep_one and len(y) == 1:
            y = y[0]
        result[x[idx]].append(y)
    return result

def generate_lineidx(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        fbar_last_pos = 0
        fbar = qd_tqdm(total=fsize, unit_scale=True)
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n");
            tsvin.readline()
            fpos = tsvin.tell();
            fbar.update(fpos - fbar_last_pos)
            fbar_last_pos = fpos
    os.rename(idxout_tmp, idxout)

def drop_second_batch_in_bn(net):
    assert net.layer[0].type == 'TsvBoxData'
    assert net.layer[1].type == 'TsvBoxData'
    slice_batch_layers = [l for l in net.layer if l.name == 'slice_batch']
    assert len(slice_batch_layers) == 1
    slice_batch_layer = slice_batch_layers[0]
    slice_point = slice_batch_layer.slice_param.slice_point[0]

    for i, l in enumerate(net.layer):
        if l.type == 'BatchNorm':
            top_name = l.top[0]
            top_name2 = top_name + '_n'
            l.top[0] = top_name2
            for m in net.layer[i + 1:]:
                for j, b in enumerate(m.bottom):
                    if b == top_name:
                        m.bottom[j] = top_name2
                for j, t in enumerate(m.top):
                    if t == top_name:
                        m.top[j] = top_name2

    all_origin_layer = []
    for l in net.layer:
        all_origin_layer.append(l)
    all_layer = []
    for l in all_origin_layer:
        if l.type != 'BatchNorm':
            all_layer.append(l)
            continue
        bn_input = l.bottom[0]
        bn_output = l.top[0]

        slice_layer = net.layer.add()
        slice_layer.name = l.name + '/slice'
        slice_layer.type = 'Slice'
        slice_layer.bottom.append(bn_input)
        slice_layer.top.append(l.name + '/slice0')
        slice_layer.top.append(l.name + '/slice1')
        slice_layer.slice_param.axis = 0
        slice_layer.slice_param.slice_point.append(slice_point)
        all_layer.append(slice_layer)

        l.bottom.remove(l.bottom[0])
        l.bottom.append(l.name + '/slice0')
        l.top.remove(l.top[0])
        l.top.append(l.name + '/slice0')
        all_layer.append(l)

        fix_bn_layer = net.layer.add()
        fix_bn_layer.name = l.name + '/bn1'
        fix_bn_layer.bottom.append(l.name + '/slice1')
        fix_bn_layer.top.append(l.name + '/slice1')
        fix_bn_layer.type = 'BatchNorm'
        for _ in range(3):
            p = fix_bn_layer.param.add()
            p.lr_mult = 0
            p.decay_mult = 0
        fix_bn_layer.batch_norm_param.use_global_stats = True
        all_layer.append(fix_bn_layer)

        cat_layer = net.layer.add()
        cat_layer.name = l.name + '/concat'
        cat_layer.type = 'Concat'
        cat_layer.bottom.append(l.name + '/slice0')
        cat_layer.bottom.append(l.name + '/slice1')
        cat_layer.top.append(bn_output)
        cat_layer.concat_param.axis = 0
        all_layer.append(cat_layer)

    while len(net.layer) > 0:
        net.layer.remove(net.layer[0])
    net.layer.extend(all_layer)

def fix_net_bn_layers(net, num_bn_fix):
    for l in net.layer:
        if l.type == 'BatchNorm':
            if num_bn_fix > 0:
                l.batch_norm_param.use_global_stats = True
                num_bn_fix = num_bn_fix - 1
            else:
                break

def is_cluster(ssh_info):
    return '-p' in ssh_info and '-i' not in ssh_info

def visualize_net(net):
    delta = 0.000001
    data_values = []
    for key in net.blobs:
        data_value = np.mean(np.abs(net.blobs[key].data))
        data_values.append(data_value + delta)
    diff_values = []
    for key in net.blobs:
        diff_values.append(np.mean(np.abs(net.blobs[key].diff))
            + delta)
    param_keys = []
    param_data = []
    for key in net.params:
        for i, b in enumerate(net.params[key]):
            param_keys.append('{}_{}'.format(key, i))
            param_data.append(np.mean(np.abs(b.data)) + delta)
    param_diff = []
    for key in net.params:
        for i, b in enumerate(net.params[key]):
            param_diff.append(np.mean(np.abs(b.diff)) + delta)

    xs = range(len(net.blobs))
    plt.gcf().clear()
    plt.subplot(2, 1, 1)

    plt.semilogy(xs, data_values, 'r-o')
    plt.semilogy(xs, diff_values, 'b-*')
    plt.xticks(xs, net.blobs.keys(), rotation='vertical')
    plt.grid()

    plt.subplot(2, 1, 2)
    xs = range(len(param_keys))
    plt.semilogy(xs, param_data, 'r-o')
    plt.semilogy(xs, param_diff, 'b-*')
    plt.xticks(xs, param_keys, rotation='vertical')
    plt.grid()
    plt.draw()
    plt.pause(0.001)

def visualize_train(solver):
    plt.figure()
    features = []
    for i in range(100):
        visualize_net(solver.net)
        solver.step(10)

def network_input_to_image(data, mean_value, std_value=[1.0,1.0,1.0]):
    all_im = []
    for d in data:
        im = (d.transpose((1, 2, 0))
            * np.asarray(std_value).reshape(1, 1, 3)
            + np.asarray(mean_value).reshape(1, 1, 3)
            ).astype(np.uint8).copy()
        all_im.append(im)
    return all_im

def remove_data_augmentation(data_layer):
    assert data_layer.type == 'TsvBoxData'
    data_layer.box_data_param.jitter = 0
    data_layer.box_data_param.hue = 0
    data_layer.box_data_param.exposure = 1
    data_layer.box_data_param.random_scale_min = 1
    data_layer.box_data_param.random_scale_max = 1
    data_layer.box_data_param.fix_offset = True
    data_layer.box_data_param.saturation = True

def check_best_iou(biases, gt_w, gt_h, n):
    def iou(gt_w, gt_h, w, h):
        inter = min(gt_w, w) * min(gt_h, h)
        return inter / (gt_w * gt_h + w * h - inter)

    best_iou = -1
    best_n = -1
    for i in range(len(biases) / 2):
        u = iou(gt_w, gt_h, biases[2 * i], biases[2 * i + 1])
        if u > best_iou:
            best_iou = u
            best_n = i
    assert best_n == n

def calculate_iou1(rect0, rect1):
    '''
    x0, y1, x2, y3
    '''
    w = min(rect0[2], rect1[2]) - max(rect0[0], rect1[0]) + 1
    if w < 0:
        return 0
    h = min(rect0[3], rect1[3]) - max(rect0[1], rect1[1]) + 1
    if h < 0:
        return 0
    i = w * h
    a1 = (rect1[2] - rect1[0] + 1) * (rect1[3] - rect1[1] + 1)
    a0 = (rect0[2] - rect0[0] + 1) * (rect0[3] - rect0[1] + 1)
    if a0 == 0 and a1 == 0 and i == 0:
        return 1.
    return 1. * i / (a0 + a1 - i)

def calculate_iou_xywh(r0, r1):
    r0 = [r0[0] - r0[2] / 2.,
            r0[1] - r0[3] / 2.,
            r0[0] + r0[2] / 2.,
            r0[1] + r0[3] / 2.]
    r1 = [r1[0] - r1[2] / 2.,
            r1[1] - r1[3] / 2.,
            r1[0] + r1[2] / 2.,
            r1[1] + r1[3] / 2.]

    return calculate_iou(r0, r1)

def calculate_iou(rect0, rect1):
    '''
    x0, y1, x2, y3
    '''
    w = min(rect0[2], rect1[2]) - max(rect0[0], rect1[0])
    if w < 0:
        return 0
    h = min(rect0[3], rect1[3]) - max(rect0[1], rect1[1])
    if h < 0:
        return 0
    i = w * h
    a1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    a0 = (rect0[2] - rect0[0]) * (rect0[3] - rect0[1])
    if a0 == 0 and a1 == 0 and i == 0:
        return 1.
    return 1. * i / (a0 + a1 - i)

#def process_run(func, *args, **kwargs):
    #def internal_func(queue):
        #result = func(*args, **kwargs)
        #queue.put(result)
    #queue = mp.Queue()
    #p = Process(target=internal_func, args=(queue,))
    #p.start()
    #p.join()
    #assert p.exitcode == 0
    #return queue.get()

class ExceptionWrapper(object):
    def __init__(self, m):
        self.message = m

def process_run(func, *args, **kwargs):
    def internal_func(queue):
        try:
            result = func(*args, **kwargs)
            queue.put(result)
        except Exception:
            queue.put(ExceptionWrapper(traceback.format_exc()))
    queue = mp.Queue()
    p = Process(target=internal_func, args=(queue,))
    p.start()
    result = queue.get()
    p.join()
    if isinstance(result, ExceptionWrapper):
        raise Exception(result.message)
    return result

def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)
    try:
        yaml.add_representer(unicode, unicode_representer)
    except NameError:
        logging.info('python 3 env')

def init_logging():
    np.seterr(divide = "raise", over="warn", under="warn",  invalid="raise")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
    ch.setFormatter(logger_fmt)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(ch)
    root.setLevel(logging.INFO)

    setup_yaml()

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path, exist_ok=True)
            except FileExistsError:
                # Ignore the exception since the directory already exists.
                pass
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise

def parse_pattern(pattern, s):
    result = parse_pattern_as_is(pattern, s)
    if result is None:
        return
    return [float(g) for g in result]

def parse_pattern_as_is(pattern, s):
    result = re.search(pattern, s)
    if result is None:
        return result
    return [g for g in result.groups()]

def iter_match_document(pattern, fname):
    for line in read_lines(fname):
        result = parse_pattern_as_is(pattern, line)
        if result is None:
            continue
        yield result

def parse_yolo_log(log_file):
    pattern = 'loss_xy: ([0-9, .]*); loss_wh: ([0-9, .]*); '
    pattern = pattern + 'loss_objness: ([0-9, .]*); loss_class: ([0-9, .]*)'

    base_log_lines = read_lines(log_file)
    xys = []
    whs = []
    loss_objnesses = []
    loss_classes = []
    for line in base_log_lines:
        gs = parse_pattern(pattern, line)
        if gs is None:
            continue
        idx = 0
        xys.append(float(gs[idx]))
        idx = idx + 1
        whs.append(float(gs[idx]))
        idx = idx + 1
        loss_objnesses.append(float(gs[idx]))
        idx = idx + 1
        loss_classes.append(float(gs[idx]))

    return xys, whs, loss_objnesses, loss_classes

def parse_nums(p, log_file):
    result = []
    for line in read_lines(log_file):
        gs = parse_pattern(p, line)
        if gs is None:
            continue
        result.append(gs)
    return result

def parse_yolo_log_st(log_file):
    p = 'region_loss_layer\.cpp:1138] ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*)'
    ss = parse_nums(p, log_file)
    p = 'region_loss_layer\.cpp:1140] ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*)'
    tt = parse_nums(p, log_file)
    return ss, tt

def parse_yolo_log_acc(log_file):
    p = 'Region Avg IOU: ([0-9, .]*), Class: ([0-9, .]*), '
    p = p + 'Obj: ([0-9, .]*), No Obj: ([0-9, .]*), Avg Recall: ([0-9, .]*),  count: ([0-9]*)'
    all_ious = []
    all_probs = []
    all_obj = []
    all_noobj = []
    all_recall = []
    all_count = []
    for line in read_lines(log_file):
        gs = parse_pattern(p, line)
        if gs is None:
            continue
        all_ious.append(gs[0])
        all_probs.append(gs[1])
        all_obj.append(gs[2])
        all_noobj.append(gs[3])
        all_recall.append(gs[4])
        all_count.append(gs[5])
    return all_ious, all_probs, all_obj, all_noobj, all_recall, all_count


def read_lines(file_name, **kwargs):
    with open(file_name, 'r', **kwargs) as fp:
        for line in fp:
            yield line

#@deprecated('use tsv_io.read_to_buffer')
def read_to_buffer(file_name):
    with open(file_name, 'rb') as fp:
        all_line = fp.read()
    return all_line

class Model(object):
    def __init__(self, test_proto_file, train_proto_file, model_param, mean_value, scale, model_iter):
        self.test_proto_file = test_proto_file
        self.model_param = model_param
        self.mean_value = mean_value
        self.model_iter = model_iter
        self.scale = scale
        self.train_proto_file = train_proto_file

def adjust_tree_prediction_threshold(n, tree_th):
    found = False
    for l in n.layer:
        if l.type == 'SoftmaxTreePrediction':
            found = True
            l.softmaxtreeprediction_param.threshold = tree_th
    assert found

def remove_nms(n):
    for l in n.layer:
        if l.type == 'RegionOutput':
            l.region_output_param.nms = -1
        if l.type == 'RegionPrediction':
            l.region_prediction_param.nms = -1

def update_conv_channels(net, factor, skip):
    c = 0
    s = 0
    for l in net.layer:
        if l.type == 'Convolution':
            if s < skip:
                s = s + 1
                continue
            o = l.convolution_param.num_output
            l.convolution_param.num_output = int(o * factor)
            c = c + 1
    logging.info('updated {} layers for channel factor'.format(c))

def get_channel(net, blob_name):
    for l in net.layer:
        if l.type == 'Convolution':
            assert len(l.top) == 1
            if l.top[0] == blob_name:
                return l.convolution_param.num_output
    assert False, 'not found'

def fix_net_parameters(net, last_fixed_param):
    found = False
    no_param_layers = set(['TsvBoxData', 'ReLU', 'Pooling', 'Reshape',
            'EuclideanLoss', 'Sigmoid'])
    unknown_layers = []
    for l in net.layer:
        if l.type == 'Convolution' or l.type == 'Scale':
            if l.type == 'Convolution':
                assert len(l.param) >= 1
            else:
                if len(l.param) == 0:
                    p = l.param.add()
                    p.lr_mult = 0
                    p.decay_mult = 0
                    if l.scale_param.bias_term:
                        p = l.param.add()
                        p.lr_mult = 0
                        p.decay_mult = 0
            for p in l.param:
                p.lr_mult = 0
                p.decay_mult = 0
        elif l.type == 'BatchNorm':
            l.batch_norm_param.use_global_stats = True
        else:
            if l.type not in no_param_layers:
                unknown_layers.append(l.type)
        if l.name == last_fixed_param:
            for b in l.bottom:
                l.propagate_down.append(False)
            found = True
            break
    assert len(unknown_layers) == 0, ', '.join(unknown_layers)
    assert found

def set_no_bias(net, layer_name):
    for l in net.layer:
        if l.name == layer_name:
            assert l.type == 'Convolution'
            l.convolution_param.bias_term = False
            if len(l.param) == 2:
                del l.param[1]
            else:
                assert len(l.param) == 0
            return
    assert False

def add_yolo_angular_loss_regularizer(net, **kwargs):
    for l in net.layer:
        if l.name == 'angular_loss':
            logging.info('angular loss exists')
            return
    conf_layer = None
    for l in net.layer:
        if l.name == 'conf':
            conf_layer = l
            assert 'conf' in l.top
    found_t_label = False
    for l in net.layer:
        if 't_label' in l.top:
            found_t_label = True
            break
    assert conf_layer and found_t_label

    conf_layer.param[0].name = 'conf_w'
    CA = conf_layer.convolution_param.num_output
    assert len(conf_layer.bottom) == 1
    num_feature = get_channel(net, conf_layer.bottom[0])

    param_layer = net.layer.add()
    param_layer.name = 'param_conf_w'
    param_layer.type = 'Parameter'
    param_layer.parameter_param.shape.dim.append(CA)
    param_layer.parameter_param.shape.dim.append(num_feature)
    param_layer.parameter_param.shape.dim.append(1)
    param_layer.parameter_param.shape.dim.append(1)
    param_layer.top.append('conf_w')
    p = param_layer.param.add()
    p.name = 'conf_w'

    layer = net.layer.add()
    layer.name = 'angular_loss'
    layer.type = 'Python'
    layer.bottom.append(conf_layer.bottom[0])
    layer.bottom.append('t_label')
    layer.bottom.append('conf_w')
    layer.python_param.module = 'kcenter_exp'
    layer.python_param.layer = 'YoloAngularLossLayer'
    layer.propagate_down.append(True)
    layer.propagate_down.append(False)
    layer.propagate_down.append(False)
    layer.top.append('angular_loss')
    weight = kwargs.get('yolo_angular_loss_weight', 1)
    layer.loss_weight.append(weight)

def add_yolo_low_shot_regularizer(net, low_shot_label_idx):
    assert net.layer[-1].type == 'RegionLoss'
    assert net.layer[-2].type == 'Convolution'
    assert net.layer[-1].bottom[0] == net.layer[-2].top[0]
    assert net.layer[-2].convolution_param.kernel_size[0] == 1
    assert net.layer[-2].convolution_param.kernel_h == 0
    assert net.layer[-2].convolution_param.kernel_w == 0

    num_classes = net.layer[-1].region_loss_param.classes
    num_anchor = len(net.layer[-1].region_loss_param.biases) / 2

    param_dim1 = net.layer[-2].convolution_param.num_output
    param_dim2 = get_channel(net, net.layer[-2].bottom[0])

    # add the parameter name into the convolutional layer
    last_conv_param_name = 'last_conv_param_low_shot'
    net.layer[-2].param[0].name = last_conv_param_name

    # add the parameter layer to expose the parameter
    param_layer = net.layer.add()
    param_layer.type = 'Parameter'
    param_layer.name = 'param_last_conv'
    param_layer.top.append(last_conv_param_name)
    p = param_layer.param.add()
    p.name = last_conv_param_name
    p.lr_mult = 1
    p.decay_mult = 1
    param_layer.parameter_param.shape.dim.append(param_dim1)
    param_layer.parameter_param.shape.dim.append(param_dim2)

    # add the regularizer layer
    reg_layer = net.layer.add()
    reg_layer.type = 'Python'
    reg_layer.name = 'equal_norm'
    reg_layer.bottom.append(last_conv_param_name)
    reg_layer.top.append('equal_norm')
    reg_layer.loss_weight.append(1)
    reg_layer.python_param.module = 'equal_norm_loss'
    reg_layer.python_param.layer = 'YoloAlignNormToBaseLossLayer'
    reg_param = {'num_classes': num_classes,
            'low_shot_label_idx': low_shot_label_idx,
            'num_anchor': num_anchor}
    reg_layer.python_param.param_str = json.dumps(reg_param)

def update_kernel_active(net, kernel_active, kernel_active_skip):
    assert False, 'use update_kernel_active2'
    c = 0
    skipped = 0
    logging.info('{}-{}'.format(kernel_active, kernel_active_skip));
    for l in net.layer:
        if l.type == 'Convolution':
            if skipped < kernel_active_skip:
                skipped = skipped + 1
                logging.info('skiping to update active kernel')
                continue
            l.convolution_param.kernel_active = kernel_active
            c = c + 1

    logging.info('update {} layers'.format(c))


def plot_to_file(xs, ys, file_name, **kwargs):
    fig = plt.figure()
    semilogy = kwargs.get('semilogy')
    if all(isinstance(x, str) for x in xs):
        xs2 = range(len(xs))
        #plt.xticks(xs2, xs, rotation=15, ha='right')
        plt.xticks(xs2, xs, rotation='vertical')
        xs = xs2
    if type(ys) is dict:
        for key in ys:
            if semilogy:
                plt.semilogy(xs, ys[key], '-o')
            else:
                plt.plot(xs, ys[key], '-o')
    else:
        if semilogy:
            plt.semilogy(xs, ys, '-o')
        else:
            plt.plot(xs, ys, '-o')
    plt.grid()
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    ensure_directory(op.dirname(file_name))
    plt.tight_layout()
    # explicitly remove the file because philly does not support overwrite
    if op.isfile(file_name):
        try:
            os.remove(file_name)
        except:
            logging.info('{} exists but could not be deleted'.format(
                file_name))
    fig.savefig(file_name)
    plt.close(fig)

def parse_training_time(log_file):
    log = read_to_buffer(log_file)
    all_time_cost = []
    all_iters = []
    for line in log.split('\n'):
        m = re.match('.*Iteration.*iter\/s, ([0-9\.]*)s\/([0-9]*) iters.*', line)
        if m:
            r = m.groups()
            time_cost = float(r[0])
            iters = float(r[1])
            all_iters.append(iters)
            all_time_cost.append(time_cost)
    return all_iters, all_time_cost

def encode_expid(prefix, *args):
    parts = [prefix]
    for (t, a) in args:
        p = ''
        if a != None:
            if type(a) == str:
                a = a.replace(':', '_')
            if t != None and len(t) > 0:
                p = p + '_{}'.format(t)
            p = p + '_{}'.format(a)
        parts.append(p)
    return ''.join(parts)

def unicode_representer(dumper, uni):
    node = yaml.ScalarNode(tag=u'tag:yaml.org,2002:str', value=uni)
    return node

def dump_to_yaml_bytes(context):
    return yaml.dump(context, default_flow_style=False,
            encoding='utf-8', allow_unicode=True)

def dump_to_yaml_str(context):
    return dump_to_yaml_bytes(context).decode()

def write_to_yaml_file(context, file_name):
    ensure_directory(op.dirname(file_name))
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, default_flow_style=False,
                encoding='utf-8', allow_unicode=True)

def load_from_yaml_str(s):
    return yaml.load(s, Loader=yaml.UnsafeLoader)

def load_from_yaml_file(file_name):
    # do not use QDFile.open as QDFile.open depends on this function
    with exclusive_open_to_read(file_name, 'r') as fp:
    #with open(file_name, 'r') as fp:
        data = load_from_yaml_str(fp)
    while isinstance(data, dict) and '_base_' in data:
        b = op.join(op.dirname(file_name), data['_base_'])
        result = load_from_yaml_file(b)
        assert isinstance(result, dict)
        del data['_base_']
        all_key = get_all_path(data, with_list=False)
        for k in all_key:
            v = dict_get_path_value(data, k)
            dict_update_path_value(result, k, v)
        data = result
    return data

#@deprecated('use qd.tsv_io.write_to_file')
def write_to_file(contxt, file_name, append=False):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    if type(contxt) is str:
        contxt = contxt.encode()
    flag = 'wb'
    if append:
        flag = 'ab'
    with open(file_name, flag) as fp:
        fp.write(contxt)

#@deprecated('use qd.tsv_io.load_list_file')
def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

class LoopProcess(Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        '''
        same signiture with Process.__init__
        The process will keep running the function of target and will wait for
        several seconds in between. This is useful to run some monitoring job
        or regular job
        '''
        super(LoopProcess, self).__init__(group, target, name, args, kwargs)
        self._exit = Event()

    def run(self):
        sleep_time = 5
        while not self._exit.is_set():
            if self._target:
                self._target(*self._args, **self._kwargs)
            time.sleep(sleep_time)

    def init_shutdown(self):
        self._exit.set()

class PyTee(object):
    def __init__(self, logstream, stream_name):
        valid_streams = ['stderr','stdout'];
        if  stream_name not in valid_streams:
            raise IOError("valid stream names are %s" % ', '.join(valid_streams))
        self.logstream =  logstream
        self.stream_name = stream_name;
    def __del__(self):
        pass;
    def write(self, data):  #tee stdout
        self.logstream.write(data);
        self.fstream.write(data);
        self.logstream.flush();
        self.fstream.flush();

    def flush(self):
        self.logstream.flush();
        self.fstream.flush();

    def __enter__(self):
        if self.stream_name=='stdout' :
            self.fstream   =  sys.stdout
            sys.stdout = self;
        else:
            self.fstream   =  sys.stderr
            sys.stderr = self;
        self.fstream.flush();
    def __exit__(self, _type, _value, _traceback):
        if self.stream_name=='stdout' :
            sys.stdout = self.fstream;
        else:
            sys.stderr = self.fstream;

def parse_basemodel_with_depth(net):
    '''
    darknet19->darknet19
    darknet19_abc->darknet19
    '''
    if '_' not in net:
        return net
    else:
        i = net.index('_')
        return net[: i]

def worth_create(base, derived, buf_second=0):
    if not op.isfile(base) and \
            not op.islink(base) and \
            not op.isdir(base):
        return False
    if os.path.isfile(derived) and \
            os.path.getmtime(derived) > os.path.getmtime(base) - buf_second:
        return False
    else:
        return True

def basename_no_ext(file_name):
    return op.splitext(op.basename(file_name))[0]

def default_data_path(dataset):
    '''
    use TSVDataset instead
    '''
    proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)));
    result = {}
    data_root = os.path.join(proj_root, 'data', dataset)
    result['data_root'] = data_root
    result['source'] = os.path.join(data_root, 'train.tsv')
    result['trainval'] = op.join(data_root, 'trainval.tsv')
    result['test_source'] = os.path.join(data_root, 'test.tsv')
    result['labelmap'] = os.path.join(data_root, 'labelmap.txt')
    result['source_idx'] = os.path.join(data_root, 'train.lineidx')
    result['test_source_idx'] = os.path.join(data_root, 'test.lineidx')
    return result

class FileProgressingbar:
    fileobj = None
    pbar = None
    def __init__(self, fileobj, keyword='Test'):
        fileobj.seek(0,os.SEEK_END)
        flen = fileobj.tell()
        fileobj.seek(0,os.SEEK_SET)
        self.fileobj = fileobj
        widgets = ['{}: '.format(keyword), progressbar.AnimatedMarker(),' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=flen).start()
    def update(self):
        self.pbar.update(self.fileobj.tell())

def is_pil_image(image):
    from PIL import Image
    return isinstance(image, Image.Image)

def encoded_from_img(im, quality=None, save_type=None):
    if save_type is None:
        save_type = 'jpg'
    assert save_type in ['jpg', 'png']
    if not is_pil_image(im):
        if quality:
            x = cv2.imencode('.{}'.format(save_type), im,
                    (cv2.IMWRITE_JPEG_QUALITY, quality))[1]
        else:
            x = cv2.imencode('.{}'.format(save_type), im)[1]
    else:
        if save_type in ['jpg', None]:
            save_type = 'JPEG'
        import io
        x = io.BytesIO()
        im.save(x, format=save_type)
        x = x.getvalue()
    return base64.b64encode(x)

def encode_image(im, quality=None):
    if quality:
        x = cv2.imencode('.jpg', im, (cv2.IMWRITE_JPEG_QUALITY, quality))[1]
    else:
        x = cv2.imencode('.jpg', im)[1]
    return x.tobytes()

def is_valid_image(im):
    return im is not None and all(x != 0 for x in im.shape)

def pilimg_from_base64(imagestring):
    try:
        import io
        jpgbytestring = base64.b64decode(imagestring)
        image = Image.open(io.BytesIO(jpgbytestring))
        image = image.convert('RGB')
        return image
    except:
        return None

def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        return r
    except:
        return None;

def img_from_bytes(jpgbytestring):
    try:
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        return r
    except:
        return None

def img_from_base64_ignore_rotation(str_im):
    jpgbytestring = base64.b64decode(str_im)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_IGNORE_ORIENTATION);
    return im

def encode_decode_im(im, quality):
    with BytesIO() as output:
        im.save(output, 'JPEG', quality=quality)
        im = Image.open(output).convert('RGB')
    return im

def int_rect(rect, enlarge_factor=1.0, im_h=None, im_w=None):
    assert(len(rect) == 4)
    left, top, right, bot = rect
    rw = right - left
    rh = bot - top

    new_x = int(left + (1.0 - enlarge_factor) * rw / 2.0)
    new_y = int(top + (1.0 - enlarge_factor) * rh / 2.0)
    new_w = int(math.ceil(enlarge_factor * rw))
    new_h = int(math.ceil(enlarge_factor * rh))
    if im_h and im_w:
        new_x = np.clip(new_x, 0, im_w)
        new_y = np.clip(new_y, 0, im_h)
        new_w = np.clip(new_w, 0, im_w - new_x)
        new_h = np.clip(new_h, 0, im_h - new_y)

    return list(map(int, [new_x, new_y, new_x + new_w, new_y + new_h]))

def is_valid_rect(rect):
    return len(rect) == 4 and rect[0] < rect[2] and rect[1] < rect[3]

def pass_key_value_if_has(d_from, from_key, d_to, to_key):
    if from_key in d_from:
        d_to[to_key] = d_from[from_key]

def dict_update_nested_dict(a, b, overwrite=True):
    for k, v in b.items():
        if k not in a:
            dict_update_path_value(a, k, v)
        else:
            if isinstance(dict_get_path_value(a, k), dict) and isinstance(v, dict):
                dict_update_nested_dict(dict_get_path_value(a, k), v, overwrite)
            else:
                if overwrite:
                    dict_update_path_value(a, k, v)

def dict_ensure_path_key_converted(a):
    for k in list(a.keys()):
        v = a[k]
        if '$' in k:
            parts = k.split('$')
            x = {}
            x_curr = x
            for p in parts[:-1]:
                x_curr[p] = {}
                x_curr = x_curr[p]
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)
            x_curr[parts[-1]] = v
            dict_update_nested_dict(a, x)
            del a[k]
        else:
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)

def query_values_by_path_suffix(job_in_scheduler, suffix, default=None):
    found = []
    for p in get_all_path(job_in_scheduler, leaf_only=False):
        if p.endswith(suffix):
            v = dict_get_path_value(job_in_scheduler, p)
            found.append(v)
    return found

# this is to get value, rather than values, which is misleading
#@deprecated('use query_values_by_path_suffix')
def query_path_by_suffix(job_in_scheduler, suffix, default=None):
    found = []
    for p in get_all_path(job_in_scheduler, leaf_only=False):
        if p.endswith(suffix):
            v = dict_get_path_value(job_in_scheduler, p)
            found.append(v)
    if len(found) == 1:
        return found[0]
    elif len(found) > 1:
        if all(f == found[0] for f in found[1:]):
            return found[0]
        else:
            raise ValueError
    else:
        return default

def get_all_path(d, with_type=False, leaf_only=True, with_list=True):
    assert not with_type, 'will not support'
    all_path = []

    if isinstance(d, dict):
        for k, v in d.items():
            all_sub_path = get_all_path(
                v, with_type, leaf_only=leaf_only, with_list=with_list)
            all_path.extend([k + '$' + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append(k)
    elif (isinstance(d, tuple) or isinstance(d, list)) and with_list:
        for i, _v in enumerate(d):
            all_sub_path = get_all_path(
                _v, with_type,
                leaf_only=leaf_only,
                with_list=with_list,
            )
            all_path.extend(['{}$'.format(i) + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append('{}'.format(i))
    return all_path

def dict_get_all_path(d, with_type=False, leaf_only=True):
    all_path = []
    for k, v in viewitems(d):
        if with_type:
            if type(k) is str:
                k = 's' + k
            elif type(k) is int:
                k = 'i' + str(k)
            else:
                raise NotImplementedError
        if isinstance(v, dict):
            all_sub_path = dict_get_all_path(
                v, with_type, leaf_only=leaf_only)
            all_path.extend([k + '$' + p for p in all_sub_path])
            if not leaf_only:
                all_path.append(k)
        elif isinstance(v, tuple) or isinstance(v, list):
            for i, _v in enumerate(v):
                prefix = '' if not with_type else 'i'
                if isinstance(_v, (dict, list)):
                    all_sub_path = dict_get_all_path(
                        _v, with_type,
                        leaf_only=leaf_only)
                    all_path.extend([k + '${}{}$'.format(prefix, i) + p for p in all_sub_path])
                else:
                    all_path.append(k + '${}{}'.format(prefix, i))
            if not leaf_only:
                all_path.append(k)
        else:
            all_path.append(k)
    return all_path

def dict_parse_key(k, with_type):
    if with_type:
        if k[0] == 'i':
            return int(k[1:])
        else:
            return k[1:]
    return k

def dict_has_path(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, dict) and k in cur_dict:
                cur_dict = cur_dict[k]
                ps = ps[1:]
            elif isinstance(cur_dict, list):
                try:
                    k = int(k)
                except:
                    return False
                cur_dict = cur_dict[k]
                ps = ps[1:]
            else:
                return False
        else:
            return True


def dict_set_path_if_not_exist(param, k, v):
    if not dict_has_path(param, k):
        dict_update_path_value(param, k, v)
        return True
    else:
        return False

def dict_update_path_value(d, p, v):
    ps = p.split('$')
    while True:
        if len(ps) == 1:
            d[ps[0]] = v
            break
        else:
            if ps[0] not in d:
                d[ps[0]] = {}
            d = d[ps[0]]
            ps = ps[1:]

def dict_remove_path(d, p):
    ps = p.split('$')
    assert len(ps) > 0
    cur_dict = d
    need_delete = ()
    while True:
        if len(ps) == 1:
            if len(need_delete) > 0 and len(cur_dict) == 1:
                del need_delete[0][need_delete[1]]
            else:
                del cur_dict[ps[0]]
            return
        else:
            if len(cur_dict) == 1:
                if len(need_delete) == 0:
                    need_delete = (cur_dict, ps[0])
            else:
                need_delete = (cur_dict, ps[0])
            cur_dict = cur_dict[ps[0]]
            ps = ps[1:]

def dict_get_path_value(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, (tuple, list)):
                cur_dict = cur_dict[int(k)]
            else:
                cur_dict = cur_dict[k]
            ps = ps[1:]
        else:
            return cur_dict

def get_file_size(f):
    return os.stat(f).st_size

def convert_to_yaml_friendly(result):
    if type(result) is dict:
        for key, value in result.items():
            if isinstance(value, dict):
                result[key] = convert_to_yaml_friendly(value)
            elif isinstance(value, np.floating):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                raise NotImplementedError()
            elif type(value) in [int, str, float, bool]:
                continue
            else:
                raise NotImplementedError()
    else:
        raise NotImplementedError()
    return result

def natural_key(text):
    import re
    result = []
    for c in re.split(r'([0-9]+(?:[.][0-9]*)?)', text):
        try:
            result.append(float(c))
        except:
            continue
    return result

def natural_sort(strs, key=None):
    if key is None:
        strs.sort(key=natural_key)
    else:
        strs.sort(key=lambda s: natural_key(key(s)))

def get_pca(x, com):
    x -= np.mean(x, axis = 0)
    cov = np.cov(x, rowvar=False)
    from scipy import linalg as LA
    evals , evecs = LA.eigh(cov)
    total_val = np.sum(evals)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    component_val = np.sum(evals[:com])
    logging.info('kept: {}/{}={}'.format(component_val,
            total_val, component_val / total_val))
    a = np.dot(x, evecs[:, :com])
    return a

def plot_distribution(x, y, color=None, fname=None):
    import seaborn as sns
    x = sns.jointplot(x, y, kind='kde',
            color=color)
    if fname:
        x.savefig(fname)
        plt.close()
    else:
        plt.show()

g_key_to_cache = {}
def run_if_not_memory_cached(func, *args, **kwargs):
    force = False
    if '__force' in kwargs:
        if kwargs['__force']:
            force = True
        del kwargs['__force']
    if '__key' in kwargs:
        key = kwargs.pop('__key')
    else:
        import pickle as pkl
        key = hash_sha1(pkl.dumps(OrderedDict({'arg': pformat(args), 'kwargs':
            pformat(kwargs), 'func_name': func.__name__})))
    global g_key_to_cache

    if key in g_key_to_cache and not force:
        return g_key_to_cache[key]
    else:
        result = func(*args, **kwargs)
        g_key_to_cache[key] = result
        return result

def run_if_not_cached(func, *args, **kwargs):
    force = False
    if '__force' in kwargs:
        if kwargs['__force']:
            force = True
        del kwargs['__force']

    import pickle as pkl
    key = hash_sha1(pkl.dumps(OrderedDict({'arg': pformat(args), 'kwargs':
        pformat(kwargs), 'func_name': func.__name__})))
    cache_folder = op.expanduser('/tmp/run_if_not_cached/')
    cache_file = op.join(cache_folder, key)

    if op.isfile(cache_file) and not force:
        return pkl.loads(read_to_buffer(cache_file))
    else:
        result = func(*args, **kwargs)
        write_to_file(pkl.dumps(result), cache_file)
        return result

def convert_to_command_line(param, script):
    logging.info(pformat(param))
    x = copy.deepcopy(param)
    result = "python {} -bp {}".format(
            script,
            base64.b64encode(dump_to_yaml_bytes(x)).decode())
    return result

def print_table(a_to_bs, all_key=None, latex=False, **kwargs):
    if len(a_to_bs) == 0:
        return
    if not latex:
        all_line = get_table_print_lines(a_to_bs, all_key)
        logging.info('\n{}'.format('\n'.join(all_line)))
    else:
        from qd.latex_writer import print_simple_latex_table
        if all_key is None:
            all_key = list(set(a for a_to_b in a_to_bs for a in a_to_b))
            all_key = sorted(all_key)
        x = print_simple_latex_table(a_to_bs,
                all_key, **kwargs)
        logging.info('\n{}'.format(x))
        return x

def get_table_print_lines(a_to_bs, all_key):
    if len(a_to_bs) == 0:
        logging.info('no rows')
        return []
    if not all_key:
        all_key = []
        for a_to_b in a_to_bs:
            all_key.extend(a_to_b.keys())
        all_key = sorted(list(set(all_key)))
    all_width = [max([len(str(a_to_b.get(k, ''))) for a_to_b in a_to_bs] +
        [len(k)]) for k in all_key]
    row_format = ' '.join(['{{:{}}}'.format(w) for w in all_width])

    all_line = []
    line = row_format.format(*all_key)
    all_line.append(line.strip())
    for a_to_b in a_to_bs:
        line = row_format.format(*[str(a_to_b.get(k, '')) for k in all_key])
        all_line.append(line)
    return all_line

def is_hvd_initialized():
    try:
        import horovod.torch as hvd
        hvd.size()
        return True
    except ImportError:
        return False
    except ValueError:
        return False

def get_user_name():
    import getpass
    return getpass.getuser()

def decode_general_cmd(extraParam):
    re_result = re.match('.*python (?:scripts|src)/.*\.py -bp (.*)', extraParam)
    if re_result and len(re_result.groups()) == 1:
        ps = load_from_yaml_str(base64.b64decode(re_result.groups()[0]))
        return ps

def print_job_infos(all_job_info):
    all_key = [
        'cluster',
        'status',
        'appID-s',
        'result',
        'elapsedTime',
        'elapsedFinished',
        'mem_used',
        'gpu_util',
        'speed',
        'left']
    keys = ['data', 'net', 'expid']
    meta_keys = ['num_gpu']
    all_key.extend(keys)
    all_key.extend(meta_keys)

    # find the keys whose values are the same
    def all_equal(x):
        assert len(x) > 0
        return all(y == x[0] for y in x[1:])

    if len(all_job_info) > 1:
        equal_keys = [k for k in all_key if all_equal([j.get(k) for j in all_job_info])]
        if len(equal_keys) > 0:
            logging.info('equal key values for all jobs')
            print_table(all_job_info[0:1], all_key=equal_keys)
        all_key = [k for k in all_key if not all_equal([j.get(k) for j in all_job_info])]

    print_table(all_job_info, all_key=all_key)

def parse_eta_in_hours(left):
    pattern = '(?:([0-9]*) day[s]?, )?([0-9]*):([0-9]*):([0-9]*)'
    result = re.match(pattern, left)
    if result:
        gs = result.groups()
        gs = [float(g) if g else 0 for g in gs]
        assert int(gs[0]) == gs[0]
        days = int(gs[0])
        hours = gs[1] + gs[2] / 60. + gs[3] / 3600
        return days, hours
    return -1, -1

def attach_philly_maskrcnn_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        # Philly, maskrcnn-benchmark log
        pattern = '([0-9\. :-]): .*: eta: (.*) iter: [0-9]*[ ]*speed: ([0-9\.]*).*'

        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left, speed = result.groups()
            job_info['speed'] = speed
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            delay = (now - log_time).total_seconds()
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h({:.1f}s)'.format(d, h, delay)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def calc_eta(days, hours):
    from datetime import timedelta
    x = datetime.now() + timedelta(days=days, hours=hours + 1)
    return '{}/{}-{}'.format(x.month, x.day, x.hour)

def attach_mmask_caption_itp_multi_line_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        pattern = r'.*(202[0-9:\ \-\.]+)\,.* (?:train|mmask_caption|mmask_pretrain)\.py\:.*(?:train)\(\):.*eta[ ]*:(.*) iter: [0-9]*'
            #2020-10-13 20:15:24 [21ddde126d88bf2a12f76f300ddaac02-ps-0, 10.43.224.4] [1,0]<stdout>:2020-10-13 20:15:23,329.329 mmask_caption.py:186      train(): eta : 3:37:16  iter: 2460  max mem : 4541
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left = result.groups()
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left.strip())
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def attach_mmask_aml_multi_line_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        pattern = r'(202[0-9:\ \-\.]+)\,.* (?:mmask_caption|mmask_pretrain)\.py\:.*(?:train)\(\):.*eta[ ]*:(.*) iter: [0-9]*'
            #2020-10-08 03:00:47,292.292 mmask_caption.py:186      train(): eta : 3:40:55  iter: 1900  max mem : 4541
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left = result.groups()
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left.strip())
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def attach_itp_mmask_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        pattern = r'.*<stdout>:(.*),.* (?:trainer|mmask_pretrain)\.py.*(?:do_train|do_train_dict|train)\(\): eta: (.*) iter: [0-9]*.*'
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left = result.groups()
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def attach_itp_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        pattern = r'.*<stdout>:(.*),.*(?:base_trainer|trainer|mmask_pretrain|image_text_retrieval|vqa)\.py.*(?:do_train|_logistics|do_train_dict|train|old_train)\(\): eta:[ ]*(.*) iter: [0-9]*[ ]*speed: ([0-9\.]*).*'
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left, speed = result.groups()
            job_info['speed'] = speed
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def attach_aml_maskrcnn_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        pattern = r'(.*),.* (?:mmask_pretrain|trainer|vqa)\.py.*(?:old_train|train|do_train|do_train_dict)\(\): eta: (.*) iter: [0-9]*[ ]*speed: ([0-9\.]*).*'

        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left, speed = result.groups()
            job_info['speed'] = speed
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def attach_aml_detectron2_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        pattern = r'(.*),.* events\.py.*eta: (.*) iter: [0-9]*.*'
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left = result.groups()
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def attach_philly_caffe_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        # philly, caffe log
        pattern = '.*solver\.cpp:[0-9]*] Iteration [0-9]* \(.* iter\/s, ([0-9\.]*s\/100) iters, left: ([0-9\.]*h)\), loss = [0-9\.]*'
        result = re.match(pattern, log)
        if result and result.groups():
            job_info['speed'], job_info['left'] = result.groups()
            return True
    return False

def attach_gpu_utility_from_log(all_log, job_info):
    for log in reversed(all_log):
        # philly, caffe log aml_server or philly_server log
        pattern = '.*_server.py:.*monitor.*\[(.*)\]'
        result = re.match(pattern, log)
        if result and result.groups():
            try:
                all_info = json.loads('[{}]'.format(result.groups()[0].replace('\'', '\"')))
                min_gpu_mem = min([i['mem_used'] for i in all_info])
                max_gpu_mem = max([i['mem_used'] for i in all_info])
                min_gpu_util = min([i['gpu_util'] for i in all_info])
                max_gpu_util = max([i['gpu_util'] for i in all_info])
                # GB
                job_info['mem_used'] = '{}-{}'.format(round(min_gpu_mem/1024, 1),
                        round(max_gpu_mem/1024., 1))
                job_info['gpu_util'] = '{}-{}'.format(min_gpu_util, max_gpu_util)
                return True
            except:
                return False
    return False

def attach_log_parsing_result(job_info):
    # run unit test if modified
    logs = job_info.get('latest_log')
    if logs is None:
        return
    all_log = logs.split('\n')
    del job_info['latest_log']
    attach_gpu_utility_from_log(all_log, job_info)
    if attach_itp_log_if_is(all_log, job_info):
        return
    if attach_philly_maskrcnn_log_if_is(all_log, job_info):
        return
    if attach_aml_maskrcnn_log_if_is(all_log, job_info):
        return
    if attach_philly_caffe_log_if_is(all_log, job_info):
        return
    if attach_aml_detectron2_log_if_is(all_log, job_info):
        return
    if attach_itp_mmask_log_if_is(all_log, job_info):
        return
    if attach_mmask_aml_multi_line_log_if_is(all_log, job_info):
        return
    if attach_mmask_caption_itp_multi_line_log_if_is(all_log, job_info):
        return
    # the following is designed to cover any examples
    if attach_any_log(all_log, job_info):
        return

def attach_any_log(all_log, job_info):
    # to check the correctness, run: py.test --ipdb src/qd/unittest/test_qd_common.py -k test_attach_any_log
    for log in reversed(all_log):
        pattern = r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}).*\.py.*\(\): eta:[ ]*(.*) iter: [0-9]*[ ]*speed: ([0-9\.]*).*'
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left, speed = result.groups()
            job_info['speed'] = speed
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            job_info['eta'] = calc_eta(d, h)
            return True
    return False

def print_offensive_folder(folder):
    all_folder = os.listdir(folder)
    name_to_size = {}
    for i, f in enumerate(qd_tqdm(all_folder)):
        sec = 60 * 10
        f = op.join(folder, f)
        size = run_if_not_cached(get_folder_size, f, sec)
        name_to_size[f] = size
        logging.info('{}: {}'.format(f, size))
    logging.info(', '.join([op.basename(n) for n, s in name_to_size.items() if
        s < 0]))

def get_folder_size(f, sec):
    cmd = ['du', '--max-depth=0', f]
    import subprocess
    from subprocess import check_output
    try:
        out = check_output(cmd, timeout=sec)
    except subprocess.TimeoutExpired:
        logging.info('{}'.format(f))
        return -1
    out = out.decode()
    size = [x.strip() for x in out.split('\t')][0]
    return int(size)

class make_namespace_by_dict(object):
    def __init__(self, d):
        for k in d:
            v = d[k]
            if type(v) is dict:
                self.__dict__[k] = make_namespace_by_dict(v)
            else:
                self.__dict__[k] = v
    def clone(self):
        c = copy.deepcopy(self.__dict__)
        return make_namespace_by_dict(c)

    def __repr__(self):
        return '{}'.format(pformat(self.__dict__))

@try_once
def try_get_cpu_info():
    command = 'cat /proc/cpuinfo'
    return os.popen(command).read().strip()

# ---------------------------------------------------- pytorch speed analysis
def create_speed_node(info):
    node = Tree()
    node.add_features(**info)
    return node

def speed_tree_insert(root, node):
    while True:
        need_merge_nodes = [c for c in root.children
                if is_child_parent(c.name, node.name)]
        if len(need_merge_nodes) > 0:
            for x in need_merge_nodes:
                x.detach()
            for x in need_merge_nodes:
                node.add_child(x)
            root.add_child(node)
            return
        go_deeper_nodes = [c for c in root.children if
                is_child_parent(node.name, c.name)]
        if len(go_deeper_nodes) == 0:
            root.add_child(node)
            return
        else:
            assert len(go_deeper_nodes) == 1
            root = go_deeper_nodes[0]

def is_child_parent(c, p):
    if p == '':
        return True
    return c.startswith(p + '.')

def speed_trees_insert(roots, info):
    node = create_speed_node(info)
    # we assume the name are not equal
    need_merge_nodes = [r for r in roots
            if is_child_parent(r.name, info['name'])]
    if len(need_merge_nodes) > 0:
        for x in need_merge_nodes:
            node.add_child(x)
            roots.remove(x)
        roots.append(node)
        return
    need_insert_roots = [r for r in roots
            if is_child_parent(info['name'], r.name)]
    if len(need_insert_roots) == 0:
        roots.append(node)
    elif len(need_insert_roots) == 1:
        speed_tree_insert(need_insert_roots[0], node)
    else:
        raise Exception()

def build_speed_tree(component_speeds):
    roots = []
    for c in component_speeds:
        speed_trees_insert(roots, c)
    return roots

def get_vis_str(component_speeds):
    roots = build_speed_tree(component_speeds)
    if len(roots) == 0:
        return ''
    assert len(roots) == 1, roots
    root = roots[0]
    for n in root.iter_search_nodes():
        n.global_avg_in_ms = round(1000. * n.global_avg, 1)
    for n in root.iter_search_nodes():
        s = sum([c.global_avg for c in n.children])
        n.unique_in_ms = round(1000. * (n.global_avg - s), 1)
    return root.get_ascii(attributes=
        ['name', 'global_avg_in_ms', 'unique_in_ms', 'count'])

def create_vis_net_file(speed_yaml, vis_txt):
    info = load_from_yaml_file(speed_yaml)
    if type(info) is list:
        info = info[0]
    assert type(info) is dict
    component_speeds = info['meters']
    write_to_file(get_vis_str(component_speeds), vis_txt)

# ---------------------------------------------------------------------

def dict_add(d, k, v):
    if k not in d:
        d[k] = v
    else:
        d[k] += v

def calc_mean(x):
    return sum(x) / len(x)

def compare_gmap_evals(all_eval_file,
        label_to_testcount=None,
        output_prefix='out'):
    result = ['\n']
    all_result = [load_from_yaml_file(f) for f in all_eval_file]
    all_cat2map = [result['class_ap'] for result in all_result]

    cats = list(all_cat2map[0].keys())
    gains = [all_cat2map[1][c] - all_cat2map[0][c] for c in cats]

    all_info = [{'name': c, 'acc_gain': g} for c, g in zip(cats, gains)]
    all_info = sorted(all_info, key=lambda x: x['acc_gain'])

    all_map = [sum(cat2map.values()) / len(cat2map) for cat2map in all_cat2map]
    result.append('all map = {}'.format(', '.join(
        map(lambda x: str(round(x, 3)), all_map))))

    non_zero_cats = [cat for cat, ap in all_cat2map[0].items()
            if all_cat2map[1][cat] > 0 and  ap > 0]
    logging.info('#non zero cat = {}'.format(len(non_zero_cats)))
    for cat2map in all_cat2map:
        logging.info('non zero cat mAP = {}'.format(
            calc_mean([cat2map[c] for c in non_zero_cats])))

    if label_to_testcount is not None:
        all_valid_map = [calc_mean([ap for cat, ap in cat2map.items() if
            label_to_testcount.get(cat, 0) >
                50]) for cat2map in all_cat2map]
        result.append('all valid map = {}'.format(', '.join(
            map(lambda x: str(round(x, 3)), all_valid_map))))
        valid_cats = set([l for l, c in label_to_testcount.items() if c > 50])

    max_aps = [max([cat2map[c] for cat2map in all_cat2map]) for c in cats]
    max_map = sum(max_aps) / len(max_aps)
    result.append('max map = {:.3f}'.format(max_map))

    for info in all_info:
        for k in info:
            if type(info[k]) is float:
                info[k] = round(info[k], 2)
    result.extend(get_table_print_lines(all_info[:5] + all_info[-6:], ['name',
        'acc_gain',
        ]))
    if label_to_testcount is not None:
        result.append('valid cats only:')
        all_valid_info = [i for i in all_info if i['name'] in valid_cats]
        result.extend(get_table_print_lines(all_valid_info[:5] + all_valid_info[-6:],
            ['name', 'acc_gain',
            ]))

    all_acc_gain = [info['acc_gain'] for info in all_info]
    logging.info('\n'.join(result))

    plot_to_file(list(range(len(all_acc_gain))),
            all_acc_gain,
            output_prefix + '.png')

def merge_class_names_by_location_id(anno):
    if any('location_id' in a for a in anno):
        assert all('location_id' in a for a in anno)
        location_id_rect = [(a['location_id'], a) for a in anno]
        from .common import list_to_dict
        location_id_to_rects = list_to_dict(location_id_rect, 0)
        merged_anno = []
        for _, rects in location_id_to_rects.items():
            r = copy.deepcopy(rects[0])
            r['class'] = [r['class']]
            r['class'].extend((rects[i]['class'] for i in range(1,
                len(rects))))
            r['conf'] = [r.get('conf', 1)]
            r['conf'].extend((rects[i].get('conf', 1) for i in range(1,
                len(rects))))
            merged_anno.append(r)
        return merged_anno
    else:
        assert all('location_id' not in a for a in anno)
        for a in anno:
            a['class'] = [a['class']]
            a['conf'] = [a.get('conf', 1.)]
        return anno

def softnms_c(rects, threshold=0, method=2, **kwargs):
    from fast_rcnn.nms_wrapper import soft_nms
    nms_input = np.zeros((len(rects), 5), dtype=np.float32)
    for i, r in enumerate(rects):
        nms_input[i, 0:4] = r['rect']
        nms_input[i, -1] = r['conf']
    nms_out = soft_nms(nms_input, threshold=threshold,
            method=method, **kwargs)
    return [{'rect': list(map(float, x[:4])), 'conf': float(x[-1])} for x in nms_out]

def softnms(rects, th=0.5):
    rects = copy.deepcopy(rects)
    result = []
    while len(rects) > 0:
        max_idx = max(range(len(rects)), key=lambda i:
                rects[i]['conf'])
        max_det = rects[max_idx]
        result.append(max_det)
        rects.remove(max_det)
        for j in range(len(rects)):
            j_rect = rects[j]
            ij_iou = calculate_iou1(max_det['rect'], j_rect['rect'])
            rects[j]['conf'] *= math.exp(-ij_iou * ij_iou / th)
    return result

def acquireLock(lock_f='/tmp/lockfile.LOCK'):
    import fcntl
    ensure_directory(op.dirname(lock_f))
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor

@contextlib.contextmanager
def acquire_lock(lock_f):
    import fcntl
    ensure_directory(op.dirname(lock_f))
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    yield locked_file_descriptor
    locked_file_descriptor.close()

def releaseLock(locked_file_descriptor):
    locked_file_descriptor.close()

def inject_yolo_by_maskrcnn_log_to_board(fname, folder):
    keys = ['loss',
            'cls',
            'o_noobj',
            'o_obj',
            'wh',
            'xy',
            'time',
            'data',
            ]
    pattern = ''.join('.*{}: ([0-9\.]*) \(([0-9\.]*)\).*'.format(k)for k in keys)
    pattern = '.*iter: ([0-9]*) .*' + 'speed: ([0-9\.]*) images/sec' + pattern
    logging.info(pattern)
    all_loss = []
    result = parse_nums(pattern, fname)
    result_keys = ['iteration', 'speed']
    for k in keys:
        result_keys.append(k + '_medium')
        result_keys.append(k + '_mean')
    all_loss = [dict(zip(result_keys, r)) for r in result]

    logging.info(len(all_loss))
    from torch.utils.tensorboard import SummaryWriter
    wt = SummaryWriter(log_dir=folder)
    for loss_info in all_loss:
        for k in result_keys:
            if k == 'iteration':
                continue
            wt.add_scalar(tag=k, scalar_value=loss_info[k],
                    global_step=loss_info['iteration'])

def inject_maskrcnn_log_to_board(fname, folder, keys=None):
    if keys is None:
        keys = ['loss',
                'criterion_loss',
                #'loss_box_reg',
                #'loss_classifier',
                #'loss_objectness',
                #'loss_rpn_box_reg',
                'time',
                'data',
                ]
    pattern = ''.join('.*{}: ([0-9\.]*) \(([0-9\.]*)\).*'.format(k)for k in keys)
    pattern = '.*iter: ([0-9]*) .*' + 'speed: ([0-9\.]*) images/sec' + pattern
    logging.info(pattern)
    all_loss = []
    result = parse_nums(pattern, fname)
    result_keys = ['iteration', 'speed']
    for k in keys:
        result_keys.append(k + '_medium')
        result_keys.append(k + '_mean')
    all_loss = [dict(zip(result_keys, r)) for r in result]

    logging.info(len(all_loss))
    from torch.utils.tensorboard import SummaryWriter
    wt = SummaryWriter(log_dir=folder)
    for loss_info in all_loss:
        for k in result_keys:
            if k == 'iteration':
                continue
            wt.add_scalar(tag=k, scalar_value=loss_info[k],
                    global_step=loss_info['iteration'])

class DummyCfg(object):
    # provide a signature of clone(), used by maskrcnn checkpointer
    def clone(self):
        return

def save_frame_yaml(fn):
    assert not op.isfile(fn)
    assert fn.endswith('.yaml')
    info = get_frame_info(1)
    write_to_yaml_file(info, fn)

def get_frame_info(last=0):
    import inspect
    frame = inspect.currentframe()
    frames = inspect.getouterframes(frame)
    frame = frames[1 + last].frame
    args, _, _, vs = inspect.getargvalues(frame)
    info = {i: vs[i] for i in args}
    info['_func'] = frame.f_code.co_name
    info['_filepath'] = frame.f_code.co_filename
    return info

def print_frame_info():
    import inspect
    frame = inspect.currentframe()
    frames = inspect.getouterframes(frame)
    frame = frames[1].frame
    args, _, _, vs = inspect.getargvalues(frame)
    info = []
    info.append('func name = {}'.format(inspect.getframeinfo(frame)[2]))
    for i in args:
        try:
            info.append('{} = {}'.format(i, vs[i]))
        except:
            info.append('type({}) = {}'.format(i, type(vs[i])))
            continue
    logging.info('; '.join(info))

def merge_speed_info(speed_yamls, out_yaml):
    write_to_yaml_file([load_from_yaml_file(y) for y in speed_yamls
        if op.isfile(y)], out_yaml)

def merge_speed_vis(vis_files, out_file):
    from .common import ensure_copy_file
    if len(vis_files) > 0 and op.isfile(vis_files[0]):
        ensure_copy_file(vis_files[0], out_file)

def merge_dict_to_cfg(dict_param, cfg):
    """merge the key, value pair in dict_param into cfg

    :dict_param: TODO
    :cfg: TODO
    :returns: TODO

    """
    def trim_dict(d, c):
        """remove all the keys in the dictionary of d based on the existance of
        cfg
        """
        to_remove = [k for k in d if k not in c]
        for k in to_remove:
            del d[k]
        to_check = [(k, d[k]) for k in d if d[k] is dict]
        for k, t in to_check:
            trim_dict(t, getattr(c, k))
    trimed_param = copy.deepcopy(dict_param)
    trim_dict(trimed_param, cfg)
    from yacs.config import CfgNode
    cfg.merge_from_other_cfg(CfgNode(trimed_param))

def execute_func(info):
    # info = {'from': module; 'import': func_name, 'param': dict}
    from importlib import import_module
    modules = import_module(info['from'])
    if 'func' in info and 'param' in info:
        return getattr(getattr(modules, info['import']), info['func'])(**info['param'])
    elif ('param' not in info) and ('args' not in info):
        return getattr(modules, info['import'])()
    elif ('param' in info) and ('args' not in info):
        return getattr(modules, info['import'])(**info['param'])
    elif ('param' not in info) and ('args' in info):
        return getattr(modules, info['import'])(*info['args'])
    else:
        return getattr(modules, info['import'])(*info['args'], **info['param'])

def detect_error_codes(log_file):
    all_line = read_to_buffer(log_file).decode().split('\n')
    error_codes = []
    for _, line in enumerate(all_line):
        if "raise RuntimeError('NaN encountered!')" in line:
            error_codes.append('NaN')
    return list(set(error_codes))

def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))

def list_dir(folder):
    return [f for f in os.listdir(folder) if op.isdir(op.join(folder, f))]

def replace_place_holder(p, place_holder):
    if isinstance(p, dict):
        for k, v in p.items():
            if type(v) == str and v.startswith('$'):
                p[k] = place_holder[v[1:]]
            else:
                replace_place_holder(v, place_holder)
    elif isinstance(p, list) or isinstance(p, tuple):
        for i, x in enumerate(p):
            if isinstance(x, str) and x.startswith('$'):
                p[i] = place_holder[x[1:]]
            else:
                replace_place_holder(x, place_holder)

def execute_pipeline(all_processor):
    only_processors = [p for p in all_processor if p.get('only')]
    need_check_processors = only_processors if len(only_processors) > 0 else all_processor
    result = {}
    result['process_result'] = []
    place_holder = {}
    for p in need_check_processors:
        replace_place_holder(p, place_holder)
        if p.get('ignore', False):
            continue
        if 'execute_if_true_else_break' in p:
            r = execute_func(p['execute_if_true_else_break'])
            if not r:
                result['result'] = 'prereq_failed'
                result['failed_prereq'] = p['execute_if_true_else_break']
                break
        if p.get('continue_if_true_else_break'):
            r = execute_func(p['execute'])
            if not r:
                result['result'] = 'prereq_failed'
                result['failed_prereq'] = p['execute']
                break
            else:
                continue
        if p.get('force', False):
            r = run_if_not_cached(execute_func, p['execute'],
                    __force=True)
        else:
            r = run_if_not_cached(execute_func, p['execute'])
        if 'output' in p:
            place_holder[p['output']] = r
        else:
            if r is not None:
                result['process_result'].append({'process_info': p, 'return': r})
        if p.get('stop_after'):
            logging.info('skip the rest since stop_after=True')
            break
    result['place_holder'] = place_holder
    if 'result' not in result:
        result['result'] = 'pass'
    return result

def remove_empty_keys_(ds):
    keys = set([k for d in ds for k in d])
    empty_keys = [k for k in keys if all(d.get(k) is None for d in ds)]
    for k in empty_keys:
        for d in ds:
            if k in d:
                del d[k]

def max_iter_mult(m, factor):
    if isinstance(m, int):
        return int(m * factor)
    elif isinstance(m, str):
        assert m.endswith('e')
        return '{}e'.format(int(float(m[:-1]) * factor))
    else:
        raise NotImplementedError

def remove_empty_coco_style(rects, w, h):
    rects = [r for r in rects if r.get('iscrowd', 0) == 0]
    ret = []
    for r in rects:
        x1, y1, x2, y2 = r['rect']
        x1 = min(w, max(0, x1))
        x2 = min(w, max(0, x2))
        y1 = min(h, max(0, y1))
        y2 = min(h, max(0, y2))
        if y2 > y1 and x2 > x1:
            r['rect'] = [x1, y1, x2, y2]
            ret.append(r)
    return ret

def join_hints(hints, sep='_'):
    parts = []
    for h in hints:
        if isinstance(h, dict):
            parts.append(hash_sha1(h['hint'])[-h['max']:])
        else:
            parts.append(str(h))
    return sep.join(parts)

def qd_tqdm(*args, **kwargs):
    desc = kwargs.get('desc', '')
    import inspect
    frame = inspect.currentframe()
    frames = inspect.getouterframes(frame)
    frame = frames[1].frame
    line_number = frame.f_lineno
    fname = op.basename(frame.f_code.co_filename)
    message = '{}:{}'.format(fname, line_number)

    if 'desc' in kwargs:
        kwargs['desc'] = message + ' ' + desc
    else:
        kwargs['desc'] = message

    if 'mininterval' not in kwargs:
        # every 2 secons; default is 0.1 second which is too frequent
        kwargs['mininterval'] = 2

    return tqdm(*args, **kwargs)

def get_opened_files():
    import psutil
    proc = psutil.Process()
    return proc.open_files()

def print_opened_files():
    logging.info(pformat(get_opened_files()))

def save_parameters(param, folder):
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    write_to_yaml_file(param, op.join(folder,
        'parameters_{}.yaml'.format(time_str)))
    # save the env parameters
    # convert it to dict for py3
    write_to_yaml_file(dict(os.environ), op.join(folder,
        'env_{}.yaml'.format(time_str)))

def exclusive_open_to_read(fname, mode='r'):
    disable_lock = os.environ.get('QD_DISABLE_EXCLUSIVE_READ_BY_LOCK')
    if disable_lock is not None:
        disable_lock = int(disable_lock)
    if not disable_lock:
        user_name = get_user_name()
        lock_fd = acquireLock(op.join('/tmp',
            '{}_lock_{}'.format(user_name, hash_sha1(fname))))
    #try:
    # in AML, it could fail with Input/Output error. If it fails, we will
    # use azcopy as a fall back solution for reading
    if fname.endswith(".8b") and not os.environ["QD_USE_LINEIDX_8B"]:
        fp = limited_retry_agent(1, open, fname, mode)
    else:
        fp = limited_retry_agent(10, open, fname, mode)
    if not disable_lock:
        releaseLock(lock_fd)
    return fp

def inject_log_to_board(fname, folder, pattern, keys):
    logging.info(pattern)
    # from qd.qd_common import iter_match_document
    from .common import iter_match_document

    from torch.utils.tensorboard import SummaryWriter
    wt = SummaryWriter(log_dir=folder)
    x_keys = [k['key'] for k in keys if k['is_x']]
    x_key_time = any([
        k for k in keys
        if k['is_x'] and k.get('type') == 'time'
    ])
    if len(x_keys) == 1:
        x_key = x_keys[0]
    if len(x_keys) == 0:
        x_key = None
    added = 0
    for i, r in enumerate(iter_match_document(pattern, fname)):
        info = {}
        for k, x in zip(keys, r):
            if not k.get('type'):
                info[k['key']] = float(x)
            elif k['type'] == 'time':
                info[k['key']] = datetime.datetime.strptime(
                    x, '%Y-%m-%d %H:%M:%S')
        added += 1
        for k, v in info.items():
            if x_key and k == x_key:
                continue
            args = {'tag': k, 'scalar_value': v}
            if not x_key_time:
                if x_key:
                    args['global_step'] = info[x_key]
                else:
                    args['global_step'] = i
            else:
                args['walltime'] = (info[x_key]-datetime.datetime(1970,1,1)).total_seconds()
            wt.add_scalar(**args)
            #wt.add_scalar(tag=k, scalar_value=v, global_step=info[x_key])
    logging.info(added)

def auto_parse_log_line(line):
    must_have = ['iter', 'speed', 'loss', 'lr']
    result = {}
    if not all(m in line for m in must_have):
        # in this case, we will try to parse if there is like acc = {}
        if '=' not in line:
            return result
        parts = re.split(':|,|;', line)
        #parts = line.split(':')
        for p in parts:
            if '=' in p:
                sub_parts = p.split('=')
                if len(sub_parts) != 2:
                    continue
                k, v = map(lambda x: x.strip(), sub_parts)
                try:
                    result[k] = float(v)
                except:
                    continue
        if len(result) > 0:
            matched = re.match('([0-9-\s:]+)', list(line.split(','))[0])
            if matched is not None:
                x = matched.groups()[0]
                try:
                    result['time'] = datetime.strptime(
                        x.strip(), '%Y-%m-%d %H:%M:%S')
                except:
                    return {}
        return result
    else:
        parts = line.split('  ')
        #([0-9-\s:]*)
        for p in parts:
            kv = list(map(lambda x: x.strip(), p.split(':')))
            if len(kv) != 2:
                continue
            key = kv[0]
            vs = list(map(lambda x: x.strip(' ()'), kv[1].split(' ')))
            for idx_v, v in enumerate(vs):
                try:
                    v1 = float(vs[0])
                    if len(vs) > 1:
                        result[key + '_{}'.format(idx_v)] = v1
                    else:
                        result[key] = v1
                except:
                    continue
        if len(result):
            matched = re.match('([0-9-\s:]*)', line)
            x = matched.groups()[0]
            try:
                result['time'] = datetime.strptime(
                    x.strip(), '%Y-%m-%d %H:%M:%S')
            except:
                pass

        return result


def identity(x):
    # only used in pipeline
    return x

def recursive_type_convert(info, t, convert_func):
    if isinstance(info, (tuple, list)):
        return [recursive_type_convert(i, t, convert_func) for i in info]
    elif isinstance(info, dict):
        return dict((k, recursive_type_convert(v, t, convert_func)) for k, v in info.items())
    elif isinstance(info, t):
        return convert_func(info)
    else:
        return info

def blobfuse_umount(mount_folder):
    cmd = ['sudo', 'umount', mount_folder]
    cmd_run(cmd)

def blobfuse_mount(account_name, container_name, account_key,
                   mount_folder, cache_folder):
    ensure_directory(mount_folder)
    ensure_directory(cache_folder)
    cmd = [
        'blobfuse',
        mount_folder,
        '--tmp-path={}'.format(cache_folder),
        '--container-name={}'.format(container_name),
    ]
    env = {
        'AZURE_STORAGE_ACCOUNT': account_name,
        'AZURE_STORAGE_ACCESS_KEY': account_key,
    }
    cmd_run(cmd, env=env)

def blobfuse_mount_from_config(config, mount_folder):
    info = load_from_yaml_file(config)
    cache_folder = '/mnt/blobfuse/cache/{}'.format(hash_sha1(mount_folder))
    blobfuse_mount(
        account_name=info['account_name'],
        container_name=info['container_name'],
        account_key=info['account_key'],
        mount_folder=mount_folder,
        cache_folder=cache_folder,
    )

def query_all_opened_file_in_system():
    fs = []
    for proc in psutil.process_iter():
        for proc in psutil.process_iter():
            try:
                for item in proc.open_files():
                    fs.append(item.path)
            except Exception:
                pass
    return list(set(fs))

def has_handle(fpath, opened_files=None):
    fpath = op.abspath(op.realpath(fpath))
    if opened_files is None:
        for proc in psutil.process_iter():
            try:
                for item in proc.open_files():
                    if fpath == item.path:
                        return True
            except Exception:
                pass
        return False
    else:
        return fpath in opened_files

def submit_to_evalai(fname, message, challenge_id, phase_id):
    # if fname is too long, it will fail
    copy_file(fname, '/tmp/a.json')
    fname = '/tmp/a.json'
    cmd = [
        'evalai',
        'challenge',
        str(challenge_id),
        'phase',
        str(phase_id), 'submit', '--file',
        fname,
        '--private'
    ]
    input = 'y\n{}\n\n\n\n\n'.format(message)
    import subprocess as sp
    try:
        submission_command_stdout = cmd_run(cmd,
                         process_input=input.encode(),
                         stdout=sp.PIPE,
                         )[0].decode("utf-8")
    except Exception as ex:
        #ensure_remove_file('/tmp/a.json')
        if 'The maximum number of submission for today' in str(ex):
            logging.info(str(ex))
            return
        else:
            raise
    submission_id_regex = re.search("evalai submission ([0-9]+)", submission_command_stdout)
    submission_id = submission_id_regex.group(0).split()[-1]
    cmd = ["evalai", "submission", submission_id, "result"]
    #ensure_remove_file('/tmp/a.json')
    return ' '.join(cmd)

def submit_to_evalai_for_vqa(fname, message, std=False):
    if std:
        return submit_to_evalai(fname, message, 830, 1794)
    else:
        return submit_to_evalai(fname, message, 830, 1793)

def submit_to_evalai_for_textvqa(fname, message, std=False):
    if std:
        return submit_to_evalai(fname, message, 874, 1821)
    else:
        return submit_to_evalai(fname, message, 874, 1820)

def submit_to_evalai_for_vizwizvqa(fname, message, std=False):
    if std:
        return submit_to_evalai(fname, message, 1560, 3115)
    else:
        return submit_to_evalai(fname, message, 1560, 3114)

def submit_to_evalai_for_vizwizvqa2021(fname, message, std=False):
    if std:
        return submit_to_evalai(fname, message, 743, 1593)
    else:
        return submit_to_evalai(fname, message, 743, 1592)

def submit_to_evalai_for_vizwizvqa2022(fname, message, phase='std'):
    if phase == 'std':
        return submit_to_evalai(fname, message, 1560, 3115)
    elif phase == 'challenge':
        return submit_to_evalai(fname, message, 1560, 3116)

def submit_to_evalai_for_nocaps_xd(fname, split, message):
    challenge_id = 464
    if split == 'val':
        phase_id = 962
    else:
        assert split == 'test'
        phase_id = 963
    return submit_to_evalai(
        fname, message, challenge_id, phase_id)

def submit_to_evalai_for_nocaps(fname, split, message):
    challenge_id = 464
    if split == 'val':
        phase_id = 962
    else:
        assert split == 'test'
        phase_id = 963
    return submit_to_evalai(
        fname, message, challenge_id, phase_id)

def submit_to_evalai_for_vizwiz(fname, phase, message):
    challenge_id = 739
    if phase == 'dev':
        phase_id = 1579
    else:
        assert phase == 'std'
        phase_id = 1580
    return submit_to_evalai(
        fname, message, challenge_id, phase_id)

def submit_to_evalai_for_textcap(fname, phase, message):
    challenge_id = 906
    if phase == 'val':
        phase_id = 1872
    else:
        assert phase == 'test'
        phase_id = 1873
    return submit_to_evalai(
        fname, message, challenge_id, phase_id)

def recover_stdout_error():
    import sys
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def switch_case(switch, case, default):
    if isinstance(switch, (list, tuple)):
        path = '$'.join(map(str, switch))
        if dict_has_path(case, path):
            return dict_get_path_value(case, path)
        else:
            return default
    else:
        return case.get(switch, default)

class wb(object):
    initialized = False
    enabled = True

    @classmethod
    def ensure_initialized(cls):
        if not cls.initialized:
            cls.initialized = True
            cls.enabled = int(os.environ.get('QD_WANDB_ENABLED', '0'))
            if get_mpi_rank() != 0:
                cls.enabled = False
            if cls.enabled:
                try:
                    import wandb
                    # https://docs.wandb.ai/library/init#init-start-error
                    wandb.init(settings=wandb.Settings(start_method="fork"))
                except:
                    print_trace()
                    logging.info('init fails, disable wandb')
                    cls.enabled = False

    @classmethod
    def watch(cls, *args, **kwargs):
        cls.ensure_initialized()
        if cls.enabled:
            import wandb
            wandb.watch(*args, **kwargs)

    @classmethod
    def log(cls, *args, **kwargs):
        cls.ensure_initialized()
        if cls.enabled:
            import wandb
            wandb.log(*args, **kwargs)

def get_wandb_project(project):
    if len(project) > 64:
        from qd.qd_common import hash_sha1
        project = project[:64] + '_' + hash_sha1(project)
    return project

def query_log_by_wandb_project(project):
    project = get_wandb_project(project)
    import wandb

    api = wandb.Api()
    all_record = []
    for r in api.runs(project):
        df = r.history()
        records = df.to_dict('records')
        all_record.extend(records)
    all_record = sorted(all_record, key=lambda x: x['_timestamp'])
    return all_record

def find_mount_point(path):
    path = op.abspath(path)
    while not op.ismount(path):
        path = op.dirname(path)
    return path

@contextlib.contextmanager
def robust_open_to_write(fname, mode):
    tmp = fname + '.tmp'
    ensure_directory(op.dirname(tmp))
    with open(tmp, mode) as fp:
        yield fp
    os.rename(tmp, fname)

if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
