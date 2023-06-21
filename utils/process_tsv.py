# from .cloud_storage import create_cloud_fuse
import argparse
import base64
import copy
import cv2
import glob
import imghdr
import inspect
import json
import logging
import math
import numpy as np
import os
import os.path as op
import pymongo
import random
import re
import shutil
from collections import OrderedDict
import sys
import time
import unicodedata
import yaml
from .tsv_io import TSVSplitProperty
from .tsv_io import CompositeTSVFile
from .common import pilimg_from_base64
from .common import dict_get_all_path
from .tsv_io import get_tsv_associates
from .common import dict_get_path_value
from .common import qd_tqdm
from .common import qd_tqdm as tqdm
from .common import parallel_map
from .common import split_to_chunk_to_range
from .common import ensure_remove_file
from .common import get_tmp_folder

from collections import defaultdict
from deprecated import deprecated
from pathos.multiprocessing import ProcessingPool as Pool
from datetime import datetime
from pprint import pformat
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from future.utils import viewitems
from .cloud_storage import CloudStorage
from .process_image import draw_bb, show_image, save_image
from .process_image import show_images
from .process_image import draw_rects
from .common import calculate_image_ap
from .common import calculate_iou
from .common import dict_to_list
from .common import ensure_directory
from .common import normalize_to_str
from .common import generate_lineidx
from .common import get_mpi_rank, get_mpi_size
from .common import hash_sha1
from .common import ensure_copy_file
from .common import img_from_base64
from .common import init_logging
from .common import json_dump
from .common import list_to_dict, list_to_dict_unique
from .common import parse_test_data
from .common import read_to_buffer
from .tsv_io import load_list_file
from .common import worth_create
from .tsv_io import write_to_file
from .common import write_to_yaml_file, load_from_yaml_file
from .common import float_tolorance_equal
from .common import is_positive_uhrs_verified, is_negative_uhrs_verified
from .tsv_io import concat_files
from .common import try_delete
from .common import encoded_from_img
from .common import get_frame_info
from .common import calc_mean
from .taxonomy import child_parent_print_tree2
from .taxonomy import create_markdown_url
from .taxonomy import disambibuity_noffsets
from .taxonomy import gen_noffset
from .taxonomy import gen_term_list
from .taxonomy import get_nick_name
from .taxonomy import is_noffset
from .taxonomy import LabelToSynset, synset_to_noffset
from .taxonomy import load_all_tax, merge_all_tax
from .taxonomy import noffset_to_synset
from .taxonomy import populate_cum_images
from .taxonomy import populate_url_for_offset
from .taxonomy import Taxonomy
from .tsv_io import create_inverted_list
from .tsv_io import create_inverted_list2
from .tsv_io import extract_label
from .tsv_io import get_meta_file
from .tsv_io import load_labels
from .tsv_io import TSVDataset, TSVFile
from .tsv_io import tsv_reader, tsv_writer
from .tsv_io import is_verified_rect
from .tsv_io import tsv_copy
from .tsv_io import tsv_writers
from .tsv_io import get_default_splits
from .db import create_mongodb_client
from .db import create_bbverification_db
from .tsv_io import get_tsv_lineidx_8b
from .tsv_io import File
import torch


def delete_tsv_files(tsvs):
    for t in tsvs:
        if op.isfile(t):
            try_delete(t)
        line = op.splitext(t)[0] + '.lineidx'
        if op.isfile(line):
            try_delete(line)
        line = get_tsv_lineidx_8b(t)
        if op.isfile(line):
            try_delete(line)

def concat_tsv_files_mem_efficient(tsvs, out_tsv):
    # prefer concat_tsv_files
    if len(tsvs) == 1 and tsvs[0] == out_tsv:
        return
    concat_files(tsvs, out_tsv)
    sizes = [os.stat(t).st_size for t in tsvs]
    sizes = np.cumsum(sizes)
    debug_check = {'n': 0, 'n8b': 0}
    def gen_lineidx():
        for i, t in enumerate(tsvs):
            with File.open(op.splitext(t)[0] + '.lineidx') as fp:
                for idx in fp:
                    debug_check['n'] += 1
                    if i == 0:
                        yield int(idx)
                    else:
                        yield int(idx) + sizes[i - 1]
    with File.open(op.splitext(out_tsv)[0] + '.lineidx', 'w') as fp:
        for x in tqdm(gen_lineidx()):
            fp.write(str(x) + '\n')

    def gen_lineidx_8b():
        for i, t in enumerate(tsvs):
            with File.open(op.splitext(t)[0] + '.lineidx.8b', 'rb') as fp:
                while True:
                    x = fp.read(8)
                    if x != b'':
                        idx = int.from_bytes(x, 'little')
                        debug_check['n8b'] += 1
                        if i == 0:
                            yield idx
                        else:
                            yield int(idx) + sizes[i - 1]
                    else:
                        break
    with File.open(op.splitext(out_tsv)[0] + '.lineidx.8b', 'wb') as fp:
        for x in tqdm(gen_lineidx_8b()):
            fp.write(int(x).to_bytes(8, 'little'))
    assert debug_check['n'] == debug_check['n8b']

def concat_tsv_files(tsvs, out_tsv, gen_lineidx=True):
    if len(tsvs) == 1 and tsvs[0] == out_tsv:
        return
    File.prepare(tsvs)
    concat_files(tsvs, out_tsv)
    sizes = [File.get_file_size(t) for t in tsvs]
    sizes = np.cumsum(sizes)
    sizes = [0] + sizes[:-1].tolist()

    if gen_lineidx:
        # the generation process is much slower than that for lineidx.8b. As
        # this format is going to be deprecated, we have a flag to disable this
        # ggeneration proces
        concate_lineidx(sizes, tsvs, out_tsv)
    concate_lineidx_8b(sizes, tsvs, out_tsv)

def concate_lineidx(sizes, tsvs, out_tsv):
    File.prepare(tsvs)
    def row_processor(row):
        offset, in_tsv, out_tsv = row
        fbar = qd_tqdm(unit_scale=True)
        with File.open(in_tsv) as fp:
            with File.open(out_tsv, 'w') as fpout:
                for idx in fp:
                    fbar.update(1)
                    fpout.write(str(int(idx) + offset) + '\n')
    all_info = [(sizes[i], op.splitext(t)[0] + '.lineidx') for i, t in enumerate(tsvs)]
    folder = get_tmp_folder()
    all_info = [(offset, in_tsv, op.join(folder, in_tsv)) for offset, in_tsv in all_info]
    #for _, __, f in all_info:
        #ensure_directory(op.dirname(f))
    parallel_map(row_processor, all_info, 64)
    concat_files([i[2] for i in all_info], op.splitext(out_tsv)[0] + '.lineidx')
    for d in all_info:
        ensure_remove_file(d[2])

def concate_lineidx_8b(sizes, tsvs, out_tsv):
    File.prepare(tsvs)
    folder = get_tmp_folder()
    def row_processor_8b(row):
        offset, in_tsv, out_tsv = row
        fbar = qd_tqdm(unit_scale=True)
        bulk_size = 1024
        with File.open(in_tsv, 'rb') as fp:
            with File.open(out_tsv, 'wb') as fpout:
                while True:
                    x = fp.read(8 * bulk_size)
                    fbar.update(len(x) // 8)
                    if x != b'':
                        import struct
                        fmt = '<{}q'.format(len(x) // 8)
                        x = [i + offset for i in struct.unpack(fmt, x)]
                        fpout.write(b''.join([i.to_bytes(8, 'little') for i in
                                              x]))
                    else:
                        break
    all_info_8b = [(sizes[i], op.splitext(t)[0] + '.lineidx.8b') for i, t in enumerate(tsvs)]
    File.prepare([in_tsv for _, in_tsv in all_info_8b])
    # op.join(folder, in_tsv) may also be equal to in_tsv, although it is fine
    all_info_8b = [(offset, in_tsv, '{}/{}'.format(folder, in_tsv + '.lineidx.8b')) for offset, in_tsv
                   in all_info_8b]
    parallel_map(row_processor_8b, all_info_8b, 64)
    concat_files([i[2] for i in all_info_8b], op.splitext(out_tsv)[0] + '.lineidx.8b')
    for d in all_info_8b:
        ensure_remove_file(d[2])

def concat_tsv_files_old(tsvs, out_tsv):
    # prefer concat_tsv_files
    if len(tsvs) == 1 and tsvs[0] == out_tsv:
        return
    concat_files(tsvs, out_tsv)
    sizes = [os.stat(t).st_size for t in tsvs]
    sizes = np.cumsum(sizes)
    all_idx = []
    for i, t in enumerate(tsvs):
        for idx in load_list_file(op.splitext(t)[0] + '.lineidx'):
            if i == 0:
                all_idx.append(idx)
            else:
                all_idx.append(str(int(idx) + sizes[i - 1]))
    write_to_file('\n'.join(all_idx), op.splitext(out_tsv)[0] + '.lineidx')
    with File.open(get_tsv_lineidx_8b(out_tsv), 'wb') as fp:
        for i in all_idx:
            fp.write(int(i).to_bytes(8, 'little'))

def merge_data_feature_tsv(from_data, to_data, feature_version):
    assert from_data.startswith(to_data)
    from_dataset = TSVDataset(from_data)
    to_dataset = TSVDataset(to_data)
    src_tsvs = load_list_file(from_dataset.get_data(
        'trainX', 'feature', feature_version))
    dest_tsvs = load_list_file(to_dataset.get_data(
        'trainX',
        'feature',
        feature_version))
    assert len(dest_tsvs) == 1
    dest_tsv = dest_tsvs[0]
    logging.info(pformat(src_tsvs))
    logging.info(pformat(dest_tsv))
    if op.isfile(dest_tsv):
        logging.info('skip to merge as exists')
    else:
        concat_tsv_files(src_tsvs, dest_tsv)

def train_test_split(data_from, data_to, split_from='train', num_test=5000):
    populate_dataset_details(data_from)
    dataset_from = TSVDataset(data_from)
    dataset_to = TSVDataset(data_to)

    num_total_image = dataset_from.num_rows(split_from)
    total_idx = list(range(num_total_image))
    random.seed(666)
    random.shuffle(total_idx)
    split_names = ['test', 'train']
    all_split_idx = [total_idx[: num_test], total_idx[num_test:]]
    for split_idx, split_name in zip(all_split_idx, split_names):
        dataset_to.write_data(dataset_from.iter_data(split_from,
            filter_idx=split_idx), split_name)
        # all label file
        v = 0
        while dataset_from.has(split_from, 'label', version=v):
            dataset_to.write_data(dataset_from.iter_data(split_from,
                t='label', version=v, filter_idx=split_idx),
                split_name, t='label', version=v)
            v = v + 1

def find_best_matched_rect_idx(target, rects, check_class=True):
    target_class_lower = target['class'].lower()
    if check_class:
        same_class_rects = [r for r in rects if r['class'].lower() == target_class_lower]
    else:
        same_class_rects = rects
    idx_ious = [(i, calculate_iou(r['rect'], target['rect']))
        for i, r in enumerate(same_class_rects)]
    if len(idx_ious) == 0:
        return None, -1
    return max(idx_ious, key=lambda r: r[-1])

def find_best_matched_rect(target, rects, check_class=True, check_iou=True):
    target_class_lower = target['class'].lower()
    if check_class:
        same_class_rects = [r for r in rects if r['class'].lower() == target_class_lower]
    else:
        same_class_rects = rects
    if check_iou:
        rect_ious = [(r, calculate_iou(r['rect'], target['rect']))
            for r in same_class_rects]
    else:
        rect_ious = [(r, 1) for r in same_class_rects]
    if len(rect_ious) == 0:
        return None, -1
    return max(rect_ious, key=lambda r: r[-1])

def create_tsvdataset_from_image_folder(root_folder, data):
    from .process_image import load_image
    dataset = TSVDataset(data)
    all_full_file_name = [op.join(root, f) for root, dirnames, filenames in os.walk(root_folder)
        for f in filenames]
    def row_processor(full_file_name):
        im = load_image(full_file_name)
        if im is not None:
            key = full_file_name.replace(root_folder, '')
            return key, base64.b64encode(read_to_buffer(full_file_name))
    parallel_tsv_process(
        row_processor,
        all_full_file_name,
        dataset.get_data('train'),
        64
    )

def create_new_image_tsv_if_exif_rotated(data, split):
    dataset = TSVDataset(data)
    key_no_rotate_in_image = 'no_rotate_in_image'
    if dataset.has(split, key_no_rotate_in_image):
        return
    info = {'changed': 0, 'total': 0}
    def gen_rows():
        logging.info('{}-{}'.format(data, split))
        for key, str_rects, str_im in tqdm(dataset.iter_data(split)):
            info['total'] = info['total'] + 1
            im = img_from_base64(str_im)
            from .common import img_from_base64_ignore_rotation
            im2 = img_from_base64_ignore_rotation(str_im)
            if im.shape[0] != im2.shape[0] or im.shape[1] != im2.shape[1]:
                assert im.shape[0] == im2.shape[1]
                assert im.shape[1] == im2.shape[0]
                info['changed'] = info['changed'] + 1
                yield key, str_rects, encoded_from_img(im)
            else:
                yield key, str_rects, str_im
    tmp_tsv = dataset.get_data(split, 'tmp')
    tsv_writer(gen_rows(), tmp_tsv)
    logging.info(pformat(info))
    if info['changed'] != 0:
        from .tsv_io import tsv_mv
        tsv_mv(tmp_tsv, dataset.get_data(split))
    else:
        from .tsv_io import tsv_rm
        tsv_rm(tmp_tsv)
    dataset.write_data([], split, key_no_rotate_in_image)

def get_data_distribution(data, split, version):
    # return a dictionary to represent the distribution of the data
    # only composite dataset requires the following two information
    result = {}

    populate_dataset_details(data)
    dataset = TSVDataset(data)
    tsvsource_to_imageratio = {src[5:]:
            (float(numdestimages) / float(numsrcimages))
            for src, numdestimages, numsrcimages in
            dataset.iter_data(split + 'X', 'tsvsource.numdestimages.numsrcimages')}
    result['tsvsource_to_imageratio'] = tsvsource_to_imageratio

    tsvsource_to_boxratio = {src[5:]:
            (float(numdest) / float(numsrc))
            for src, numdest, numsrc in
            dataset.iter_data(split + 'X', 'tsvsource.numdestbbs.numsrcbbs')}
    result['tsvsource_to_boxratio'] = tsvsource_to_boxratio

    tsvsource_to_num_image = {source[5:]: int(num_image) for source, num_image in dataset.iter_data(split + 'X', 'numImagesPerSource')}
    result['tsvsource_to_num_image'] = tsvsource_to_num_image

    tsvsource_to_num_category = {source[5:]: int(num_image) for source, num_image
            in dataset.iter_data(split + 'X', 'numCategoriesPerSource')}
    result['tsvsource_to_num_category'] = tsvsource_to_num_category

    label_to_count = {l: int(c) for l, c in dataset.iter_data(split, 'inverted.label.count', version=version)}
    unique_labels = set([l if not l.startswith('-') else l[1:] for l in label_to_count])
    import math
    all_label_pos_negs = [[l, [label_to_count.get(l, 0), -label_to_count.get('-' + l, 0)]] for l in unique_labels]
    all_label_pos_negs = [[l, [math.log10(label_to_count.get(l, 0) + 1),
        -math.log10(label_to_count.get('-' + l, 0) + 1)]] for l in unique_labels]
    all_label_pos_negs = sorted(all_label_pos_negs, key=lambda x: x[1][0])
    result['all_label_pos_negs'] = all_label_pos_negs

    result['num_labels_with_neg'] = len([l for l, (pos, neg) in all_label_pos_negs if neg != 0])
    result['num_labels_with_pos'] = len([l for l, (pos, neg) in all_label_pos_negs if pos != 0])

    label_to_count = {l: int(c) for l, c in dataset.iter_data(split, 'inverted.label.count',
            version=version)}
    _, c = next(dataset.iter_data(split, 'inverted.background.count',
        version=version))
    label_to_count['__background'] = int(c)
    result['label_to_count'] = label_to_count

    return result

def convert_pred_to_dataset_label(full_expid, predict_file,
        th_file, min_value):
    pred_file = op.join('output',
            full_expid,
            'snapshot',
            predict_file)

    from .common import parse_test_data
    data, split = parse_test_data(predict_file)

    from process_tsv import load_key_rects

    dataset = TSVDataset(data)
    populate_dataset_details(data)
    latest_version = dataset.get_latest_version(split, 'label')
    gt_key_rects = load_key_rects(dataset.iter_data(split, 'label',
        latest_version))
    pred_key_rects = load_key_rects(tsv_reader(pred_file))
    pred_key_to_rects = {key: rects for key, rects in pred_key_rects}
    if th_file:
        th_file = op.join('output', full_expid, 'snapshot', th_file)
        per_cat_th = {l: max(float(th), min_value) for l, th, _ in tsv_reader(th_file)}
    else:
        per_cat_th = {}
    def gen_rows():
        for key, rects in gt_key_rects:
            pred_rects = pred_key_to_rects.get(key, [])
            pred_rects = [r for r in pred_rects
                    if r['conf'] >= per_cat_th.get(r['class'], 0)]
            yield key, json.dumps(pred_rects)
    info = [('full_expid', full_expid),
            ('predict_file', pred_file),
            ('th_file', th_file),
            ('min_value', min_value)]
    dataset.update_data(gen_rows(), split, 'label',
            generate_info=info)

    populate_dataset_details(data)

def ensure_inject_expid(full_expid):
    all_predict = []
    all_predict.extend(glob.glob(op.join('output', full_expid, 'snapshot',
        '*.predict')))
    all_predict.extend(glob.glob(op.join('output', full_expid, 'snapshot',
        '*.predict.tsv')))
    for p in all_predict:
        ensure_inject_expid_pred(full_expid, op.basename(p))

def ensure_inject_dataset(data, **kwargs):
    for split in get_default_splits():
        ensure_upload_image_to_blob(data, split)
        ensure_inject_image(data, split)
        ensure_inject_gt(data, split, **kwargs)

def ensure_inject_decorate(func):
    def func_wrapper(*args, **kwargs):
        client = create_mongodb_client()
        db = client['qd']
        task = db['task']
        # .func_name is ok for python2, but not for python3. .__name__ is ok
        # for both
        func_name = func.__name__
        if func_name.startswith('ensure_'):
            func_name = func_name[len('ensure_'): ]
        argnames, ins_varargs, ins_kwargs, ins_defaults = inspect.getargspec(func)
        # we have not supported if other paramseters is not None
        assert ins_varargs is None and ins_kwargs is None
        query = {'task_type': func_name}
        for n, v in zip(argnames[: len(args)], args):
            assert n not in query
            query[n] = v
        for k in kwargs:
            assert k not in query
            query[k] = kwargs[k]

        # create the unique index to make the update_one atomic
        logging.info('make sure the index is created: {}'.format(
            ', '.join(query.keys())))
        task.create_index([(k, pymongo.ASCENDING) for k in query])

        while True:
            result = task.update_one(filter=query,
                                    update={'$setOnInsert': query},
                                    upsert=True)
            if result.matched_count == 0:
                task.update_one(filter=query,
                        update={'$set': {'status': 'started',
                                         'create_time': datetime.now()}})
                func(*args, **kwargs)
                task.update_many(filter=query,
                        update={'$set': {'status': 'done'}})
                return True
            else:
                assert result.matched_count == 1
                existing_entries = list(task.find(query))
                if any(e for e in existing_entries if e['status'] == 'done'):
                    logging.info('ignore to inject since it is done: \n{}'.format(query))
                    break
                else:
                    logging.info('waiting to finish \n{}'.format(
                        pformat(existing_entries)))
                    time.sleep(10)
                # return False if not done
    return func_wrapper

def ensure_composite_key_url(data, split):
    dataset = TSVDataset(data)
    if not dataset.has(split):
        logging.info('data tsv does not exist')
        return
    if dataset.has(split, 'key.url'):
        logging.info('it exists')
        return
    data_to_dataset = {}
    uploaded_key_url = set()
    def gen_rows():
        for key, _ in dataset.iter_data(split, 'label'):
            c_data, c_split, c_key = parse_combine(key)
            if c_data in data_to_dataset:
                c_dataset = data_to_dataset[c_data]
            else:
                c_dataset = TSVDataset(c_data)
                data_to_dataset[c_data] = c_dataset
            if (c_data, c_split) not in uploaded_key_url:
                ensure_upload_image_to_blob(c_data, c_split)
                uploaded_key_url.add((c_data, c_split))
            _, url = c_dataset.seek_by_key(c_key, c_split, 'key.url')
            yield key, url
    dataset.write_data(gen_rows(), split, 'key.url')

def upload_image_to_blob(data, split, config=None):
    dataset = TSVDataset(data)
    if not dataset.has(split):
        logging.info('{} - {} does not exist'.format(data, split))
        return
    logging.info('{} - {}'.format(data, split))
    if op.isfile(dataset.get_data(split + 'X')):
        ensure_composite_key_url(data, split)
        logging.info('ignore to upload images for composite dataset')
        return
    parallel = True
    if not parallel:
        s = CloudStorage(config=config)
        def gen_rows():
            for key, _, str_im in tqdm(dataset.iter_data(split)):
                url_key = map_image_key_to_url_key(data, split, key)
                if sys.version_info.major == 3:
                    url = s.upload_stream(base64.b64decode(str_im),
                            'images/' + url_key + '.jpg')
                else:
                    url = s.upload_stream(StringIO(base64.b64decode(str_im)),
                            'images/' + url_key)
                yield key, url
        dataset.write_data(gen_rows(), split, 'key.url')
    else:
        from .common import split_to_chunk
        from .tsv_io import rm_tsv
        num_rows = dataset.num_rows(split)
        num_chunk = num_rows // 1000
        num_chunk = max(1, num_chunk)
        tasks = split_to_chunk(range(num_rows), num_chunk)
        tasks = [(data, split, t, i, len(tasks), config) for i, t in enumerate(tasks)]
        parallel_map(upload_image_to_blob_by_idx, tasks)
        # merge the result
        def gen_rows():
            for idx_task in range(len(tasks)):
                for key, url in dataset.iter_data(split,
                        'key.url.{}.{}'.format(idx_task, len(tasks))):
                    yield key, url
        dataset.write_data(gen_rows(), split, 'key.url')
        # remove intermediate files
        for idx_task in range(len(tasks)):
            rm_tsv(dataset.get_data(split, 'key.url.{}.{}'.format(idx_task, len(tasks))))

def upload_image_to_blob_by_idx(args):
    data, split, idxes, idx_task, num_task, config = args
    t = 'key.url.{}.{}'.format(idx_task, num_task)
    dataset = TSVDataset(data)
    if dataset.has(split, t):
        logging.info('return since exist')
        return
    s = CloudStorage(config=config)
    def gen_rows():
        for key, _, str_im in tqdm(dataset.iter_data(split, filter_idx=idxes)):
            url_key = map_image_key_to_url_key(data, split, key)
            if sys.version_info.major == 3:
                url = s.upload_stream(base64.b64decode(str_im),
                        'images/' + url_key + '.jpg')
            else:
                url = s.upload_stream(StringIO(base64.b64decode(str_im)),
                        'images/' + url_key)
            if s.sas_token:
                url += s.sas_token
            yield key, url
    dataset.write_data(gen_rows(), split, t)

@ensure_inject_decorate
def ensure_upload_image_to_blob(data, split, config=None):
    upload_image_to_blob(data, split, config=config)

@ensure_inject_decorate
def ensure_inject_image(data, split):
    dataset = TSVDataset(data)
    if not dataset.has(split):
        return

    client = create_mongodb_client()
    db = client['qd']
    images = db['image']
    images.delete_many(filter={'data': data, 'split': split})
    images.create_index([
        ('data', pymongo.ASCENDING),
        ('split', pymongo.ASCENDING),
        ('key', pymongo.ASCENDING),
        ],
        unique=True)
    all_data = []

    logging.info('injecting {} - {}'.format(data, split))
    injected = set()
    key_to_hw = None
    if dataset.has(split, 'hw'):
        key_to_hw = {key: [int(x) for x in hw.split(' ')] for key, hw in
                dataset.iter_data(split, 'hw')}
    assert dataset.has(split, 'key.url')
    key_to_url = {key: url for key, url in dataset.iter_data(split, 'key.url')}
    logging.info('injecting image for {} - {}'.format(data, split))
    for i, (key, _) in tqdm(enumerate(dataset.iter_data(split,
        'label'))):
        if key in injected:
            continue
        else:
            injected.add(key)
        url = key_to_url[key]
        doc = {'data': data,
                'split': split,
                'key': key,
                'idx_in_split': i,
                'url': url,
                'create_time': datetime.now(),
                }
        if key_to_hw:
            doc['height'], doc['width'] = key_to_hw[key]
        all_data.append(doc)
        if len(all_data) > 1000:
            images.insert_many(all_data)
            all_data = []
    if len(all_data) > 0:
        images.insert_many(all_data)
        all_data = []

@ensure_inject_decorate
def ensure_update_pred_with_correctness(data, split,
        full_expid, predict_file):
    ensure_inject_gt(data, split)
    ensure_inject_pred(full_expid, predict_file, data, split)

    update_pred_with_correctness(data, split,
        full_expid,
        predict_file)


def update_pred_with_correctness(test_data, test_split,
        full_expid,
        predict_file):
    client = create_mongodb_client()
    db = client['qd']
    _pred = db['predict_result']

    # get the latest version of the gt
    pipeline = [{'$match': {'data': test_data,
                            'split': test_split,
                            'version': {'$lte': 0}}},
                {'$group': {'_id': {'data': '$data',
                                    'split': '$split',
                                    'key': '$key',
                                    'action_target_id': '$action_target_id'},
                            'contribution': {'$sum': '$contribution'},
                            'class': {'$first': '$class'},
                            'rect': {'$first': '$rect'}}},
                {'$match': {'contribution': {'$gte': 1}}}, # if it is 0, it means we removed the box
                {'$addFields': {'data': '$_id.data',
                                'split': '$_id.split',
                                'key': '$_id.key'}},
                ]
    def get_query_pipeline(data, split, key, class_name):
        return [{'$match': {'$expr': {'$and': [{'$eq': ['$data', '$${}'.format(data)]},
                                               {'$eq': ['$split', '$${}'.format(split)]},
                                               {'$eq': ['$key', '$${}'.format(key)]},
                                               {'$eq': ['$class', '$${}'.format(class_name)]}]}}},
                 {'$group': {'_id': '$action_target_id',
                             'contribution': {'$sum': '$contribution'},
                             'create_time': {'$max': '$create_time'},
                             'rect': {'$first': '$rect'}}},
                 {'$match': {'contribution': {'$gte': 1}}},
                ]

    # get the target prediction bounding boxes
    pipeline = [{'$match': {'full_expid': full_expid,
                            'pred_file': predict_file}},
                {'$group': {'_id': {'data': '$data',
                                   'split': '$split',
                                   'key': '$key',
                                   'class': '$class'},
                           'pred_box': {'$push': {'rect': '$rect',
                                                  'conf': '$conf',
                                                  'pred_box_id': '$_id'}}}},
                {'$lookup': {'from': 'ground_truth',
                             'let': {'target_data': '$_id.data',
                                     'target_split': '$_id.split',
                                     'target_key': '$_id.key',
                                     'target_class': '$_id.class'},
                             'pipeline': get_query_pipeline('target_data',
                                                            'target_split',
                                                            'target_key',
                                                            'target_class'),
                             'as': 'gt_box'}},
            ]
    all_correct_box, all_wrong_box = [], []
    for row in tqdm(_pred.aggregate(pipeline, allowDiskUse=True)):
        curr_pred = row['pred_box']
        curr_gt = row['gt_box']
        curr_pred = sorted(curr_pred, key=lambda x: -x['conf'])
        for p in curr_pred:
            matched = [g for g in curr_gt if
                    not g.get('used', False) and
                    ('rect' not in p or
                        not p['rect'] or
                        'rect' not in g or
                        not g['rect'] or
                        calculate_iou(g['rect'], p['rect']) > 0.3)]
            if len(matched) == 0:
                all_wrong_box.append(p['pred_box_id'])
            else:
                all_correct_box.append(p['pred_box_id'])
                matched[0]['used'] = True
        if len(all_correct_box) > 1000:
            _pred.update_many({'_id': {'$in': all_correct_box}},
                    {'$set': {'correct': 1}})
            all_correct_box = []
        if len(all_wrong_box) > 1000:
            _pred.update_many({'_id': {'$in': all_wrong_box}},
                    {'$set': {'correct': 0}})
            all_wrong_box = []
    if len(all_correct_box) > 0:
        _pred.update_many({'_id': {'$in': all_correct_box}},
                {'$set': {'correct': 1}})
        all_correct_box = []
    if len(all_wrong_box) > 0:
        _pred.update_many({'_id': {'$in': all_wrong_box}},
                {'$set': {'correct': 0}})
        all_wrong_box = []

def ensure_inject_expid_pred(full_expid, predict_file):
    try:
        data, split = parse_test_data(predict_file)
    except:
        logging.info('ignore to inject {} - {}'.format(full_expid, predict_file))
        return
    ensure_upload_image_to_blob(data, split)
    ensure_inject_image(data, split)
    ensure_inject_gt(data, split)
    ensure_inject_pred(full_expid,
            predict_file,
            data,
            split)
    ensure_update_pred_with_correctness(data, split,
        full_expid, predict_file)

@ensure_inject_decorate
def ensure_inject_pred(full_expid, pred_file, test_data, test_split):
    client = create_mongodb_client()
    db = client['qd']
    pred_collection = db['predict_result']
    logging.info('cleaning {} - {}'.format(full_expid, pred_file))
    pred_collection.delete_many({'full_expid': full_expid, 'pred_file': pred_file})
    pred_file = op.join('output', full_expid, 'snapshot', pred_file)
    all_rect = []
    for key, label_str in tqdm(tsv_reader(pred_file)):
        rects = json.loads(label_str)
        rects = [r for r in rects if r['conf'] > 0.05]
        for i, r in enumerate(rects):
            r['full_expid'] = full_expid
            r['pred_file'] = op.basename(pred_file)
            r['data'] = test_data
            r['split'] = test_split
            r['key'] = key
        all_rect.extend(rects)
        if len(all_rect) > 10000:
            pred_collection.insert_many(all_rect)
            all_rect = []
    if len(all_rect) > 0:
        pred_collection.insert_many(all_rect)

def ensure_inject_gt(data, split, **kwargs):
    client = create_mongodb_client()
    db = client['qd']
    task = db['task']
    assert split is not None
    dataset = TSVDataset(data)
    version = 0
    set_previous_key_rects = False
    while dataset.has(split, 'label', version=version):
        query = {'task_type': 'inject_gt',
                                'data': data,
                                'split': split,
                                'version': version}
        result = task.update_one(filter=query,
                                update={'$setOnInsert': query},
                                upsert=True)
        if result.matched_count == 0:
            logging.info('start to inserting {}-{}-{}'.format(data,
                split, version))
            # no process is working on inserting current version
            task.update_one(filter=query,
                    update={'$set': {'status': 'started',
                                     'create_time': datetime.now()}})
            if not set_previous_key_rects:
                if version == 0:
                    previous_key_to_rects = {}
                else:
                    key_rects = load_key_rects(dataset.iter_data(split, 'label',
                        version=version-1))
                    previous_key_to_rects = {key: rects for key, rects in key_rects}
                set_previous_key_rects = True
            inject_gt_version(data, split, version, previous_key_to_rects,
                    **kwargs)
            task.update_many(filter={'task_type': 'inject_gt',
                'data': data,
                'split': split,
                'version': version}, update={'$set': {'status': 'done'}})
        else:
            # it is done or it is started by anohter process. let's skip
            # this version
            assert result.matched_count == 1
            while True:
                existing_entries = list(task.find(query))
                if any(e for e in existing_entries if e['status'] == 'done'):
                    logging.info('ignore to inject since it is done: \n{}'.format(query))
                    break
                elif len(existing_entries) == 0:
                    logging.info('we will do it')
                    version = version - 1
                    break
                else:
                    logging.info('waiting to finish \n{}'.format(
                        pformat(existing_entries)))
                    time.sleep(10)
            set_previous_key_rects = False
        version = version + 1

def inject_gt_version(data, split, version, previous_key_to_rects,
        delete_existing=True):
    client = create_mongodb_client()
    db = client['qd']
    gt = db['ground_truth']
    dataset = TSVDataset(data)
    if delete_existing:
        logging.info('deleting data={}, split={}, version={}'.format(
            data, split, version))
        gt.delete_many({'data': data, 'split': split, 'version': version})
    dataset = TSVDataset(data)

    if not dataset.has(split, 'label', version):
        return False
    all_rect = []
    logging.info('{}-{}-{}'.format(data, split, version))
    total_inserted = 0
    for idx_in_split, (key, label_str) in tqdm(enumerate(dataset.iter_data(
        split, 'label', version=version))):
        rects = json.loads(label_str)
        def add_to_all_rect(r, extra_info):
            r2 = copy.deepcopy(r)
            r2.update(extra_info)
            r2['idx_in_split'] = idx_in_split
            r2['data'] = data
            r2['split'] = split
            r2['key'] = key
            r2['action_target_id'] = hash_sha1([data, split, key, r['class'],
                r.get('rect', [])])
            r2['version'] = version
            r2['create_time'] = datetime.now()
            all_rect.append(r2)
        if version == 0:
            assert key not in previous_key_to_rects
            for r in rects:
                add_to_all_rect(r, {'contribution': 1})
        else:
            previous_rects = previous_key_to_rects[key]
            # use strict_rect_in_rects rather than rect_in_rects: if the higher
            # version contains more properies, we want to have the properies in
            # teh database also.
            previous_not_in_current = copy.deepcopy([r for r in previous_rects if
                    not strict_rect_in_rects(r, rects)])
            current_not_in_previous = copy.deepcopy([r for r in rects if
                    not strict_rect_in_rects(r, previous_rects)])
            for r in previous_not_in_current:
                add_to_all_rect(r, {'contribution': -1})
            for r in current_not_in_previous:
                add_to_all_rect(r, {'contribution': 1})
        previous_key_to_rects[key] = rects
        if len(all_rect) > 1000:
            db_insert_many(gt, all_rect)
            total_inserted = total_inserted + len(all_rect)
            logging.info('inserting data={}, split={}, version={}, curr_insert={}, total={}'.format(
                data, split, version, len(all_rect), total_inserted))
            all_rect = []

    if len(all_rect) > 0:
        total_inserted = total_inserted + len(all_rect)
        logging.info('inserting data={}, split={}, version={}, curr_insert={}, total={}'.format(
            data, split, version, len(all_rect), total_inserted))
        db_insert_many(gt, all_rect)
        all_rect = []
    return True

def db_insert_many(collection, all_info):
    for a in all_info:
        if '_id' in a:
            del a['_id']
    collection.insert_many(all_info)

class VisualizationDatabaseByMongoDB():
    def __init__(self):
        self._client = create_mongodb_client()
        self._db = self._client['qd']
        self._pred = self._db['predict_result']
        self._gt = self._db['ground_truth']

    def _get_positive_start(self, start_id, max_item):
        if start_id < 0:
            rank = pymongo.DESCENDING
            start_id = min(0, start_id + max_item)
            start_id = abs(start_id)
        else:
            rank = pymongo.ASCENDING
        return rank, start_id

    def query_pipeline(self, pipeline, collection, db_name):
        image_info = list(self._client[db_name][collection].aggregate(
            pipeline, allowDiskUse=True))
        logging.info(len(image_info))
        if len(image_info) > 0:
            logging.info(pformat(image_info[0]))
        image_info = [{'key': x['key'],
            'url': x.get('url', ''),
            'gt': x.get('gt', []),
            'pred': x.get('pred', [])} for x in image_info]
        return image_info

    def insert(self, dic, collection, db_name):
        return self._client[db_name][collection].insert(dic)

    def query_by_id(self, _id, collection, db_name):
        from bson.objectid import ObjectId
        return list(self._client[db_name][collection].find({'_id':
            ObjectId(_id)}))[0]

    def query_predict_recall(self, full_expid, pred_file, class_name, threshold, start_id, max_item):
        rank, start_id = self._get_positive_start(start_id, max_item)
        # from the predict file
        row = self._pred.find_one({'full_expid': full_expid, 'pred_file':
            pred_file})
        if row is None:
            logging.info('no prediction data in db')
            return []
        data = row['data']
        split = row['split']
        logging.info(data)
        logging.info(split)
        pipeline = [{'$match': {'data': data, 'split': split, 'class': class_name}},
                    {'$group': {'_id': {'key': '$key', 'target_action_id': '$target_action_id'},
                                'contribution': {'$sum': '$contribution'},
                                'num_target_label': {'$sum': 1}}},
                    {'$match': {'contribution': {'$gte': 1}}},
                    {'$group': {'_id': {'key': '$_id.key'},
                                'num_contribution': {'$sum': 1}}},
                    ## add the number of correct predicted
                    {'$lookup': {
                        'from': 'predict_result',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'full_expid': full_expid,
                                                 'pred_file': pred_file,
                                                 'class': class_name,
                                                 'conf': {'$gte': threshold}}},
                                     {'$group': {'_id': {'key': '$key'},
                                                 'correct': {'$sum': '$correct'}}},
                                      {'$match': {'$expr': {'$and': [{'$eq': ['$_id.key', '$$key']}]}}},
                                      ],
                        'as': 'recall',
                        }},
                    {'$unwind': {'path': '$recall', 'preserveNullAndEmptyArrays': True}},
                    {'$addFields': {'recall': {'$divide': ['$recall.correct', '$num_target_label']}}},
                    {'$sort': {'recall': rank}},
                    {'$skip': start_id},
                    {'$limit': max_item},
                    # add the field of url
                    {'$lookup': {
                        'from': 'image',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', data]},
                                                                    {'$eq': ['$split', split]},
                                                                    {'$eq': ['$key', '$$key']},],}}},
                                     {'$project': {'url': True, '_id': 0}}],
                        'as': 'url',
                        }},
                    {'$addFields': {'url': {'$arrayElemAt': ['$url', 0]}}},
                    {'$addFields': {'url': '$url.url'}},
                    ## add the field of pred
                    {'$lookup': {
                        'from': 'predict_result',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$full_expid', full_expid]},
                                                                    {'$eq': ['$pred_file', pred_file]},
                                                                    {'$eq': ['$key', '$$key']},
                                                                    {'$gte': ['$conf', threshold]}]}}},
                                     {'$project': {'conf': True, 'rect': True, 'class': True, '_id': 0}}],
                        'as': 'pred',
                        }
                    },
                    ## add the gt field
                    {'$lookup': {
                        'from': 'ground_truth',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', data]},
                                                                    {'$eq': ['$split', split]},
                                                                    {'$eq': ['$key', '$$key']},],}}},
                                    {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                                'contribution': {'$sum': '$contribution'},
                                                'rect': {'$first': '$rect'},
                                                'class': {'$first': '$class'},
                                                'conf': {'$first': '$conf'}}, },
                                    {'$match': {'contribution': {'$gte': 1}}},
                                    ],
                        'as': 'gt',
                        }}
                    ]
        image_info = list(self._gt.aggregate(pipeline))
        logging.info(len(image_info))
        image_info = [{'key': x['_id']['key'],
            'url': x['url'],
            'gt': x['gt'],
            'pred': x['pred']} for x in image_info]
        return image_info

    def query_predict_precision(self, full_expid, pred_file, class_name, threshold, start_id, max_item):
        rank, start_id = self._get_positive_start(start_id, max_item)
        filter_pred = {'full_expid': full_expid,
                            'pred_file': pred_file,
                            'conf': {'$gte': threshold}}
        if class_name != 'None' and class_name:
            filter_pred['class'] = class_name
        pipeline = [
                {'$match': filter_pred},
                {'$group': {'_id': {'data': '$data',
                                    'split': '$split',
                                    'key': '$key'},
                            'correct': {'$sum': '$correct'},
                            'total': {'$sum': 1}}},
                {'$addFields': {'precision': {'$divide': ['$correct', '$total']}}},
                {'$sort': {'precision': rank}},
                {'$skip': start_id},
                {'$limit': max_item},
                # add the field of url
                {'$lookup': {
                    'from': 'image',
                    'let': {'data': '$_id.data',
                            'split': '$_id.split',
                            'key': '$_id.key'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', '$$data']},
                                                                {'$eq': ['$split', '$$split']},
                                                                {'$eq': ['$key', '$$key']},],}}},
                                 {'$project': {'url': True, '_id': 0}}],
                    'as': 'url',
                    }},
                {'$unwind': '$url'},
                {'$addFields': {'url': '$url.url'}},
                ## add the field of pred
                {'$lookup': {
                    'from': 'predict_result',
                    'let': {'data': '$_id.data',
                            'split': '$_id.split',
                            'key': '$_id.key'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$full_expid', full_expid]},
                                                                {'$eq': ['$pred_file', pred_file]},
                                                                {'$eq': ['$key', '$$key']},
                                                                {'$gte': ['$conf', threshold]}]}}},
                                 {'$project': {'conf': True, 'rect': True, 'class': True, '_id': 0}}],
                    'as': 'pred',
                    }
                },
                ## add the gt field
                {'$lookup': {
                    'from': 'ground_truth',
                    'let': {'data': '$_id.data',
                            'split': '$_id.split',
                            'key': '$_id.key'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', '$$data']},
                                                                {'$eq': ['$split', '$$split']},
                                                                {'$eq': ['$key', '$$key']},],}}},
                                {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                            'contribution': {'$sum': '$contribution'},
                                            'rect': {'$first': '$rect'},
                                            'class': {'$first': '$class'},
                                            'conf': {'$first': '$conf'}}, },
                                {'$match': {'contribution': {'$gte': 1}}},
                                ],
                    'as': 'gt',
                    }}
                ]

        logging.info(pformat(pipeline))
        image_info = list(self._pred.aggregate(pipeline))
        image_info = [{'key': x['_id']['key'],
            'url': x['url'],
            'gt': x['gt'],
            'pred': x['pred']} for x in image_info]
        return image_info

    def query_ground_truth(self, data, split, version, label, start_id, max_item):
        pipeline = self.get_ground_truth_pipeline(data, split, version, label, start_id, max_item)
        image_info = list(self._gt.aggregate(pipeline['pipeline']))
        logging.info(len(image_info))
        image_info = [{'key': x['key'],
            'url': x['url'],
            'gt': x['gt']} for x in image_info]
        return pipeline, image_info

    def get_ground_truth_pipeline(self, data, split, version, label, start_id, max_item):
        rank, start_id = self._get_positive_start(start_id, max_item)

        match_pairs = {'data': data}
        gt_match = []
        gt_match.append({'$eq': ['$data', data]})
        url_match = []
        url_match.append({'$eq': ['$data', data]})
        if split is not None:
            match_pairs['split'] = split
            gt_match.append({'$eq': ['$split', split]})
            url_match.append({'$eq': ['$split', split]})
        if label is not None:
            match_pairs['class'] = label
        if version is not None:
            match_pairs['version'] = {'$lte': version}
            gt_match.append({'$lte': ['$version', version]})
        gt_match.append({'$eq': ['$key', '$$key']})
        url_match.append({'$eq': ['$key', '$$key']})
        pipeline = [{'$match': match_pairs},
                    {'$group': {'_id': {'key': '$key'},
                                'contribution': {'$sum': '$contribution'}}},
                    {'$match': {'contribution': {'$gte': 1}}},
                    {'$sort': {'_id.key': rank}},
                    {'$skip': start_id},
                    {'$limit': max_item},
                    # add gt boxes
                    {'$lookup': {
                        'from': 'ground_truth',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': gt_match}}},
                                     {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                                 'contribution': {'$sum': '$contribution'},
                                                 'rect': {'$first': '$rect'},
                                                 'class': {'$first': '$class'},
                                                 'conf': {'$first': '$conf'}}, },
                                     {'$match': {'contribution': {'$gte': 1}}},
                                    ],
                        'as': 'gt',
                        }},
                    # add url
                    {'$lookup': {
                        'from': 'image',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': url_match}}},
                                     {'$project': {'url': True, '_id': 0}}],
                        'as': 'url',
                        }},
                    {'$unwind': '$url'},
                    {'$addFields': {'url': '$url.url',
                                    'key': '$_id.key'}},
                    ]

        return {'pipeline': pipeline,
                'database': 'qd',
                'collection': 'ground_truth'}

def get_class_count(data, splits):
    dataset = TSVDataset(data)
    result = {}
    for split in splits:
        result[split] = {row[0]: int(row[1])
                for row in dataset.iter_data(
                    split, 'inverted.label.count', -1)}
    return result

def get_colored_name(node, colors):
    name = node.name
    if hasattr(node, 'sub_group'):
        sub_group = node.sub_group
    else:
        sub_group = -1
    idx = sub_group + 1
    while idx >= len(colors):
        colors.append('rgb({},{},{})'.format(
            int(random.random() * 255),
            int(random.random() * 255),
            int(random.random() * 255)))
    return "<span style='color:{}'>{}</span>".format(colors[idx],
            name)

def get_vis_url(data, l, split=None):
    if split is None:
        return '/detection/view_image?data={}&label={}&start_id=0'.format(
                data, l)
    else:
        return '/detection/view_image?data={}&split={}&label={}&start_id=0'.format(
                data, split, l)


def get_readable_label(l):
    if is_noffset(l):
        return '{}({})'.format(l, get_nick_name(noffset_to_synset(l)))
    else:
        return l

def get_node_key(node, k, default_value=-1):
    r = default_value
    if hasattr(node, k):
        r = node.__getattribute__(k)
    return r

def get_node_info(data, node, colors, all_data):
    ss = []
    if hasattr(node, 'ap'):
        ss.append('ap({0:.2f})'.format(node.ap))
    if hasattr(node, 'images_with_bb') and \
            hasattr(node, 'images_no_bb'):
        keys = ['with_bb', 'no_bb']
        for k in keys:
            ss.append("{}({},<span><a href='{}' target='_blank'>{}</a></span>,<span><a href='{}' target='_blank'>{}</a></span>)".format(
                k,
                get_node_key(node, 'images_' + k, -1),
                get_vis_url(data + '_' + k, node.name, split='train'),
                get_node_key(node, k + '_train', -1),
                get_vis_url(data + '_' + k, node.name, split='test'),
                get_node_key(node, k + '_test', -1)))
    ignore_keys = ['support', 'dist', 'name', 'url',
            'with_bb_train',
            'with_bb_test',
            'no_bb_train',
            'no_bb_test',
            'ap',
            'images_with_bb',
            'images_no_bb',
            'noffset',
            'sub_group',
            'cum_images_with_bb',
            'cum_images_no_bb']
    for key in node.features:
        if any(x == key for x in ignore_keys):
            continue
        if key.endswith('_readable') or \
                key.endswith('_toTrain') or \
                key.endswith('_toTest') or \
                key.endswith('_total'):
            continue
        value = node.__getattribute__(key)
        if key in all_data and value != None:
            total = get_node_key(node, '{}_total'.format(key))
            toTrain = get_node_key(node, '{}_toTrain'.format(key))
            toTest = get_node_key(node, '{}_toTest'.format(key))
            extra_info = '({},{},{})'.format(total, toTrain, toTest)
            labels = value.split(',')
            value = ','.join(["<a href='{}' target='_blank'>{}</a>".format(get_vis_url(key, l),
                get_readable_label(l)) for l in labels])
            value = '{}{}'.format(value, extra_info)
        ss.append('{}[{}]'.format(key, value))
    if len(ss) > 0:
        return "{}: {}".format(
                get_colored_name(node, colors),
                '; '.join(ss))
    else:
        return get_colored_name(node, colors)

def gen_html_tree_view(data, full_expid=None,
        predict_file=None):
    colors=['rgb(0,0,0)',
                'rgb(255,0,0)',
                'rgb(0,0,255)']
    dataset = TSVDataset(data)
    all_data = os.listdir('./data')
    file_name = op.join(dataset._data_root, 'root_enriched.yaml')
    logging.info('loading {}'.format(file_name))
    with open(file_name, 'r') as fp:
        config_tax = yaml.load(fp)
    tax = Taxonomy(config_tax)
    if full_expid is not None and \
            predict_file is not None:
        map_file = op.join('output',full_expid, 'snapshot',
                op.splitext(predict_file)[0] + '.report.class_ap.json')
        if op.isfile(map_file):
            class_ap = json.loads(read_to_buffer(map_file))
            class_ap = class_ap['overall']['0.3']['class_ap']
            for node in tax.root.iter_search_nodes():
                node.add_feature('ap',
                        class_ap.get(node.name, -1))

    def gen_html_tree_view_rec(root):
        '''
        include itself
        '''
        if len(root.children) == 0:
            s = u"<li data-jstree='{{\"icon\":\"glyphicon glyphicon-leaf\"}}'><span>{}</span></li>".format(
                    get_node_info(data, root, colors, all_data))
            return s
        else:
            result = []
            # we cannot remove span tag here
            result.append("<li data-jstree='{{\"opened\":true}}' ><span>{}</span>".format(
                get_node_info(data, root, colors, all_data)))
            result.append('<ul>')
            all_child = sorted(root.children, key=lambda c: c.sub_group)
            for c in all_child:
                r = gen_html_tree_view_rec(c)
                result.append(r)
            result.append('</ul>')
            result.append('</li>')
            return '\n'.join(result)
    s = gen_html_tree_view_rec(tax.root)
    return s


def gt_predict_images(predicts, gts, test_data, target_images, label, start_id, threshold,
        label_to_idx, image_aps, test_data_split='test'):
    test_dataset = TSVDataset(test_data)
    test_tsv = TSVFile(test_dataset.get_data(test_data_split))
    for i in xrange(start_id, len(target_images)):
        key = target_images[i]
        logging.info('key = {}, ap = {}'.format(key, image_aps[i][1]))
        idx = label_to_idx[key]
        row = test_tsv.seek(idx)
        origin = img_from_base64(row[2])
        im_gt = np.copy(origin)
        draw_bb(im_gt, [g['rect'] for g in gts[key]],
                [g['class'] for g in gts[key]])
        im_gt_target = np.copy(origin)
        gts_target = [g for g in gts[key] if g['class'] == label]
        draw_bb(im_gt_target, [g['rect'] for g in gts_target],
                [g['class'] for g in gts_target])
        im_pred = np.copy(origin)
        rects = [p for p in predicts[key] if p['conf'] > threshold]
        draw_bb(im_pred, [r['rect'] for r in rects],
                [r['class'] for r in rects],
                [r['conf'] for r in rects])
        im_pred_target = np.copy(origin)
        rects = [p for p in rects if p['class'] == label]
        draw_bb(im_pred_target, [r['rect'] for r in rects],
                [r['class'] for r in rects],
                [r['conf'] for r in rects])
        yield key, origin, im_gt_target, im_pred_target, im_gt, im_pred, image_aps[i][1]

def get_confusion_matrix_by_predict_file_label(full_expid,
        predict_file, label, threshold):
    '''
    get confusion matrix for specific label
    '''
    test_data, test_data_split = parse_test_data(predict_file)

    # load the gt
    logging.info('loading {} - {}'.format(test_data, test_data_split))
    test_dataset = TSVDataset(test_data)
    rows = test_dataset.iter_data(test_data_split, 'label', -1)
    gts = {}
    label_to_idx = {}
    keys_with_label = []
    for i, row in enumerate(rows):
        rects = json.loads(row[1])
        gts[row[0]] = rects
        label_to_idx[row[0]] = i
        if any(r['class'] == label for r in rects):
            keys_with_label.append(row[0])

    predict_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    if not full_expid.startswith('Tax'):
        logging.info('loading {}'.format(predict_file))
        predicts, _ = load_labels(predict_full_path)
    else:
        # create the inverted index
        pred_label_file = '{}.labelmap.tsv'.format(predict_full_path)
        pred_inverted_file = '{}.inverted.tsv'.format(predict_full_path)
        pred_key_file = '{}.key.tsv'.format(predict_full_path)
        if not op.isfile(pred_label_file) or \
                not op.isfile(pred_inverted_file) or \
                not op.isfile(pred_key_file):
            pred_inverted = {}
            pred_keys = []
            rows = tsv_reader(predict_full_path)
            logging.info('loading data and creating index')
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(predict_full_path))
            logging.info('done loading data and creating index')
            pred_labels = inverted.keys()
            tsv_writer(([l] for l in pred_labels), pred_label_file)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_file)
            tsv_writer([[k] for k in pred_keys], pred_key_file)
        # find out data whose prediction has the target label
        all_labels = load_list_file(pred_label_file)
        if label in all_labels:
            row = TSVFile(pred_inverted_file).seek(all_labels.index(label))
            assert row[0] == label
            idx = map(int, row[1].split(' '))
        else:
            idx = []
        # find out the index from the ground truth
        key_to_predidx = {k: i for i, k in enumerate(load_list_file(pred_key_file))}
        idx.extend([key_to_predidx[k] for k in keys_with_label])
        idx = set(idx)
        tsv = TSVFile(predict_full_path)
        predicts = {}
        logging.info('loading')
        for i in idx:
            row = tsv.seek(i)
            assert len(row) == 2
            predicts[row[0]] = json.loads(row[1])
        logging.info('done')
        # load data from the inverted index
    logging.info('done loading {}'.format(predict_file))

    return {'predicts': predicts,
            'gts': gts,
            'label_to_idx': label_to_idx}

def get_confusion_matrix_by_predict_file(full_expid,
        predict_file, threshold):
    '''
    get confusion matrix for all classes
    '''

    test_data, test_data_split = parse_test_data(predict_file)

    # load the gt
    logging.info('loading {} - {}'.format(test_data, test_data_split))
    test_dataset = TSVDataset(test_data)
    rows = test_dataset.iter_data(test_data_split, 'label', -1)
    gts = {}
    label_to_idx = {}
    for i, row in enumerate(rows):
        gts[row[0]] = json.loads(row[1])
        label_to_idx[row[0]] = i

    logging.info('loading {}'.format(predict_file))
    predicts, _ = load_labels(op.join('output', full_expid, 'snapshot', predict_file))
    logging.info('done loading {}'.format(predict_file))

    # calculate the confusion matrix
    confusion_pred_gt = {}
    confusion_gt_pred = {}
    update_confusion_matrix(predicts, gts, threshold,
            confusion_pred_gt,
            confusion_gt_pred)

    return {'predicts': predicts,
            'gts': gts,
            'confusion_pred_gt': confusion_pred_gt,
            'confusion_gt_pred': confusion_gt_pred,
            'label_to_idx': label_to_idx}

def inc_one_dic_dic(dic, c1, c2):
    if c1 not in dic:
        dic[c1] = {}
    if c2 not in dic[c1]:
        dic[c1][c2] = 0
    dic[c1][c2] = dic[c1][c2] + 1

def update_confusion_matrix(predicts, gts, threshold,
            confusion_pred_gt,
            confusion_gt_pred):
    for key in predicts:
        curr_pred = [p for p in predicts[key] if p['conf'] > threshold]
        if key not in gts:
            continue
        curr_gt = gts[key]
        if len(curr_pred) == 0 and len(curr_gt) > 0:
            for g in curr_gt:
                inc_one_dic_dic(confusion_gt_pred, g['class'], 'None')
            continue
        elif len(curr_pred) > 0 and len(curr_gt) == 0:
            for p in curr_pred:
                inc_one_dic_dic(confusion_pred_gt, p['class'], 'None')
            continue
        elif len(curr_pred) == 0 and len(curr_gt) == 0:
            continue
        ious = np.zeros((len(curr_pred), len(curr_gt)))
        for i, p in enumerate(curr_pred):
            for j, g in enumerate(curr_gt):
                iou = calculate_iou(p['rect'], g['rect'])
                ious[i, j] = iou
        gt_idx = np.argmax(ious, axis=1)
        for i, p in enumerate(curr_pred):
            j = gt_idx[i]
            predict_class = p['class']
            gt_class = curr_gt[j]['class']
            if ious[i, j] > 0.3:
                inc_one_dic_dic(confusion_pred_gt,
                        predict_class, gt_class)
            else:
                inc_one_dic_dic(confusion_pred_gt,
                        predict_class, 'None')
        pred_idx = np.argmax(ious, axis=0)
        for j, g in enumerate(curr_gt):
            i = pred_idx[j]
            predict_class = curr_pred[i]['class']
            gt_class = g['class']
            if ious[i, j] > 0.3:
                inc_one_dic_dic(confusion_gt_pred,
                        gt_class, predict_class)
            else:
                inc_one_dic_dic(confusion_gt_pred,
                        gt_class, 'None')

def normalize_str_in_rects(data, out_data):
    '''
    normalize all the unicode to string
    '''
    dataset = TSVDataset(data)
    dest_dataset = TSVDataset(out_data)
    splits = get_default_splits()
    for split in splits:
        if not op.isfile(dataset.get_data(split)):
            continue
        def gen_rows():
            for key, label_str, im_str in tsv_reader(dataset.get_data(split)):
                rects = json.loads(label_str)
                for rect in rects:
                    s = rect['class']
                    rect['class'] = normalize_to_str(s)
                    if s != rect['class']:
                        logging.info(u'{}->{}'.format(s, rect['class']))
                label_str = json.dumps(rects)
                yield key, label_str, im_str
        tsv_writer(gen_rows(), dest_dataset.get_data(split))

def tsv_details(row_hw, row_label, num_rows):
    label_count = {}
    sizes = []
    logging.info('tsv details...')
    for r_hw, r_label in tqdm(zip(row_hw, row_label), total=num_rows):
        if r_label[1] == 'd':
            # this is the deleted label
            rects = []
        else:
            rects = json.loads(r_label[1])
        assert r_hw[0] == r_label[0]
        height, width = map(int, r_hw[1].split(' '))
        if type(rects) is list:
            # this is the detection dataset
            # convert it to str. if it is unicode, in yaml, there will be some
            # special tags, which is annoying
            curr_labels = set(normalize_to_str(rect['class']) for rect in rects)
            for rect in rects:
                if 'rect' not in rect or all(x == 0 for x in rect['rect']):
                    # it is image-level annotation
                    continue
                r = rect['rect']
                # this should be a valid bounding box
                cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                rw, rh = r[2] - r[0], r[3] - r[1]
                #assert cx >= 0 and cx < width \
                        #and cy >= 0 and cy < height
                if rw < 1 or rh < 1:
                    logging.warn('rw or rh too small: {} - {}'.format(r_label[0],
                        ','.join(map(str, r))))
        else:
            # this is classification dataset
            assert type(rects) is int
            curr_labels = [rects]
        for c in curr_labels:
            if c in label_count:
                label_count[c] = label_count[c] + 1
            else:
                label_count[c] = 1
        sizes.append((height, width))
    size_counts = [s[0] * s[1] for s in sizes]
    min_size = sizes[np.argmin(size_counts)]
    max_size = sizes[np.argmax(size_counts)]
    min_size = map(float, min_size)
    max_size = map(float, max_size)
    mean_size = (np.mean([s[0] for s in sizes]),
            np.mean([s[1] for s in sizes]))
    mean_size = map(float, mean_size)

    return {'label_count': label_count,
            'min_image_size': min_size,
            'max_image_size': max_size,
            'mean_image_size': mean_size}


def detect_duplicate_key(tsv, duplicate_tsv):
    rows = tsv_reader(tsv)
    key_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        if key in key_to_idx:
            key_to_idx[key].append(i)
        else:
            key_to_idx[key] = [i]
    def gen_rows():
        for key in key_to_idx:
            idxs = key_to_idx[key]
            if len(idxs) > 1:
                logging.info('duplicate key: {}: {}'.format(key, ', '.join(map(str,
                    idxs))))
                yield key, ','.join(map(str, idxs))
    tsv_writer(gen_rows(), duplicate_tsv)
    return TSVFile(duplicate_tsv).num_rows()

def populate_all_dataset_details():
    all_data = os.listdir('data/')
    for data in all_data:
        try:
            populate_dataset_details(data)
        except:
            continue

def update_labelmap(rows, num_rows, labelmap):
    '''
    labelmap is a hashset and can be added
    '''
    logging.info('updating labelmap')
    for row in tqdm(rows, total=num_rows):
        assert len(row) == 2
        try:
            labelmap.update(set([rect['class'] for rect in
                json.loads(row[1])]))
        except:
            labelmap.add(row[1])
    logging.info('done')

def derive_composite_meta_data(data, split, t):
    data_to_dataset = {}
    dataset = TSVDataset(data)
    all_source_data_split = dataset.load_composite_source_data_split(split)
    def gen_rows():
        assert dataset.has(split, 'label')
        iter_label = dataset.iter_data(split, 'label')
        iter_shuffle = tsv_reader(dataset.get_shuffle_file(split))
        for (idx_source, idx_row), (label_row) in zip(iter_shuffle,
                iter_label):
            idx_source = int(idx_source)
            idx_row = int(idx_row)
            c_data, c_split = all_source_data_split[idx_source]
            if c_data in data_to_dataset:
                c_dataset = data_to_dataset[c_data]
            else:
                c_dataset = TSVDataset(c_data)
                data_to_dataset[c_data] = c_dataset
            assert c_dataset.has(c_split, t)
            key, t_data = c_dataset.seek_by_idx(idx_row, c_split, t)
            yield label_row[0], t_data
    assert not dataset.has(split, t)
    dataset.write_data(gen_rows(), split, t)

def ensure_create_inverted_tsvs(dataset, splits):
    # for each data tsv, generate the inverted file, and the labelmap
    is_parallel = True
    for split in splits:
        if not dataset.has(split):
            continue
        latest = dataset.get_latest_version(split, 'label')
        if not is_parallel:
            for v in range(latest + 1):
                assert dataset.has(split, 'label', v)
                ensure_create_inverted_tsv_for_each((dataset, split, v))
        else:
            params = [(dataset, split, v) for v in range(latest + 1)]
            params = [param for param in params if
                    not has_inverted(param)]
            if len(params) > 0:
                p = Pool()
                p.map(ensure_create_inverted_tsv_for_each, params)

    # generate the inverted tsv for background images without any labels
    for split in splits:
        if not dataset.has(split):
            continue
        latest = dataset.get_latest_version(split, 'label')
        if not is_parallel:
            for v in range(latest + 1):
                assert dataset.has(split, 'label', v)
                ensure_create_inverted_tsv_background_for_each((dataset, split, v))
        else:
            params = [(dataset, split, v) for v in range(latest + 1)]
            params = [param for param in params if
                    not has_inverted_background(param)]
            if len(params) > 0:
                p = Pool()
                p.map(ensure_create_inverted_tsv_background_for_each, params)

def has_inverted(param):
    dataset, split, v = param
    inverted_keys = ['inverted.label',
            'inverted.label.with_bb',
            'inverted.label.no_bb',
            'inverted.label.with_bb.verified',
            'inverted.label.with_bb.noverified',]
    return all(dataset.has(split, k, v) for k in inverted_keys)

def has_inverted_background(args):
    dataset, split, v = args
    if dataset.has(split, 'inverted.background', version=v) and \
            dataset.has(split, 'inverted.background.count'):
        return True
    else:
        return False

def ensure_create_inverted_tsv_background_for_each(args):
    dataset, split, v = args
    if dataset.has(split, 'inverted.background', version=v) and \
            dataset.has(split, 'inverted.background.count'):
        return
    label_to_indices = dataset.load_inverted_label(split, version=v)
    all_idx = [i for l in label_to_indices for i in label_to_indices[l]]
    all_idx = set(all_idx)
    num = dataset.num_rows(split)
    background_idx = set(range(num)).difference(all_idx)

    def gen_rows():
        yield 'background', ' '.join(map(str, background_idx))
    dataset.write_data(gen_rows(), split, 'inverted.background', version=v)

    def gen_row_count():
        yield 'background', len(background_idx)
    dataset.write_data(gen_row_count(), split, 'inverted.background.count',
            version=v)

def ensure_create_inverted_tsv_for_each(args):
    dataset, split, v = args
    if not dataset.has(split, 'labelmap', v) or \
        dataset.last_update_time(split, 'labelmap', v) < dataset.last_update_time(split, 'label', v):
        curr_labelmap = set()
        update_labelmap(dataset.iter_data(split, 'label', v),
                dataset.num_rows(split),
                curr_labelmap)
        curr_labelmap = sorted(list(curr_labelmap))
        dataset.write_data([[l] for l in curr_labelmap], split, 'labelmap', v)
    else:
        curr_labelmap = None
    inverted_keys = ['inverted.label',
            'inverted.label.with_bb',
            'inverted.label.no_bb',
            'inverted.label.with_bb.verified',
            'inverted.label.with_bb.noverified',]
    if any(not dataset.has(split, k, v) for k in inverted_keys):
        logging.info('version = {}'.format(v))
        if curr_labelmap is None:
            curr_labelmap = []
            for row in dataset.iter_data(split, 'labelmap', v):
                assert len(row) == 1
                curr_labelmap.append(row[0])
        def gen_inverted_rows(inv):
            logging.info('re-orderring')
            set_labelmap = set(curr_labelmap)
            for label in tqdm(inv):
                assert label in set_labelmap
            for label in tqdm(curr_labelmap):
                i = inv[label] if label in inv else []
                yield label, ' '.join(map(str, i))
        inverted_result = create_inverted_list(
                dataset.iter_data(split, 'label', v))
        for k in inverted_keys:
            dataset.write_data(gen_inverted_rows(inverted_result[k]),
                    split, k, v)

def ensure_extract_from_data_source(data, split, t):
    assert t != 'label'
    dataset = TSVDataset(data)
    if dataset.has(split, t):
        logging.info('ignore to generate {}, {}, {}'.format(data, split, t))
        return
    all_data_split = dataset.load_composite_source_data_split(split)
    source_data_to_dataset = {}
    label_iter = dataset.iter_data(split, 'label')
    shuffle_iter = tsv_reader(dataset.get_shuffle_file(split))
    def gen_rows():
        for (idx_source, idx_row), label_row in zip(shuffle_iter, label_iter):
            idx_source, idx_row = int(idx_source), int(idx_row)
            source_data, source_split = all_data_split[idx_source]
            if source_data not in source_data_to_dataset:
                source_data_to_dataset[source_data] = TSVDataset(source_data)
            source_dataset = source_data_to_dataset[source_data]
            source_key, str_t = source_dataset.seek_by_idx(idx_row, source_split, t)
            assert label_row[0] == '_'.join([source_data, source_split, source_key])
            yield label_row[0], str_t
    dataset.write_data(gen_rows(), split, t)

def populate_dataset_hw(data,
                        splits=('train', 'trainval', 'test', 'val'),
                        img_t=None,
                        multi_thread=True,
                        ):
    dataset = TSVDataset(data)
    for split in splits:
        if dataset.has(split, img_t) and not dataset.has(split, 'hw'):
            success = False
            if op.isfile(dataset.get_data(split + 'X', img_t)):
                try:
                    derive_composite_meta_data(data, split, 'hw')
                    success = True
                except:
                    pass
            if not success:
                if not multi_thread:
                    logging.info('generating hw')
                    rows = dataset.iter_data(split, img_t, progress=True)
                    dataset.write_data(((row[0], ' '.join(map(str,
                        img_from_base64(row[-1]).shape[:2]))) for
                        row in rows), split, 'hw')
                else:
                    num_images = dataset.num_rows(split, img_t)
                    num_tasks = min(num_images, 128 * 3)
                    num_worker = (num_tasks + 2) // 3
                    num_image_per_worker = (num_images + num_tasks - 1) // num_tasks
                    assert num_image_per_worker > 0
                    all_idx = []
                    for i in range(num_tasks):
                        curr_idx_start = i * num_image_per_worker
                        if curr_idx_start >= num_images:
                            break
                        curr_idx_end = curr_idx_start + num_image_per_worker
                        curr_idx_end = min(curr_idx_end, num_images)
                        if curr_idx_end > curr_idx_start:
                            all_idx.append(range(curr_idx_start, curr_idx_end))
                    logging.info('creating pool')
                    m = Pool(num_worker)
                    def get_hw(filter_idx):
                        dataset = TSVDataset(data)
                        rows = dataset.iter_data(split, img_t, progress=True,
                                filter_idx=filter_idx)
                        return [(row[0], ' '.join(map(str,
                            img_from_base64(row[-1]).shape[:2])))
                            for row in rows]
                    all_result = m.map(get_hw, all_idx)
                    x = []
                    for r in all_result:
                        x.extend(r)
                    dataset.write_data(x, split, 'hw')

def ensure_label_extract(data, splits):
    dataset = TSVDataset(data)
    for split in splits:
        if (not dataset.has(split, 'label')) and \
                dataset.has(split):
            def gen_rows():
                if op.isfile(dataset.get_data(split)):
                    iter_row = dataset.iter_data(split)
                else:
                    iter_row = dataset.iter_composite(split, t=None, version=0)
                for row in tqdm(iter_row):
                    if len(row) == 3:
                        str_rects = row[1]
                    else:
                        assert len(row) == 2
                        str_rects = json_dump([])
                    yield row[0], str_rects
            dataset.write_data(gen_rows(), split, 'label')

def ensure_labelmap_extract(data):
    splits = get_default_splits()
    dataset = TSVDataset(data)
    # generate the label map if there is no
    if not op.isfile(dataset.get_labelmap_file()) and \
            not op.islink(dataset.get_labelmap_file()):
        logging.info('no labelmap, generating...')
        labelmap = []
        for split in splits:
            label_tsv = dataset.get_data(split, 'label', version=0)
            if not op.isfile(label_tsv):
                continue
            for row in tqdm(tsv_reader(label_tsv)):
                try:
                    labelmap.extend(set([rect['class'] for rect in
                        json.loads(row[1])]))
                except:
                    labelmap.append(row[1])
        if len(labelmap) == 0:
            logging.warning('there are no labels!')
        labelmap = sorted(list(set(labelmap)))
        logging.info('find {} labels'.format(len(labelmap)))
        need_update = False
        if op.isfile(dataset.get_labelmap_file()):
            origin_labelmap = dataset.load_labelmap()
            if len(origin_labelmap) == len(labelmap):
                for o in origin_labelmap:
                    if o not in labelmap:
                        need_update = True
                        break
            else:
                need_update = True
        else:
            need_update = True
        if need_update:
            logging.info('updating {}'.format(dataset.get_labelmap_file()))
            write_to_file('\n'.join(labelmap), dataset.get_labelmap_file())

def populate_dataset_details(data, check_image_details=False,
        splits=None, check_box=False, data_root=None):
    logging.info(data)
    dataset = TSVDataset(data)

    if not splits:
        splits = get_default_splits()

    # populate the height and with
    if check_image_details:
        populate_dataset_hw(data, splits)

    ensure_label_extract(data, splits)

    for split in splits:
        tsv_file = dataset.get_data(split)
        out_file = get_meta_file(tsv_file)
        if not op.isfile(out_file) and \
                dataset.has(split, 'hw') and \
                dataset.has(split):
            row_hw = dataset.iter_data(split, 'hw')
            row_label = dataset.iter_data(split, 'label')
            num_rows = dataset.num_rows(split)
            if check_image_details:
                details = tsv_details(row_hw, row_label, num_rows)
                write_to_yaml_file(details, out_file)

    ensure_labelmap_extract(data)

    labelmap = []
    if not op.isfile(dataset.get_pos_labelmap_file()):
        ls = dataset.load_labelmap()
        write_to_file('\n'.join([l for l in ls if not l.startswith('-')]),
                dataset.get_pos_labelmap_file())

    # generate the rows with duplicate keys
    for split in splits:
        label_tsv = dataset.get_data(split, 'label')
        duplicate_tsv = dataset.get_data(split, 'key_duplicate')
        if op.isfile(label_tsv) and not op.isfile(duplicate_tsv):
            detect_duplicate_key(label_tsv, duplicate_tsv)

    # generate lineidx if it is not generated
    for split in splits:
        lineidx = dataset.get_lineidx(split)
        full_tsv = dataset.get_data(split)
        if not op.isfile(lineidx) and op.isfile(full_tsv):
            logging.info('no lineidx for {}. generating...'.format(split))
            generate_lineidx(full_tsv, lineidx)

    ensure_create_inverted_tsvs(dataset, splits)
    # check if the number of rows from the label tsv are equal to the number of rows in image tsv
    for split in splits:
        v = 0
        num_rows = None
        while True:
            if not dataset.has(split, 'label', v):
                break
            if dataset.has(split, 'check_count', v):
                v = v + 1
                continue
            if num_rows is None:
                num_rows = dataset.num_rows(split)
            num_rows_in_label = dataset.num_rows(split, 'label', v)
            # we can remove the assert
            assert num_rows == num_rows_in_label
            if num_rows != num_rows_in_label:
                dataset.write_data([['num_rows', num_rows], ['num_rows_in_label', num_rows_in_label]],
                        split, 'check_count', v)
            else:
                dataset.write_data([], split, 'check_count', v)
            v = v + 1

    if not op.isfile(dataset.get_noffsets_file()):
        logging.info('no noffset file. generating...')
        labelmap = dataset.load_labelmap()
        mapper = LabelToSynset(True)
        ambigous = []
        ss = [mapper.convert(l) for l in labelmap]
        for l, (success, s) in zip(labelmap, ss):
            if not success and len(s) > 1:
                d = create_info_for_ambigous_noffset(l, [synset_to_noffset(s1)
                    for s1 in s])
                ambigous.append(d)
        if len(ambigous) > 0:
            logging.info('ambigous term which has no exact noffset: {}'.format(
                dataset.name))
            write_to_yaml_file(ambigous, dataset.get_noffsets_file() +
                    '.ambigous.yaml')
        noffsets = []
        for success, s in ss:
            if not success:
                noffsets.append('')
            else:
                noffsets.append(','.join([synset_to_noffset(o) for o in s]))
        write_to_file('\n'.join(noffsets), dataset.get_noffsets_file())

    if not op.isfile(dataset.get_labelmap_of_noffset_file()):
        noffsets = dataset.load_noffsets()
        all_line = []
        for noffset in noffsets:
            if len(noffset) == 0:
                all_line.append('unkown')
            else:
                ss = [noffset_to_synset(n) for n in noffset.split(',')]
                all_line.append(','.join([get_nick_name(s) for s in ss]))
        write_to_file('\n'.join(all_line),
                dataset.get_labelmap_of_noffset_file())

    # generate the label -> count tsv
    for split in splits:
        v = 0
        while True:
            label_idx_file = dataset.get_data(split, 'inverted.label', v)
            label_count_file = dataset.get_data(split, 'inverted.label.count', v)
            if op.isfile(label_idx_file):
                if not op.isfile(label_count_file):
                    label_idx = dataset.load_inverted_label_as_list(split, v)
                    tsv_writer(((l, str(len(i))) for l, i in label_idx), label_count_file)
            else:
                break
            v = v + 1

    if op.isfile(dataset.get_data('trainX')):
        # composite dataset.
        create_index_composite_dataset(dataset)

    populate_num_images_composite(dataset)

    add_node_to_ancestors(dataset)

    populate_all_label_counts(dataset)

    if check_box:
        populate_bbcount(dataset)

def create_label_bbcount(dataset, split, v):
    class_to_bbcount = defaultdict(int)
    logging.info('reading {}, {}, {}'.format(split, 'label', v))
    for key, str_rects in dataset.iter_data(split, 'label',
        v, progress=True):
        rects = json.loads(str_rects)
        for r in rects:
            if 'rect' in r and not all(x == 0 for x in r['rect']):
                c = r['class']
                class_to_bbcount[c] = class_to_bbcount[c] + 1
    labelmap = [l for l, in dataset.iter_data(split, 'labelmap',
            v)]
    dataset.write_data(((l, class_to_bbcount.get(l, 0)) for l in labelmap),
            split, 'label.bbcount', v)

def create_label_verifiedbbcount(dataset, split, v):
    class_to_bbcount = defaultdict(int)
    logging.info('reading {}, {}, {}'.format(split, 'label', v))
    for key, str_rects in dataset.iter_data(split, 'label',
        v, progress=True):
        rects = json.loads(str_rects)
        for r in rects:
            if 'rect' in r and \
                    not all(x == 0 for x in r['rect']) and \
                    is_verified_rect(r):
                c = r['class']
                class_to_bbcount[c] = class_to_bbcount[c] + 1
    labelmap = [l for l, in dataset.iter_data(split, 'labelmap',
            v)]
    dataset.write_data(((l, class_to_bbcount.get(l, 0)) for l in labelmap),
            split, 'label.verifiedbbcount', v)

def populate_bbcount(dataset):
    for split in get_default_splits():
        v = 0
        while True:
            if dataset.has(split, 'label', v):
                if not dataset.has(split, 'label.verifiedbbcount', v):
                    create_label_verifiedbbcount(dataset, split, v)
                if not dataset.has(split, 'label.bbcount', v):
                    create_label_bbcount(dataset, split, v)
                v = v + 1
            else:
                break

def populate_all_label_counts(dataset):
    for split in get_default_splits():
        v = 0
        while True:
            if dataset.has(split, 'inverted.label.count', v):
                if not dataset.has(split, 'inverted.label.count.total', v):
                    count = sum([int(count) for _, count in dataset.iter_data(split,
                        'inverted.label.count', v)])
                    dataset.write_data([(str(count),)],
                            split,
                            'inverted.label.count.total', v)
                v = v + 1
            else:
                break

def add_node_to_ancestors(dataset):
    tree_file = op.join(dataset._data_root, 'root.yaml')
    out_file = op.join(dataset._data_root, 'treenode_to_ancestors.tsv')
    if op.isfile(tree_file) and worth_create(tree_file, out_file):
        tax = Taxonomy(load_from_yaml_file(tree_file))
        tax.update()
        tsv_writer([[name, ','.join(tax.name_to_ancestors_list[name])] for name in
            tax.name_to_ancestors_list], out_file)

def populate_num_images_composite(dataset):
    data_with_bb = dataset.name + '_with_bb'
    data_no_bb = dataset.name + '_no_bb'
    datas = [data_with_bb, data_no_bb]
    datasets = [TSVDataset(d) for d in datas]
    suffixes = ['with_bb', 'no_bb']
    splits = ['train', 'test']
    dest_tree_file = op.join(dataset._data_root, 'root_enriched.yaml')
    src_tree_file = op.join(dataset._data_root, 'root.yaml')
    if op.isfile(dest_tree_file) or not op.isfile(src_tree_file):
        return
    tax = Taxonomy(load_from_yaml_file(src_tree_file))
    for split in splits:
        for suffix, d in zip(suffixes, datasets):
            if not d.has(split, 'inverted.label.count'):
                continue
            label_to_count_with_bb = {label: int(count) for label, count in d.iter_data(split,
                    'inverted.label.count')}
            for label in label_to_count_with_bb:
                count = label_to_count_with_bb[label]
                if not label.startswith('-'):
                    nodes = tax.root.search_nodes(name=label)
                else:
                    nodes = tax.root.search_nodes(name=label[1:])
                assert len(nodes) == 1, label
                node = nodes[0]
                if not label.startswith('-'):
                    node.add_feature('{}_{}'.format(suffix, split), count)
                else:
                    node.add_feature('{}_{}_neg'.format(suffix, split), count)
    write_to_yaml_file(tax.dump(), dest_tree_file)

def create_index_composite_dataset(dataset):
    split = 'train'
    splitx = split + 'X'
    fname_numImagesPerSource = dataset.get_data(splitx,
            'numImagesPerSource')
    if op.isfile(fname_numImagesPerSource):
        return
    # how many images are contributed from each data source
    trainX_file = dataset.get_data(splitx)
    source_tsvs = load_list_file(trainX_file)

    shuffle_file = dataset.get_shuffle_file(split)
    if not op.isfile(shuffle_file):
        return
    rows = tsv_reader(shuffle_file)
    all_idxSource_idxRow = []
    for idx_source, idx_row in rows:
        all_idxSource_idxRow.append((int(idx_source), int(idx_row)))
    # note, the data may be duplicated.
    all_idxSource_idxRow = list(set(all_idxSource_idxRow))

    idxSource_to_idxRows = list_to_dict(all_idxSource_idxRow, 0)
    num_images_per_datasource = [None] * len(source_tsvs)
    num_srcimages_per_datasource = [TSVFile(s).num_rows() for s in source_tsvs]
    for idxSource in idxSource_to_idxRows:
        assert num_images_per_datasource[idxSource] is None
        num_images_per_datasource[idxSource] = len(idxSource_to_idxRows[idxSource])
    tsv_writer([(name, str(num)) for name, num in zip(source_tsvs,
        num_images_per_datasource)], fname_numImagesPerSource)
    dataset.write_data([(source_tsvs[i], num_images_per_datasource[i], num_srcimages_per_datasource[i])
        for i in range(len(source_tsvs))], splitx,
        'tsvsource.numdestimages.numsrcimages')

    # for each data source, how many labels are contributed and how many are
    # not
    if op.isfile(dataset.get_data(splitx, 'origin.label')):
        source_tsv_label_files = load_list_file(dataset.get_data(splitx,
            'origin.label'))
        source_tsv_labels = [TSVFile(t) for t in source_tsv_label_files]
        trainX_label_file = dataset.get_data(splitx, 'label')
        all_dest_label_file = load_list_file(trainX_label_file)
        dest_labels = [TSVFile(f) for f in all_dest_label_file]
        all_idxSource_sourceLabel_destLabel = []
        logging.info('each datasource and each idx row')
        idxSource_to_numRect = {}
        all_idxSource_numSourceRects_numDestRects = []
        for idx_source, idx_row in tqdm(all_idxSource_idxRow):
            source_rects = json.loads(source_tsv_labels[idx_source].seek(idx_row)[-1])
            dest_rects = json.loads(dest_labels[idx_source].seek(idx_row)[-1])
            if idx_source not in idxSource_to_numRect:
                idxSource_to_numRect[idx_source] = 0
            idxSource_to_numRect[idx_source] = idxSource_to_numRect[idx_source] + \
                len(dest_rects)
            all_idxSource_numSourceRects_numDestRects.append((
                idx_source, len(source_rects), len(dest_rects)))
            for r in dest_rects:
                all_idxSource_sourceLabel_destLabel.append((idx_source,
                    r.get('class_from', r['class']), r['class']))
        idxSource_to_numSourceRects_numDestRects = list_to_dict(
                all_idxSource_numSourceRects_numDestRects, 0)
        idxSource_to_numSourceRects_numDestRect = {idxSource: [sum(x1 for x1, x2 in idxSource_to_numSourceRects_numDestRects[idxSource]),
                 sum(x2 for x1, x2 in idxSource_to_numSourceRects_numDestRects[idxSource])]
            for idxSource in idxSource_to_numSourceRects_numDestRects}

        dataset.write_data([(source_tsvs[idxSource],
            idxSource_to_numSourceRects_numDestRect[idxSource][1],
            idxSource_to_numSourceRects_numDestRect[idxSource][0])
            for idxSource in range(len(source_tsvs))],
            splitx, 'tsvsource.numdestbbs.numsrcbbs')

        sourcetsv_to_num_rect = {source_tsvs[idx_source]: idxSource_to_numRect[idx_source]
                for idx_source in idxSource_to_numRect}
        dataset.write_data([(s, sourcetsv_to_num_rect[s]) for s in source_tsvs],
            splitx, 'numRectsPerSource')

        idxSource_to_sourceLabel_destLabels = list_to_dict(
                all_idxSource_sourceLabel_destLabel, 0)
        source_numSourceLabels = [(s, 0) for s in source_tsvs]
        source_includedSourceLabels = [(s, []) for s in source_tsvs]
        for idxSource in idxSource_to_sourceLabel_destLabels:
            sourceLabel_destLabels = idxSource_to_sourceLabel_destLabels[idxSource]
            sourceLabel_to_destLabels = list_to_dict(sourceLabel_destLabels, 0)
            source_numSourceLabels[idxSource] = (source_tsvs[idxSource],
                    len(sourceLabel_to_destLabels))
            source_includedSourceLabels[idxSource][1].extend(
                    sourceLabel_to_destLabels.keys())

        dataset.write_data([(n, str(i)) for (n, i) in source_numSourceLabels],
                splitx, 'numCategoriesPerSource')

        # save teh list of included labels
        sourceDataset_to_includedSourceLabels = {}
        for source, sourceLabels in source_includedSourceLabels:
            source_dataset_name = op.basename(op.dirname(source))
            if source_dataset_name not in sourceDataset_to_includedSourceLabels:
                sourceDataset_to_includedSourceLabels[source_dataset_name] = []
            sourceDataset_to_includedSourceLabels[source_dataset_name].extend(sourceLabels)
        for source_dataset_name in sourceDataset_to_includedSourceLabels:
            sourceDataset_to_includedSourceLabels[source_dataset_name] = \
                    set(sourceDataset_to_includedSourceLabels[source_dataset_name])

        tsv_writer([(n, ','.join(sourceDataset_to_includedSourceLabels[n])) for n in
            sourceDataset_to_includedSourceLabels], op.join(dataset._data_root,
                'trainX.includeCategoriesPerSourceDataset.tsv'))

        tsv_writer([(n, get_nick_name(noffset_to_synset(s)) if is_noffset(s) else
            s) for n in sourceDataset_to_includedSourceLabels
              for s in sourceDataset_to_includedSourceLabels[n]],
              op.join(dataset._data_root, 'trainX.includeCategoriesPerSourceDatasetReadable.tsv'))

class TSVTransformer(object):
    def __init__(self):
        self._total_rows = 0
        self._row_processor = None

    def ReadProcess(self, source_tsv, row_processor):
        self._row_processor = row_processor
        self._total_rows = 0

        rows = tsv_reader(source_tsv)
        x = [self._over_row_processor(row) for row in rows]

        logging.info('total rows = {}; total_processed = {}'.format(self._total_rows,
                len(x)))

    def Process(self, source_tsv, dst_tsv, row_processor):
        '''
        row_processor: a function whose input should be a list of tsv cols and
                      whose return is also a list of tsv colums (will be saved into dst_tsv)
        '''
        self._row_processor = row_processor
        self._total_rows = 0

        rows = tsv_reader(source_tsv)
        result = (self._over_row_processor(row) for row in rows)
        tsv_writer(result, dst_tsv)

        logging.info('total rows = {}'.format(self._total_rows))

    def _over_row_processor(self, row):
        out = self._row_processor(row)
        self._total_rows = self._total_rows + 1
        if self._total_rows % 500 == 0:
            logging.info('processed = {}'.format(self._total_rows))
        return out

def randomize_tsv_file(tsv_file):
    prefix = os.path.splitext(tsv_file)[0]
    shuffle_file = prefix + '.shuffle'
    if os.path.exists(shuffle_file):
        return shuffle_file
    idx_file = prefix + '.lineidx'
    with open(idx_file, 'r') as fp:
        num = len([line for line in fp.readlines() if len(line.strip()) > 0])
    np.random.seed(777)
    nums = np.random.permutation(num)
    result = '\n'.join(map(str, nums))
    with open(shuffle_file, 'w') as fp:
        fp.write(result)
    return shuffle_file

def gen_tsv_from_labeling(input_folder, output_folder):
    fs = glob.glob(op.join(input_folder, '*'))
    labels = set()
    def gen_rows():
        for f in fs:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            yaml_file = op.splitext(f)[0] + '.yaml'
            if not op.isfile(yaml_file):
                logging.info('{} not exist'.format(yaml_file))
            with open(yaml_file, 'r') as fp:
                bb_labels = yaml.loads(fp.read())
            for bb_label in bb_labels:
                labels.add(bb_label['class'])
            with open(f, 'r') as fp:
                encoded_im = base64.b64encode(fp.read())
            yield op.basename(f), json.dumps(bb_labels), encoded_im

    tsv_writer(gen_rows(), op.join(output_folder, 'train.tsv'))
    write_to_file('\n'.join(labels), op.join(output_folder, 'labelmap.txt'))

def try_json_parse(s):
    try:
        return json.loads(s)
    except ValueError:
        return s

def get_label_type(full_expid, predict_file):
    label_type = 'label'
    for row in tsv_reader(
        op.join('output', full_expid, 'snapshot', predict_file)):
        rects = json.loads(row[-1])
        if len(rects) > 0:
            if 'caption' in rects[0]:
                label_type = 'caption'
                break
    return label_type

def iter_full_expid_pred(
    full_expid, predict_file, start_id, threshold,
        label_type, label_version=None):
    test_data, test_data_split = parse_test_data(predict_file)
    pred_tsv = TSVFile(op.join('output', full_expid, 'snapshot', predict_file))
    num_total = len(pred_tsv)
    while start_id < 0:
        start_id += num_total
    test_dataset = TSVDataset(test_data)
    iter_image = test_dataset.iter_data(
        test_data_split,
        filter_idx=range(start_id, num_total))
    has_gt = test_dataset.has(test_data_split, t=label_type, version=label_version)
    if has_gt:
        iter_gt = test_dataset.iter_data(test_data_split, t=label_type,
                                     version=label_version,
                                     filter_idx=range(start_id, num_total))
    for i, row_image in enumerate(iter_image):
        key = row_image[0]
        if has_gt:
            row_gt = next(iter_gt)
            str_gt = row_gt[-1]
            key = row_gt[0]
            rects_gt = json.loads(str_gt)
            if not isinstance(rects_gt, list):
                rects_gt = [{'class': str(rects_gt)}]
        else:
            rects_gt = []
        str_im = row_image[-1]
        pred_key, str_pred = pred_tsv[i + start_id]
        assert pred_key == key, (i,pred_key,key)
        im = img_from_base64(str_im)
        rects_pred = [r for r in json.loads(str_pred) if r['conf'] > threshold]
        yield {
            'key': key,
            'im': im,
            'rects_gt': rects_gt,
            'rects_pred': rects_pred,
        }

def visualize_predict_no_draw(full_expid, predict_file, label, start_id,
        threshold):
    test_data, test_data_split = parse_test_data(predict_file)
    logging.info('{}->{}/{}'.format(predict_file, test_data, test_data_split))
    pred_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    pred_key_path = '{}.key.tsv'.format(pred_full_path)
    pred_label_path = '{}.labelmap.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_inverted_path = '{}.inverted.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_sorted_cache_path = '{}.key_idxGT_idxPred_ap.th{}.{}.tsv'.format(
            pred_full_path, threshold, label)

    test_dataset = TSVDataset(test_data)
    if not op.isfile(pred_sorted_cache_path):
        if not op.isfile(pred_key_path) or \
                not op.isfile(pred_label_path) or \
                not op.isfile(pred_inverted_path):
            logging.info('loading {}'.format(pred_full_path))
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(pred_full_path), threshold)
            pred_labels = inverted.keys()
            logging.info('writing {}'.format(pred_key_path))
            tsv_writer([[k] for k in pred_keys], pred_key_path)
            tsv_writer(([l] for l in pred_labels), pred_label_path)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_path)
        keys_from_pred = []
        labelmap = load_list_file(pred_label_path)
        pred_keys = load_list_file(pred_key_path)
        if label in labelmap:
            inverted_row = TSVFile(pred_inverted_path).seek(labelmap.index(label))
            assert inverted_row[0] == label
            assert len(inverted_row) == 2
            idx_from_pred = map(int, inverted_row[1].split(' '))
            keys_from_pred = [pred_keys[i] for i in idx_from_pred]
        else:
            keys_from_pred = []
        inverted_test_split = test_dataset.load_inverted_label(test_data_split, version=-1,
                label=label)
        if label in inverted_test_split:
            idx_from_gt = inverted_test_split[label]
        else:
            idx_from_gt = []
        rows = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=idx_from_gt, unique=True)
        keys_from_gt = [row[0] for row in rows]
        target_keys = list(set(keys_from_pred + keys_from_gt))
        target_keys = [k for k in target_keys if k in pred_keys]
        target_idx_in_pred = [pred_keys.index(k) for k in target_keys]
        gt_keys = test_dataset.load_keys(test_data_split)
        target_idx_in_gt = [gt_keys.index(k) for k in target_keys]
        rows_in_gt = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=target_idx_in_gt)
        pred_tsv = TSVFile(pred_full_path)
        rows_in_pred = (pred_tsv.seek(i) for i in target_idx_in_pred)
        target_aps = []
        for row_in_gt, row_in_pred in zip(rows_in_gt, rows_in_pred):
            assert row_in_gt[0] == row_in_pred[0]
            assert len(row_in_gt) == 2
            assert len(row_in_pred) == 2
            rects_gt = json.loads(row_in_gt[1])
            rects_pred = json.loads(row_in_pred[1])
            rects_gt = [r for r in rects_gt if r['class'] == label]
            rects_pred = [r for r in rects_pred if r['class'] == label]
            ap = calculate_image_ap([r['rect'] for r in rects_gt if 'rect' in r],
                    [r['rect'] for r in rects_pred])
            target_aps.append(ap)
        key_idxGT_idxPred_aps = zip(target_keys, target_idx_in_gt,
                target_idx_in_pred, target_aps)
        key_idxGT_idxPred_aps = sorted(key_idxGT_idxPred_aps, key=lambda x:
                x[-1])
        tsv_writer(key_idxGT_idxPred_aps, pred_sorted_cache_path)

    tsv = TSVFile(pred_sorted_cache_path)
    total_num = tsv.num_rows()
    if total_num == 0:
        return
    while start_id < 0:
        start_id = start_id + total_num
    while start_id >= total_num:
        start_id = start_id - total_num
    i = start_id
    tsv_pred = TSVFile(pred_full_path)
    for i in range(start_id, total_num):
        key, idx_gt, idx_pred, ap = tsv.seek(i)
        idx_gt, idx_pred, ap = int(idx_gt), int(idx_pred), float(ap)
        row_gt = next(test_dataset.iter_data(test_data_split,
            filter_idx=[idx_gt]))
        row_pred = tsv_pred.seek(idx_pred)
        assert row_gt[0] == row_pred[0], (row_gt[0], row_pred[0])

        rects_gt = json.loads(next(test_dataset.iter_data(test_data_split, 'label',
            filter_idx=[idx_gt], version=-1))[1])

        rects_pred = json.loads(row_pred[1])
        rects_pred = [r for r in rects_pred if r['conf'] > threshold]
        im_origin = img_from_base64(row_gt[-1])
        yield key, im_origin, rects_gt, rects_pred, ap

def visualize_predict(full_expid, predict_file, label, start_id, threshold):
    test_data, test_data_split = parse_test_data(predict_file)
    pred_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    pred_key_path = '{}.key.tsv'.format(pred_full_path)
    pred_label_path = '{}.labelmap.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_inverted_path = '{}.inverted.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_sorted_cache_path = '{}.key_idxGT_idxPred_ap.th{}.{}.tsv'.format(
            pred_full_path, threshold, label)

    test_dataset = TSVDataset(test_data)
    if not op.isfile(pred_sorted_cache_path):
        if not op.isfile(pred_key_path) or \
                not op.isfile(pred_label_path) or \
                not op.isfile(pred_inverted_path):
            logging.info('loading {}'.format(pred_full_path))
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(pred_full_path), threshold)
            pred_labels = inverted.keys()
            logging.info('writing {}'.format(pred_key_path))
            tsv_writer([[k] for k in pred_keys], pred_key_path)
            tsv_writer(([l] for l in pred_labels), pred_label_path)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_path)
        keys_from_pred = []
        labelmap = load_list_file(pred_label_path)
        pred_keys = load_list_file(pred_key_path)
        if label in labelmap:
            inverted_row = TSVFile(pred_inverted_path).seek(labelmap.index(label))
            assert inverted_row[0] == label
            assert len(inverted_row) == 2
            idx_from_pred = map(int, inverted_row[1].split(' '))
            keys_from_pred = [pred_keys[i] for i in idx_from_pred]
        else:
            keys_from_pred = []
        inverted_test_split = test_dataset.load_inverted_label(test_data_split, version=-1,
                label=label)
        if label in inverted_test_split:
            idx_from_gt = inverted_test_split[label]
        else:
            idx_from_gt = []
        rows = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=idx_from_gt, unique=True)
        keys_from_gt = [row[0] for row in rows]
        target_keys = list(set(keys_from_pred + keys_from_gt))
        target_keys = [k for k in target_keys if k in pred_keys]
        target_idx_in_pred = [pred_keys.index(k) for k in target_keys]
        gt_keys = test_dataset.load_keys(test_data_split)
        target_idx_in_gt = [gt_keys.index(k) for k in target_keys]
        rows_in_gt = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=target_idx_in_gt)
        pred_tsv = TSVFile(pred_full_path)
        rows_in_pred = (pred_tsv.seek(i) for i in target_idx_in_pred)
        target_aps = []
        for row_in_gt, row_in_pred in zip(rows_in_gt, rows_in_pred):
            assert row_in_gt[0] == row_in_pred[0]
            assert len(row_in_gt) == 2
            assert len(row_in_pred) == 2
            rects_gt = json.loads(row_in_gt[1])
            rects_pred = json.loads(row_in_pred[1])
            rects_gt = [r for r in rects_gt if r['class'] == label]
            rects_pred = [r for r in rects_pred if r['class'] == label]
            ap = calculate_image_ap([r['rect'] for r in rects_gt],
                    [r['rect'] for r in rects_pred])
            target_aps.append(ap)
        key_idxGT_idxPred_aps = zip(target_keys, target_idx_in_gt,
                target_idx_in_pred, target_aps)
        key_idxGT_idxPred_aps = sorted(key_idxGT_idxPred_aps, key=lambda x:
                x[-1])
        tsv_writer(key_idxGT_idxPred_aps, pred_sorted_cache_path)

    tsv = TSVFile(pred_sorted_cache_path)
    total_num = tsv.num_rows()
    if total_num == 0:
        return
    while start_id < 0:
        start_id = start_id + total_num
    while start_id >= total_num:
        start_id = start_id - total_num
    i = start_id
    tsv_pred = TSVFile(pred_full_path)
    for i in range(start_id, total_num):
        key, idx_gt, idx_pred, ap = tsv.seek(i)
        idx_gt, idx_pred, ap = int(idx_gt), int(idx_pred), float(ap)
        row_gt = next(test_dataset.iter_data(test_data_split,
            filter_idx=[idx_gt]))
        row_pred = tsv_pred.seek(idx_pred)
        assert row_gt[0] == row_pred[0], (row_gt[0], row_pred[0])

        rects_gt = json.loads(row_gt[1])
        rects_pred = json.loads(row_pred[1])
        rects_pred = [r for r in rects_pred if r['conf'] > threshold]
        rects_gt_target = [r for r in rects_gt if r['class'] == label]
        rects_pred_target = [r for r in rects_pred if r['class'] == label]
        if len(rects_gt_target) == 0 and len(rects_pred_target) == 0:
            logging.info('skipping to next')
            continue
        im_origin = img_from_base64(row_gt[-1])
        im_gt_target = np.copy(im_origin)
        draw_bb(im_gt_target, [r['rect'] for r in rects_gt_target],
                [r['class'] for r in rects_gt_target])
        im_pred_target = np.copy(im_origin)
        draw_bb(im_pred_target, [r['rect'] for r in rects_pred_target],
                [r['class'] for r in rects_pred_target])
        im_gt = np.copy(im_origin)
        draw_bb(im_gt, [r['rect'] for r in rects_gt],
                [r['class'] for r in rects_gt])
        im_pred = np.copy(im_origin)
        draw_bb(im_pred, [r['rect'] for r in rects_pred],
                [r['class'] for r in rects_pred])
        yield key, im_origin, im_gt_target, im_pred_target, im_gt, im_pred, ap

def visualize_box_no_draw(data, split, version, label, start_id, color_map={},
                          split_type='label',
        min_conf=-1, max_conf=2):
    dataset = TSVDataset(data)
    if split is None:
        # guess which split should be used. only support non-composite tsv
        candidate_split = get_default_splits()
        for c in candidate_split:
            if not op.isfile(dataset.get_data(c)):
                continue
            inverted = dataset.load_inverted_label(c, version, label)
            if label not in inverted:
                continue
            n = len(inverted[label])
            if n <= start_id:
                start_id = start_id - n
            else:
                logging.info('split = {}'.format(split))
                split = c
                break
        if not split:
            logging.info('cannot find the valid')
            return
    if label is not None:
        logging.info('loading inverted label')
        idx = dataset.load_inverted_label(split, version, label)[label]
        curr_num_image = len(idx)
    else:
        curr_num_image = dataset.num_rows(split, t=split_type, version=version)
    if curr_num_image == 0:
        logging.info('there is no image')
        return
    while start_id > curr_num_image:
        start_id = start_id - curr_num_image
    while start_id < 0:
        start_id = start_id + curr_num_image
    if label is not None:
        filter_idx = idx[start_id:]
    else:
        filter_idx = range(start_id, curr_num_image)
    logging.info('start to read')
    rows_image = dataset.iter_data(split, filter_idx=filter_idx)
    rows_label = dataset.iter_data(split, split_type, version=version,
            filter_idx=filter_idx)
    for idx_row, (row_image, row_label) in enumerate(zip(rows_image, rows_label)):
        key = row_image[0]
        assert key == row_label[0], (key, row_label[0], idx_row)
        assert len(row_label) == 2
        label_str = row_label[-1]
        img_str = row_image[-1]
        im = img_from_base64(img_str)
        assert im is not None
        origin = np.copy(im)
        rects = try_json_parse(label_str)
        if type(rects) is list:
            if label is not None:
                rects = [r for r in rects if \
                        r['class'] == label and \
                        r.get('conf', 1) >= min_conf and \
                        r.get('conf', 1) <= max_conf or r['class'] != label]
                if all(r['class'] != label for r in rects):
                    continue
            yield (key, origin, rects)
        else:
            yield key, origin, [{'class': label_str, 'rect': [0, 0, 0, 0]}]

def visualize_box(data, split, version, label, start_id, color_map={}):
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    if split is None:
        # guess which split should be used. only support non-composite tsv
        candidate_split = get_default_splits()
        for c in candidate_split:
            if not op.isfile(dataset.get_data(c)):
                continue
            inverted = dataset.load_inverted_label(c, version, label)
            if label not in inverted:
                continue
            n = len(inverted[label])
            if n <= start_id:
                start_id = start_id - n
            else:
                logging.info('split = {}'.format(split))
                split = c
                break
        if not split:
            logging.info('cannot find the valid')
            return
    else:
        inverted = dataset.load_inverted_label(split, version, label)
    logging.info('inverted label loaded')
    logging.info('keys: {}'.format(inverted.keys()))
    if label not in inverted:
        return
    idx = inverted[label]
    if len(idx) == 0:
        return
    while start_id > len(idx):
        start_id = start_id - len(idx)
    while start_id < 0:
        start_id = start_id + len(idx)
    logging.info('start to read')
    rows_image = dataset.iter_data(split, filter_idx=idx[start_id:])
    rows_label = dataset.iter_data(split, 'label', version=version,
            filter_idx=idx[start_id:])
    for row_image, row_label in zip(rows_image, rows_label):
        key = row_image[0]
        assert key == row_label[0]
        assert len(row_image) == 3
        assert len(row_label) == 2
        label_str = row_label[-1]
        img_str = row_image[-1]
        im = img_from_base64(img_str)
        origin = np.copy(im)
        rects = try_json_parse(label_str)
        new_name = key.replace('/', '_').replace(':', '')
        if type(rects) is list:
            #rects = [l for l in rects if 'conf' not in l or l['conf'] > 0.3]
            def get_rect_class(rects):
                all_class = []
                all_rect = []
                for rect in rects:
                    label_class = rect['class']
                    rect = rect['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                return all_rect, all_class
            all_rect, all_class = get_rect_class(rects)
            draw_bb(im, all_rect, all_class)
            target_rects = [l for l in rects if l['class']
                    == label]
            all_rect, all_class = get_rect_class(target_rects)
            im_label = np.copy(origin)
            draw_bb(im_label, all_rect, all_class)
            yield new_name, origin, im_label, im, pformat(target_rects)
        else:
            yield new_name, origin, im, im, ''


def visualize_tsv2(data, split, label):
    '''
    by default, pass split as 'train'
    TODO: try to refactor it with visualize_box
    '''
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    inverted = dataset.load_inverted_label(split)
    logging.info('inverted label loaded')
    logging.info('keys: {}'.format(inverted.keys()))
    assert label in inverted
    idx = inverted[label]
    is_composite = False
    if split == 'train' and not op.isfile(dataset.get_data(split)):
        is_composite = True
        tsvs = [TSVFile(f) for f in dataset.get_train_tsvs()]
        shuffle_tsv_rows = tsv_reader(dataset.get_shuffle_file(split))
        shuffle = []
        for row in shuffle_tsv_rows:
            shuffle.append([int(row[0]), int(row[1])])
    else:
        tsv = TSVFile(dataset.get_data(split))
    num_image = 0
    num_rows = 2
    num_cols = 2
    num_image_one_fig = num_rows * num_cols
    idx.extend([0] * (num_image_one_fig - len(idx) % num_image_one_fig))
    idx = np.asarray(idx)
    idx = idx.reshape((-1, num_image_one_fig))
    logging.info('start to read')
    color_map = {}
    for i in idx:
        all_image = []
        for j in i:
            logging.info(j)
            if is_composite:
                row_image = tsvs[shuffle[j][0]].seek(shuffle[j][1])
            else:
                row_image = tsv.seek(j)
            im = img_from_base64(row_image[-1])
            labels = try_json_parse(row_image[1])
            num_image = num_image + 1
            if type(labels) is list:
                labels = [l for l in labels if 'conf' not in l or l['conf'] > 0.3]
                all_class = []
                all_rect = []
                for label in labels:
                    label_class = label['class']
                    rect = label['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                new_name = row_image[0].replace('/', '_').replace(':', '')
                draw_bb(im, all_rect, all_class, color=color_map)
            all_image.append(im)
        logging.info('start to show')
        show_images(all_image, num_rows, num_cols)

    logging.info('#image: {}'.format(num_image))

def visualize_tsv(tsv_image, tsv_label, out_folder=None, label_idx=1):
    '''
    deprecated, use visualize_tsv2
    '''
    rows_image = tsv_reader(tsv_image)
    rows_label = tsv_reader(tsv_label)
    assert out_folder == None or not op.exists(out_folder)
    source_folder = op.dirname(tsv_image)
    num_image = 0
    for row_image, row_label in zip(rows_image, rows_label):
        assert row_image[0] == row_label[0]
        im = img_from_base64(row_image[-1])
        labels = try_json_parse(row_label[label_idx])
        num_image = num_image + 1
        if type(labels) is list:
            labels = [l for l in labels if 'conf' not in l or l['conf'] > 0.3]
            all_class = []
            all_rect = []
            for label in labels:
                label_class = label['class']
                rect = label['rect']
                all_class.append(label_class)
                if not (rect[0] == 0 and rect[1] == 0
                        and rect[2] == 0 and rect[3] == 0):
                    all_rect.append(rect)
                else:
                    all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
            new_name = row_image[0].replace('/', '_').replace(':', '')
            draw_bb(im, all_rect, all_class)
            if out_folder:
                fname = os.path.join(out_folder,
                        '_'.join(set(c.replace(' ', '_') for c in all_class)),
                        new_name +'.png')
        else:
            fname = op.join(out_folder, row_image[0],
                    '{}_{}_{}.png'.format(num_image, labels, row_image[0]))
        if out_folder:
            save_image(im, fname)
        else:
            show_image(im)

    logging.info('#image: {}'.format(num_image))

def iou(wh1, wh2):
    w1, h1 = wh1
    w2, h2 = wh2
    return min(w1, w2) * min(h1, h2) / max(w1, w2) / max(h1, h2)

class ImageTypeParser(object):
    def __init__(self):
        self.m = None
        pass

    def parse_type(self, im_binary):
        return imghdr.what('', im_binary)
        self._ensure_init()
        mime_type = self.m.buffer(im_binary)
        t = op.basename(mime_type)
        return t

    def _ensure_init(self):
        if self.m is None:
            import magic
            #m = magic.open(magic.MAGIC_MIME_TYPE)
            m = magic.from_file(magic.MAGIC_MIME_TYPE)
            m.load()
            self.m = m

def collect_label(row, stat, **kwargs):
    labels = json.loads(row[1])
    labels = [label['class'] for label in labels]
    remove_labels = kwargs['remove_image'].split(',')
    is_remove = False
    for label in labels:
        if label in remove_labels or remove_labels == 'all':
            if random.random() <= kwargs['remove_image_prob']:
                is_remove = True
                break

    stat.append((is_remove, labels))

class TSVDatasetSource(TSVDataset):
    def __init__(self, name, root=None,
            split_infos=None,
            cleaness=10,
            use_all=False,
            use_negative_label=False,
            select_by_verified=False):
        super(TSVDatasetSource, self).__init__(name)
        self._noffset_count = {}
        self._type = None
        self._root = root
        # the list of <datasetlabel, rootlabel>
        self._sourcelabel_to_targetlabels = None
        self._targetlabel_to_sourcelabels = None
        self._initialized = False
        self._type_to_datasetlabel_to_split_idx = None
        self._type_to_datasetlabel_to_count = None
        self._type_to_split_label_idx = None
        if split_infos is not None:
            self._split_infos = split_infos
        else:
            self._split_infos = [{'split': s, 'version': -1}
                    for s in get_default_splits()]
        assert len(set([split_info['split'] for split_info in
            self._split_infos])) == len(self._split_infos)
        self._use_all = use_all
        self._select_by_verified = select_by_verified
        self.cleaness = cleaness
        self.use_negative_label = use_negative_label

    def get_version_by_split(self, split_name):
        for split_info in self._split_infos:
            if split_info['split'] == split_name:
                return split_info['version']
        return -1

    def get_label_tsv(self, split_name):
        version_by_config = self.get_version_by_split(split_name)
        return super(TSVDatasetSource, self).get_data(split_name, 'label',
                version_by_config)

    def populate_info(self, root):
        self._ensure_initialized()
        for node in root.iter_search_nodes():
            if root == node:
                continue
            if node.name in self._targetlabel_to_sourcelabels:
                sourcelabels = self._targetlabel_to_sourcelabels[node.name]
                node.add_feature(self.name, ','.join(sourcelabels))
                if any(is_noffset(l) for l in sourcelabels):
                    node.add_feature(self.name + '_readable',
                            ','.join(get_nick_name(noffset_to_synset(l)) if
                                is_noffset(l) else l for l in sourcelabels))
                total = 0
                for sourcelabel in sourcelabels:
                    for t in self._type_to_datasetlabel_to_count:
                        datasetlabel_to_count = self._type_to_datasetlabel_to_count[t]
                        c = datasetlabel_to_count.get(sourcelabel, 0)
                        total = total + c
                        key = 'images_{}'.format(t)
                        node.add_feature(key,
                                node.__getattribute__(key) + c)
                node.add_feature('{}_total'.format(self.name), total)

    def _ensure_initialized(self):
        if self._initialized:
            return
        populate_dataset_details(self.name)

        self.update_label_mapper()

        self._load_inverted()

        self._initialized = True

    def _load_inverted(self):
        # make sure self.update_label_mapper() is called
        types = ['with_bb', 'no_bb']
        self._type_split_label_idx = []
        for split_info in self._split_infos:
            split = split_info['split']
            version = split_info['version']
            logging.info('loading the inverted file: {}-{}'.format(self.name,
                split))
            if not self.has(split, 'label', version=version):
                continue
            # load inverted list
            type_to_inverted = {}
            for i, t in enumerate(types):
                if self._select_by_verified:
                    inverted_label_type = 'inverted.label.{}.verified'.format(t)
                else:
                    inverted_label_type = 'inverted.label.{}'.format(t)
                rows = self.iter_data(split, inverted_label_type,
                        version=version)
                type_to_inverted[t] = {r[0]: map(int, r[1].split(' '))
                        for r in tqdm(rows) if
                    r[0] in self._sourcelabel_to_targetlabels and
                    len(r[1]) > 0}

            # register the positive labels
            for i, t in enumerate(types):
                inverted = type_to_inverted[t]
                inverted = {l: inverted[l] for l in inverted if not l.startswith('-')}
                label_idx = dict_to_list(inverted, 0)
                for label, idx in label_idx:
                    self._type_split_label_idx.append((t, split, label, idx))
                    # for no_bb, we need to add the with_bb into the list
                    if t == 'with_bb':
                        self._type_split_label_idx.append(('no_bb', split, label, idx))

            if self.use_negative_label:
                # currently, we only have a scenario where we have negative
                # annotations in the no_bb data source and the need to apply it
                # to no_bb target set.
                inverted = type_to_inverted['no_bb']
                inverted = {l: inverted[l] for l in inverted if l.startswith('-')}
                label_idx = dict_to_list(inverted, 0)
                for label, idx in label_idx:
                    #self._type_split_label_idx.append(('with_bb', split, label, idx))
                    self._type_split_label_idx.append(('no_bb', split, label, idx))

        self._type_split_label_idx = list(set(self._type_split_label_idx))
        self._type_to_split_label_idx = list_to_dict(
                self._type_split_label_idx, 0)
        self._type_to_datasetlabel_to_split_idx = {}
        for t in self._type_to_split_label_idx:
            split_label_idx = self._type_to_split_label_idx[t]
            label_to_split_idx = list_to_dict(split_label_idx, 1)
            self._type_to_datasetlabel_to_split_idx[t] = label_to_split_idx
        self._type_to_datasetlabel_to_count = {}
        for t in self._type_to_datasetlabel_to_split_idx:
            datasetlabel_to_split_idx = self._type_to_datasetlabel_to_split_idx[t]
            datasetlabel_to_count = {l: len(datasetlabel_to_split_idx[l]) for l in
                    datasetlabel_to_split_idx}
            self._type_to_datasetlabel_to_count[t] = datasetlabel_to_count

        self._split_to_num_image = {split_info['split']: self.num_rows(split_info['split']) for split_info in
                self._split_infos if self.has(split_info['split'])}

    def update_label_mapper(self):
        root = self._root
        # load the labelmap for all splits, self.load_labelmap is not correct,
        # since we will update the label and will not update the labelmap
        labelmap = []
        for split_info in self._split_infos:
            split, version = split_info['split'], split_info['version']
            if self.has(split, 'labelmap', version):
                for row in self.iter_data(split, 'labelmap', version):
                    labelmap.append(row[0])
        # if it has a prefix of -, it means it has no that tag.
        labelmap = [l for l in labelmap if not l.startswith('-')]
        hash_labelmap = set(labelmap)
        labelmap = list(hash_labelmap)

        tree_noffsets = {}
        for node in root.iter_search_nodes():
            if node == root or not node.noffset:
                continue
            for s in node.noffset.split(','):
                tree_noffsets[s] = node.name
        name_to_targetlabels = {}
        targetlabel_has_whitelist = set()
        invalid_list = []
        any_source_key = 'label_names_in_all_dataset_source'
        for node in root.iter_search_nodes():
            if node == root:
                continue
            if hasattr(node, self.name) or hasattr(node, any_source_key):
                if hasattr(node, self.name):
                    # this is like a white-list
                    values = node.__getattribute__(self.name)
                else:
                    values = node.__getattribute__(any_source_key)
                if values is not None:
                    source_terms = values.split(',')
                    for t in source_terms:
                        t = t.strip()
                        if t not in name_to_targetlabels:
                            name_to_targetlabels[t] = set()
                        if t not in hash_labelmap:
                            invalid_list.append((t, self.name, node.name))
                            continue
                        name_to_targetlabels[t].add(node.name)
                # even if it is None, we will also add it to white-list so that
                # we will not automatically match the term.
                targetlabel_has_whitelist.add(node.name)
            else:
                # we will keep the lower case always for case-insensitive
                # comparison
                all_candidate_src_names = [node.name.lower()]
                if hasattr(node, 'alias_names'):
                    all_candidate_src_names.extend([s.strip() for s in
                        node.alias_names.split(',')])
                for t in set(all_candidate_src_names):
                    if t not in name_to_targetlabels:
                        name_to_targetlabels[t] = set()
                    name_to_targetlabels[t].add(node.name)

        sourcelabel_targetlabel = []
        if len(invalid_list) != 0:
            logging.warn('invalid white list information: {}'.format(pformat(invalid_list)))

        #result = {}
        label_to_synset = LabelToSynset()
        for l in labelmap:
            matched = False
            if l.lower() in name_to_targetlabels:
                matched = True
                for t in name_to_targetlabels[l.lower()]:
                    sourcelabel_targetlabel.append((l, t))
            if l in name_to_targetlabels:
                for t in name_to_targetlabels[l]:
                    sourcelabel_targetlabel.append((l, t))
                matched = True
            if not matched:
                succeed, ns = label_to_synset.convert(l)
                if not succeed:
                    continue
                for n in ns:
                    n = synset_to_noffset(n)
                    if n in tree_noffsets:
                        t = tree_noffsets[n]
                        if t in targetlabel_has_whitelist:
                            # if it has white list, we will not respect the
                            # noffset to do the autmatic matching
                            continue
                        sourcelabel_targetlabel.append((l, t))
                        #result[l] = t
        if self.use_negative_label:
            # we just add the mapping here. no need to check if -s is in the
            # source label list
            sourcelabel_targetlabel.extend([('-' + s, '-' + t) for s, t in
                sourcelabel_targetlabel])

        self._sourcelabel_to_targetlabels = list_to_dict_unique(sourcelabel_targetlabel,
                0)
        self._targetlabel_to_sourcelabels = list_to_dict_unique(sourcelabel_targetlabel,
                1)

        return self._sourcelabel_to_targetlabels

    def select_tsv_rows(self, label_type):
        self._ensure_initialized()
        result = []
        if label_type in self._type_to_split_label_idx:
            split_label_idx = self._type_to_split_label_idx[label_type]
            datasetlabel_to_splitidx = list_to_dict(split_label_idx, 1)
            for datasetlabel in datasetlabel_to_splitidx:
                if datasetlabel in self._sourcelabel_to_targetlabels:
                    split_idxes = datasetlabel_to_splitidx[datasetlabel]
                    targetlabels = self._sourcelabel_to_targetlabels[datasetlabel]
                    for targetlabel in targetlabels:
                        result.extend([(targetlabel, split, idx) for split, idx in
                            split_idxes])
        # must_have_indices
        for split_info in self._split_infos:
            split = split_info['split']
            must_have_indices = split_info.get('must_have_indices', [])
            # we set the target label here as None so that the post-processing
            # will not ignore it. The real labels will also be converted
            # corrected since we do not depend on this target label only.
            result.extend((None, split, i) for i in must_have_indices)
        if self._use_all:
            split_to_targetlabel_idx = list_to_dict(result, 1)
            for s in split_to_targetlabel_idx:
                rootlabel_idxes = split_to_targetlabel_idx[s]
                idx_to_rootlabel = list_to_dict(rootlabel_idxes, 1)
                num_image = self._split_to_num_image[s]
                idxes = set(range(num_image)).difference(set(idx_to_rootlabel.keys()))
                for i in idxes:
                    # for these images, the root label is hard-coded as None
                    result.append((None, s, i))
            for split_info in self._split_infos:
                s = split_info['split']
                if s in split_to_targetlabel_idx:
                    continue
                if s not in self._split_to_num_image:
                    continue
                result.extend([(None, s, i) for i in
                    range(self._split_to_num_image[s])])
        return result

def initialize_images_count(root):
    for node in root.iter_search_nodes():
        node.add_feature('images_with_bb', 0)
        node.add_feature('images_no_bb', 0)

def trainval_split(dataset, num_test_each_label):
    if op.isfile(dataset.get_train_tsv()):
        logging.info('skip to run trainval split for {} because it has been done'.format(
            dataset.name))
        return
    random.seed(777)
    label_to_idx = {}
    for i, row in enumerate(tsv_reader(dataset.get_trainval_tsv('label'))):
        rects = json.loads(row[1])
        if len(rects) == 0:
            logging.info('{} has empty label for {}'.format(dataset.name, i))
            continue
        random.shuffle(rects)
        label = rects[0]['class']
        if label in label_to_idx:
            label_to_idx[label].append(i)
        else:
            label_to_idx[label] = [i]

    test_idx = []
    train_idx = []
    for label in label_to_idx:
        if len(label_to_idx[label]) < num_test_each_label:
            logging.fatal('dataset {} has less than {} images for label {}'.
                    format(dataset.name, num_test_each_label, label))
        random.shuffle(label_to_idx[label])
        test_idx.extend(label_to_idx[label][: num_test_each_label])
        train_idx.extend(label_to_idx[label][num_test_each_label: ])

    trainval = TSVFile(dataset.get_trainval_tsv())

    def gen_train():
        for i in train_idx:
            row = trainval.seek(i)
            yield row
    tsv_writer(gen_train(), dataset.get_train_tsv())

    def gen_test():
        for i in test_idx:
            yield trainval.seek(i)
    tsv_writer(gen_test(), dataset.get_test_tsv_file())

def convert_one_label(rects, label_mapper):
    to_remove = []
    rects2 = []
    for rect in rects:
        if rect['class'] in label_mapper:
            for t in label_mapper[rect['class']]:
                r2 = copy.deepcopy(rect)
                assert type(t) is str or type(t) is unicode
                r2['class'] = t
                rects2.append(r2)
    rects[:] = []
    rects.extend(rects2)

def convert_label_by_dataset(dataset, split, version, idx, label_mapper, with_bb):
    result = None
    for i, row in zip(idx, dataset.iter_data(split, 'label', version=version,
        filter_idx=idx)):
        assert len(row) == 2
        rects = json.loads(row[1])
        def eval_with_bb(r):
            if not r['class'].startswith('-'):
                return 'rect' in r and any(x != 0 for x in r['rect'])
            else:
                return 'rect' not in r or all(x == 0 for x in r['rect'])
        if with_bb:
            # in the case with -, we will not add rect
                                # with all zeros, thus, no need to check if it is all zeros
                                # when it is negative samples
            rects = [r for r in rects if eval_with_bb(r)]
        # all annotations if eval_with_bb(r) is valid for no_bb. Thus, disable
        # the following
        #else:
            #rects = [r for r in rects if not eval_with_bb(r)]
        if result is None:
            # don't use this, because all list will be shared
            #result = [len(row) * ['d']] * tsv.num_rows()
            result = [None] * dataset.num_rows(split)
            for _ in range(len(result)):
                result[_] = ['d'] * len(row)
        rects2 = []
        # the following code should use convert_one_label
        for rect in rects:
            if rect['class'] in label_mapper:
                for t in label_mapper[rect['class']]:
                    r2 = copy.deepcopy(rect)
                    r2['class'] = t
                    if rect['class'] != t:
                        # keep this for logging
                        r2['class_from'] = rect['class']
                    rects2.append(r2)
        row[1] = rects2
        result[i] = row
    return result

def convert_label(label_tsv, idx, label_mapper, with_bb):
    tsv = TSVFile(label_tsv)
    result = None
    for i in tqdm(idx):
        row = tsv.seek(i)
        assert len(row) == 2
        rects = json.loads(row[1])
        def eval_with_bb(r):
            if not r['class'].startswith('-'):
                return 'rect' in r and any(x != 0 for x in r['rect'])
            else:
                return 'rect' not in r or all(x == 0 for x in r['rect'])
        if with_bb:
            # in the case with -, we will not add rect
                                # with all zeros, thus, no need to check if it is all zeros
                                # when it is negative samples
            rects = [r for r in rects if eval_with_bb(r)]
        # all annotations if eval_with_bb(r) is valid for no_bb. Thus, disable
        # the following
        #else:
            #rects = [r for r in rects if not eval_with_bb(r)]
        if result is None:
            # don't use this, because all list will be shared
            #result = [len(row) * ['d']] * tsv.num_rows()
            result = [None] * tsv.num_rows()
            for _ in range(len(result)):
                result[_] = ['d'] * len(row)
        rects2 = []
        # the following code should use convert_one_label
        for rect in rects:
            if rect['class'] in label_mapper:
                for t in label_mapper[rect['class']]:
                    r2 = copy.deepcopy(rect)
                    r2['class'] = t
                    if rect['class'] != t:
                        # keep this for logging
                        r2['class_from'] = rect['class']
                    rects2.append(r2)
        row[1] = rects2
        result[i] = row
    return result

def create_info_for_ambigous_noffset(name, noffsets):
    definitions = [str(noffset_to_synset(n).definition()) for n in noffsets]
    de = [{'noffset': n,
          'definition': d.replace("`", '').replace("'", ''),
          'nick_name': str(get_nick_name(noffset_to_synset(n)))}
            for n, d in zip(noffsets, definitions)]
    d = {'name': name,
            'definitions': de,
            'noffset': None,
            'markdown_url': create_markdown_url(noffsets)}
    return d

def node_should_have_images(root, th, fname):
    enough = True
    few_training_with_bb = []
    for node in root.iter_search_nodes():
        if node == root:
            continue
        if node.cum_images_with_bb < th:
            few_training_with_bb.append({'name': node.name,
                'cum_images_with_bb': node.cum_images_with_bb,
                'parent list': [p.name for p in node.get_ancestors()[:-1]]})
            enough = False
            logging.warn('less images: {} ({})'.format(
                node.name.encode('utf-8'),
                node.cum_images_with_bb))
    if enough:
        logging.info('con. every node has at least {} images'.format(th))
    else:
        write_to_yaml_file(few_training_with_bb, fname)

def clean_dataset2(source_dataset_name, dest_dataset_name):
    source_dataset = TSVDataset(source_dataset_name)
    dest_dataset = TSVDataset(dest_dataset_name)
    splits = get_default_splits()
    for split in splits:
        src_tsv = source_dataset.get_data(split)
        if op.isfile(src_tsv):
            valid_idxs = []
            dest_tsv = dest_dataset.get_data(split)
            def gen_rows():
                rows = tsv_reader(src_tsv)
                num_removed_images = 0
                num_removed_rects = 0
                for i, row in enumerate(tqdm(rows)):
                    if (i % 1000) == 0:
                        logging.info('{} - #removedImages={};#removedRects={}'.format(
                            i, num_removed_images, num_removed_rects))
                    im = img_from_base64(row[-1])
                    if im is None:
                        num_removed_images = num_removed_images + 1
                        continue
                    height, width = im.shape[:2]
                    rects = json.loads(row[1])
                    invalid = False
                    to_remove = []
                    for rect in rects:
                        r = rect['rect']
                        if all(s == 0 for s in r):
                            continue
                        changed = False
                        origin = copy.deepcopy(rect)
                        for j in range(4):
                            if r[j] < 0:
                                r[j] = 0
                                changed = True
                        for j in range(2):
                            if r[2 * j] >= width -1:
                                changed = True
                                r[2 * j] = width - 1
                            if [2 * j + 1] >= height - 1:
                                changed = True
                                r[2 * j + 1] = height - 1
                        if changed:
                            rect['changed_from_rect'] = origin
                        cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                        w, h = r[2] - r[0], r[3] - r[1]
                        if cx < 0 or cy < 0 or cx >= width or \
                                cy >= height or w <= 1 or h <= 1 or \
                                w >= width or h >= height:
                            to_remove.append(rect)
                    if len(to_remove) > 0:
                        num_removed_rects = num_removed_rects + len(to_remove)
                    for rect in to_remove:
                        rects.remove(rect)
                    valid_idxs.append(i)
                    if len(to_remove) > 0:
                        yield row[0], json.dumps(rects), row[2]
                    else:
                        yield row
            tsv_writer(gen_rows(), dest_tsv)
            dest_dataset.write_data(source_dataset.iter_data(split, 'label',
                filter_idx=valid_idxs), split, 'label')

def clean_dataset(source_dataset_name, dest_dataset_name):
    '''
    use version 2
    '''
    source_dataset = TSVDataset(source_dataset_name)
    dest_dataset = TSVDataset(dest_dataset_name)
    splits = get_default_splits()
    for split in splits:
        src_tsv = source_dataset.get_data(split)
        if op.isfile(src_tsv):
            dest_tsv = dest_dataset.get_data(split)
            def gen_rows():
                rows = tsv_reader(src_tsv)
                num_removed_images = 0
                num_removed_rects = 0
                for i, row in enumerate(rows):
                    if (i % 1000) == 0:
                        logging.info('{} - #removedImages={};#removedRects={}'.format(
                            i, num_removed_images, num_removed_rects))
                    im = img_from_base64(row[-1])
                    height, width = im.shape[:2]
                    rects = json.loads(row[1])
                    invalid = False
                    to_remove = []
                    for rect in rects:
                        r = rect['rect']
                        if all(s == 0 for s in r):
                            continue
                        cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                        w, h = r[2] - r[0], r[3] - r[1]
                        if cx < 0 or cy < 0 or cx >= width or \
                                cy >= height or w <= 1 or h <= 1 or \
                                w >= width or h >= height:
                            to_remove.append(rect)
                    if len(to_remove) > 0:
                        logging.info('before removing {}'.format(len(rects)))
                        num_removed_rects = num_removed_rects + len(to_remove)
                    for rect in to_remove:
                        rects.remove(rect)
                        r = rect['rect']
                        logging.info('removing {}'.format(','.join(map(str,
                                r))))
                    if len(to_remove) > 0:
                        logging.info('after removing {}'.format(len(rects)))
                        if len(rects) > 0:
                            yield row[0], json.dumps(rects), row[2]
                        else:
                            num_removed_images = num_removed_images + 1
                            logging.info('removing image {}'.format(row[0]))
                    else:
                        yield row
            tsv_writer(gen_rows(), dest_tsv)

def parallel_convert_label(func, all_task, num_worker=128):
    num_split = num_worker
    num_task_each_split = (len(all_task) + num_split - 1) / num_split
    all_sub_tasks = []
    for i in range(num_split):
        start_idx = i * num_task_each_split
        if start_idx >= len(all_task):
            break
        end_idx = start_idx + num_task_each_split
        if end_idx > len(all_task):
            end_idx = len(all_task)
        all_sub_tasks.append(all_task[start_idx:end_idx])
    m = Pool(num_worker)
    all_sub_results = m.map(func, all_sub_tasks)
    result = all_sub_results[0]
    for s in all_sub_results[1:]:
        assert len(result) == len(s)
        for i in range(len(result)):
            assert len(result[i]) == 2
            assert len(s[i]) == 2
            if result[i][1] == 'd' and s[i][1] != 'd':
                result[i][0] = s[i][0]
                result[i][1] = s[i][1]
            else:
                assert not (result[i][1] != 'd' and s[i][1] != 'd')
    return result

def parallel_map_to_array(func, all_task, num_worker=128):
    num_split = num_worker * 2
    num_task_each_split = (len(all_task) + num_split - 1) / num_split
    all_sub_tasks = []
    for i in range(num_split):
        start_idx = i * num_task_each_split
        if start_idx > len(all_task):
            break
        end_idx = start_idx + num_task_each_split
        if end_idx > len(all_task):
            end_idx = len(all_task)
        all_sub_tasks.append(all_task[start_idx:end_idx])
    m = Pool(num_worker)
    all_sub_results = m.map(func, all_sub_tasks)
    result = []
    for s in all_sub_results:
        result.extend(s)
    return result

def convert_label_db(dataset, split, idx, with_bb):
    result = None
    label_mapper = dataset._sourcelabel_to_targetlabels
    queried = []
    for row_with_idx in tqdm(dataset.iter_gt(split, idx=idx)):
        i = row_with_idx[0]
        queried.append(i)
        row = list(row_with_idx[1:])
        rects = row[1]
        def eval_with_bb(r):
            if not r['class'].startswith('-'):
                return 'rect' in r and any(x != 0 for x in r['rect'])
            else:
                return 'rect' not in r or all(x == 0 for x in r['rect'])
        if with_bb:
            # in the case with -, we will not add rect
                                # with all zeros, thus, no need to check if it is all zeros
                                # when it is negative samples
            rects = [r for r in rects if eval_with_bb(r)]
        # all annotations if eval_with_bb(r) is valid for no_bb. Thus, disable
        # the following
        #else:
            #rects = [r for r in rects if not eval_with_bb(r)]
        if result is None:
            result = [None] * dataset.num_rows(split)
            for _ in range(len(result)):
                result[_] = ['d'] * len(row)
        rects2 = []
        # the following code should use convert_one_label
        for rect in rects:
            if rect['class'] in label_mapper:
                for t in label_mapper[rect['class']]:
                    r2 = copy.deepcopy(rect)
                    r2['class'] = t
                    if rect['class'] != t:
                        # keep this for logging
                        r2['class_from'] = rect['class']
                    rects2.append(r2)
        row[1] = rects2
        result[i] = row
    not_coverred = set(idx).difference(queried)
    assert len(not_coverred) == 0
    return result

def create_trainX_db(train_ldtsi, extra_dtsi, tax, out_dataset,
        lift_train=False):
    t_to_ldsi = list_to_dict(train_ldtsi, 2)
    extra_t_to_dsi = list_to_dict(extra_dtsi, 1)
    train_ldtsik = []
    extra_dtsik = []
    for label_type in t_to_ldsi:
        ldsi = t_to_ldsi[label_type]
        extra_dsi = extra_t_to_dsi.get(label_type, [])
        d_to_lsi = list_to_dict(ldsi, 1)
        extra_d_to_si = list_to_dict(extra_dsi, 0)
        k = 0
        sources = []
        sources_label = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            extra_si = extra_d_to_si.get(dataset, [])
            s_li = list_to_dict(lsi, 1)
            extra_s_to_i = list_to_dict(extra_si, 0)
            for split in s_li:
                li = s_li[split]
                idx_to_l = list_to_dict(li, 1)
                idx = idx_to_l.keys()
                extra_i = extra_s_to_i.get(split, [])
                # link the data tsv
                source = dataset.get_data(split)
                out_split = 'train{}'.format(k)
                train_ldtsik.extend([(l, dataset, label_type, split, i,
                    k) for l, i in li])
                extra_dtsik.extend([(dataset, label_type, split, i, k)
                    for i in extra_i])
                k = k + 1
                sources.append(source)
                logging.info('converting labels: {}-{}'.format(
                    dataset.name, split))

                converted_label = convert_label_db(dataset,
                        split, idx, with_bb=with_bb)
                # convert the file name
                logging.info('delifting the labels')
                for i in tqdm(idx):
                    l = converted_label[i]
                    if lift_train:
                        l[1] = json.dumps(lift_one_image(l[1], tax))
                    else:
                        l[1] = json.dumps(delift_one_image(l[1], tax))
                    l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                label_file = out_dataset[label_type].get_data(out_split, 'label')
                logging.info('writing the label file {}'.format(label_file))
                tsv_writer(converted_label, label_file)
                sources_label.append(label_file)
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('trainX'))
        write_to_file('\n'.join(sources_label),
                out_dataset[label_type].get_data('trainX', 'label'))

    logging.info('saving the shuffle file')
    type_to_ldsik = list_to_dict(train_ldtsik, 2)
    extra_type_to_dsik = list_to_dict(extra_dtsik, 1)
    for label_type in type_to_ldsik:
        ldsik = type_to_ldsik[label_type]
        shuffle_info = [(str(k), str(i)) for l, d, s, i, k in ldsik]
        shuffle_info = list(set(shuffle_info))
        if label_type in extra_type_to_dsik:
            dsik = extra_type_to_dsik[label_type]
            # we should not de-duplicate it because it comes from the duplicate
            # policy
            extra_shuffle_info = [(str(k), str(i) ) for d, s, i, k in dsik]
            shuffle_info.extend(extra_shuffle_info)
        random.shuffle(shuffle_info)
        tsv_writer(shuffle_info,
                out_dataset[label_type].get_shuffle_file('train'))

    populate_output_num_images(train_ldtsik, 'toTrain', tax.root)

def create_trainX(train_ldtsi, extra_dtsi, tax, out_dataset,
        lift_train=False, use_original_label=False):
    t_to_ldsi = list_to_dict(train_ldtsi, 2)
    extra_t_to_dsi = list_to_dict(extra_dtsi, 1)
    train_ldtsik = []
    extra_dtsik = []
    for label_type in t_to_ldsi:
        ldsi = t_to_ldsi[label_type]
        extra_dsi = extra_t_to_dsi.get(label_type, [])
        d_to_lsi = list_to_dict(ldsi, 1)
        extra_d_to_si = list_to_dict(extra_dsi, 0)
        k = 0
        sources = []
        sources_origin_label = []
        sources_label = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            extra_si = extra_d_to_si.get(dataset, [])
            s_li = list_to_dict(lsi, 1)
            extra_s_to_i = list_to_dict(extra_si, 0)
            for split in s_li:
                li = s_li[split]
                idx_to_l = list_to_dict(li, 1)
                idx = idx_to_l.keys()
                extra_i = extra_s_to_i.get(split, [])
                # link the data tsv
                source = dataset.get_data(split)
                out_split = 'train{}'.format(k)
                train_ldtsik.extend([(l, dataset, label_type, split, i,
                    k) for l, i in li])
                extra_dtsik.extend([(dataset, label_type, split, i, k)
                    for i in extra_i])
                k = k + 1
                sources.append(source)
                logging.info('converting labels: {}-{}'.format(
                    dataset.name, split))
                source_origin_label = dataset.get_label_tsv(split)
                if not use_original_label:
                    #converted_label = convert_label(source_origin_label,
                            #idx, dataset._sourcelabel_to_targetlabels,
                            #with_bb=with_bb)
                    converted_label = convert_label_by_dataset(
                            dataset, split,
                            dataset.get_version_by_split(split),
                            idx, dataset._sourcelabel_to_targetlabels,
                            with_bb=with_bb)
                    sources_origin_label.append(source_origin_label)
                    # convert the file name
                    logging.info('delifting the labels')
                    for i in tqdm(idx):
                        l = converted_label[i]
                        if lift_train:
                            l[1] = json.dumps(lift_one_image(l[1], tax))
                        else:
                            l[1] = json.dumps(delift_one_image(l[1], tax))
                        l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                    label_file = out_dataset[label_type].get_data(out_split, 'label')
                    logging.info('writing the label file {}'.format(label_file))
                    tsv_writer(converted_label, label_file)
                else:
                    label_file = source_origin_label
                sources_label.append(label_file)
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('trainX'))
        write_to_file('\n'.join(sources_label),
                out_dataset[label_type].get_data('trainX', 'label'))
        write_to_file('\n'.join(sources_origin_label),
                out_dataset[label_type].get_data('trainX', 'origin.label'))
    logging.info('saving the shuffle file')
    type_to_ldsik = list_to_dict(train_ldtsik, 2)
    extra_type_to_dsik = list_to_dict(extra_dtsik, 1)
    for label_type in type_to_ldsik:
        ldsik = type_to_ldsik[label_type]
        shuffle_info = [(str(k), str(i)) for l, d, s, i, k in ldsik]
        shuffle_info = list(set(shuffle_info))
        if label_type in extra_type_to_dsik:
            dsik = extra_type_to_dsik[label_type]
            # we should not de-duplicate it because it comes from the duplicate
            # policy
            extra_shuffle_info = [(str(k), str(i) ) for d, s, i, k in dsik]
            shuffle_info.extend(extra_shuffle_info)
        random.shuffle(shuffle_info)
        tsv_writer(shuffle_info,
                out_dataset[label_type].get_shuffle_file('train'))

        # expand nested trainX if needed
        expand_nested_splitX(out_dataset[label_type].name, 'train')

    populate_output_num_images(train_ldtsik, 'toTrain', tax.root)

def expand_nested_splitX(data, split):
    dataset = TSVDataset(data)
    if not dataset.has(split):
        logging.info('{}-{} does not exist'.format(data, split))
        return
    if op.isfile(dataset.get_data(split)):
        logging.info('{}/{}.tsv already exists'.format(data, split))
        return
    assert op.isfile(dataset.get_data(split + 'X'))

    need_to_expand = False
    src_data_splits = dataset.load_composite_source_data_split(split)
    for src_data, src_split in src_data_splits:
        src_dataset = TSVDataset(src_data)
        if not op.isfile(src_dataset.get_data(src_split)):
            assert op.isfile(src_dataset.get_data(src_split + 'X'))
            need_to_expand = True
            break
    if not need_to_expand:
        logging.info('no need to expand')
        return

    if not op.isfile(dataset.get_data(split, 'label')):
        convertcomposite_to_standard(data, split, ignore_image=True)

    cache_shuffle = {}
    def get_shuffle(d, s):
        if (d, s) in cache_shuffle:
            return cache_shuffle[(d, s)]
        else:
            shuffle = [(int(i), int(j)) for i, j in
                tsv_reader(TSVDataset(d).get_shuffle_file(s))]
            cache_shuffle[(d, s)] = shuffle
            return shuffle

    cache_src_data_splits = {}
    def get_src_data_splits(d, s):
        if (d, s) in cache_src_data_splits:
            return cache_src_data_splits[(d, s)]
        else:
            ssrc_data_splits = TSVDataset(d).load_composite_source_data_split(s)
            cache_src_data_splits[(d, s)] = ssrc_data_splits
            return ssrc_data_splits

    cache_single_tsv = {}
    def get_single(d, s):
        if (d, s) in cache_single_tsv:
            return cache_single_tsv[(d, s)]
        else:
            single = TSVDataset(d).get_data(s)
            cache_single_tsv[(d, s)] = single
            return single

    cache_is_single = {}
    def get_is_single(d, s):
        if (d, s) in cache_is_single:
            return cache_is_single[(d, s)]
        else:
            is_single = op.isfile(TSVDataset(d).get_data(s))
            cache_is_single[(d, s)] = is_single
            return is_single

    def get_tsv_line(d, s, i):
        single_tsv = get_is_single(d, s)
        if single_tsv:
            return get_single(d, s), i
        else:
            ssrc_data_splits = get_src_data_splits(d, s)
            shuffle = get_shuffle(d, s)
            idx_src, idx_row = shuffle[i]
            src_d, src_s = ssrc_data_splits[idx_src]
            return get_tsv_line(src_d, src_s, idx_row)

    shuffle = get_shuffle(data, split)
    all_tsv_idx = []
    for idx_src, idx_row in shuffle:
        ss_data, ss_split = src_data_splits[idx_src]
        tsv, i = get_tsv_line(ss_data, ss_split, idx_row)
        all_tsv_idx.append((tsv, i))
    unique_tsvs = list(list_to_dict(all_tsv_idx, 0).keys())
    tsv_to_idx = {t: i for i, t in enumerate(unique_tsvs)}
    shuffle_result = [(tsv_to_idx[t], i) for t, i in all_tsv_idx]
    tsv_writer(shuffle_result, dataset.get_shuffle_file(split))
    write_to_file('\n'.join(unique_tsvs), dataset.get_data(split + 'X'))

def create_testX_db(test_ldtsi, tax, out_dataset):
    t_to_ldsi = list_to_dict(test_ldtsi, 2)
    for label_type in t_to_ldsi:
        sources = []
        sources_label = []
        ldsi = t_to_ldsi[label_type]
        d_to_lsi = list_to_dict(ldsi, 1)
        k = 0
        all_ki = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            s_to_li = list_to_dict(lsi, 1)
            for split in s_to_li:
                li = s_to_li[split]
                idx = list_to_dict(li, 1).keys()
                out_split = 'test{}'.format(k)
                s_file = dataset.get_data(split)
                sources.append(s_file)
                converted_label = convert_label_db(dataset,
                        split, idx, with_bb=with_bb)
                for i in tqdm(idx):
                    l = converted_label[i]
                    l[1] = json.dumps(lift_one_image(l[1], tax))
                    l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                all_ki.extend([(str(k), str(i)) for i in idx])
                label_file = out_dataset[label_type].get_data(out_split, 'label')
                tsv_writer(converted_label, label_file)
                sources_label.append(label_file)
                k = k + 1
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('testX'))
        write_to_file('\n'.join(sources_label),
                out_dataset[label_type].get_data('testX', 'label'))
        tsv_writer(all_ki, out_dataset[label_type].get_shuffle_file('test'))

def create_testX(test_ldtsi, tax, out_dataset):
    t_to_ldsi = list_to_dict(test_ldtsi, 2)
    for label_type in t_to_ldsi:
        sources = []
        sources_origin_label = []
        sources_label = []
        ldsi = t_to_ldsi[label_type]
        d_to_lsi = list_to_dict(ldsi, 1)
        k = 0
        all_ki = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            s_to_li = list_to_dict(lsi, 1)
            for split in s_to_li:
                li = s_to_li[split]
                idx = list_to_dict(li, 1).keys()
                out_split = 'test{}'.format(k)
                s_file = dataset.get_data(split)
                sources.append(s_file)
                source_origin_label = dataset.get_label_tsv(split)
                sources_origin_label.append(source_origin_label)
                #converted_label = convert_label(source_origin_label,
                        #idx, dataset._sourcelabel_to_targetlabels,
                        #with_bb=with_bb)
                converted_label = convert_label_by_dataset(
                        dataset, split, dataset.get_version_by_split(split),
                        idx, dataset._sourcelabel_to_targetlabels,
                        with_bb=with_bb)
                for i in tqdm(idx):
                    l = converted_label[i]
                    l[1] = json.dumps(lift_one_image(l[1], tax))
                    l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                all_ki.extend([(str(k), str(i)) for i in idx])
                label_file = out_dataset[label_type].get_data(out_split, 'label')
                tsv_writer(converted_label, label_file)
                sources_label.append(label_file)
                k = k + 1
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('testX'))
        write_to_file('\n'.join(sources_label),
                out_dataset[label_type].get_data('testX', 'label'))
        write_to_file('\n'.join(sources_origin_label),
                out_dataset[label_type].get_data('testX', 'origin.label'))
        tsv_writer(all_ki, out_dataset[label_type].get_shuffle_file('test'))
        expand_nested_splitX(out_dataset[label_type].name, 'test')

def remove_or_duplicate_each_type(train_ldtsi, label_to_min_image, label_to_max_image):
    label_to_dtsi = list_to_dict(train_ldtsi, 0)
    extra_dtsi = []
    for label in label_to_dtsi:
        if label is None:
            continue
        dtsi = label_to_dtsi[label]
        max_image = label_to_max_image[label]
        t_to_dsi = list_to_dict(dtsi, 1)
        if isinstance(label_to_min_image, dict):
            min_image = label_to_min_image[label]
        else:
            # label_to_min_image is an integer
            min_image = label_to_min_image
        for t in t_to_dsi:
            dsi = t_to_dsi[t]
            if len(dsi) > max_image:
                # first remove the images with no bounding box
                num_remove = len(dsi) - max_image
                random.seed(9999)
                random.shuffle(dsi)
                dsi = sorted(dsi, key=lambda x: -x[0].cleaness)
                assert len(dsi) > num_remove
                dsi = dsi[: len(dsi) - num_remove]
                t_to_dsi[t] = dsi
            elif len(dsi) < min_image:
                num_duplicate = int(np.ceil(float(min_image) / len(dsi)))
                logging.info('duplicate images for {}/{}: {}->{}, {}'.format(
                    t, label, len(dsi), min_image, num_duplicate))
                # we already has 1 copy of dsi, thus we should subtract 1
                extra_dsi = (num_duplicate - 1) * dsi
                extra_dtsi.extend([[d, t, s, i] for d, s, i in extra_dsi])
        dtsi = dict_to_list(t_to_dsi, 1)
        label_to_dtsi[label] = dtsi
    logging.info('# train instances before duplication: {}'.format(len(train_ldtsi)))
    train_ldtsi = dict_to_list(label_to_dtsi, 0)
    logging.info('# train instances after duplication: {}'.format(len(train_ldtsi)))
    return train_ldtsi, extra_dtsi

def remove_or_duplicate(train_ldtsi, min_image, label_to_max_image):
    # min_image is for the sum of with_bb and no_bb
    # prefer to use _each_type version, which interpret min_image as the value
    # for each type
    label_to_dtsi = list_to_dict(train_ldtsi, 0)
    extra_dtsi = []
    for label in label_to_dtsi:
        dtsi = label_to_dtsi[label]
        max_image = label_to_max_image[label]
        if len(dtsi) > max_image:
            # first remove the images with no bounding box
            num_remove = len(dtsi) - max_image
            type_to_dsi = list_to_dict(dtsi, 1)
            if 'no_bb' in type_to_dsi:
                dsi = type_to_dsi['no_bb']
                if num_remove >= len(dsi):
                    # remove all this images
                    del type_to_dsi['no_bb']
                    num_remove = num_remove - len(dsi)
                else:
                    random.seed(9999)
                    random.shuffle(dsi)
                    type_to_dsi['no_bb'] = dsi[: len(dsi) - num_remove]
                    num_remove = 0
            if num_remove > 0:
                assert 'with_bb' in type_to_dsi
                dsi = type_to_dsi['with_bb']
                random.seed(9999)
                random.shuffle(dsi)
                dsi = sorted(dsi, key=lambda x: -x[0].cleaness)
                assert len(dsi) > num_remove
                type_to_dsi['with_bb'] = dsi[: len(dsi) - num_remove]
                num_remove = 0
            dtsi = dict_to_list(type_to_dsi, 1)
        elif len(dtsi) < min_image:
            num_duplicate = int(np.ceil(float(min_image) / len(dtsi)))
            logging.info('duplicate images for label of {}: {}->{}, {}'.format(
                label, len(dtsi), min_image, num_duplicate))
            extra_dtsi.extend((num_duplicate - 1) * dtsi)
        label_to_dtsi[label] = dtsi
    logging.info('# train instances before duplication: {}'.format(len(train_ldtsi)))
    train_ldtsi = dict_to_list(label_to_dtsi, 0)
    logging.info('# train instances after duplication: {}'.format(len(train_ldtsi)))
    return train_ldtsi, extra_dtsi

def remove_test_in_train(train_ldtsi, test_ldtsi):
    logging.info('before len(train_ldtsi) = {}'.format(len(train_ldtsi)))
    set_train_dtsi = set((d, t, s, i) for l, d, t, s, i in train_ldtsi)
    if len(train_ldtsi) > 0 and type(train_ldtsi[0]) is list:
        # we need to convert it to immutable tuple since list is not hashable
        train_ldtsi = [(l, d, t, s, i) for l, d, t, s, i in train_ldtsi]
    set_train_ldtsi = set(train_ldtsi)
    #assert len(set_train_ldtsi) == len(train_ldtsi)
    set_test_dtsi = set((d, t, s, i) for l, d, t, s, i in test_ldtsi)
    result = [(l, d, t, s, i) for l, d, t, s, i in train_ldtsi
            if (d, t, s, i) not in set_test_dtsi]
    logging.info('after len(train_ldtsi) = {}'.format(len(result)))
    return result

def split_train_test(ldtsi, num_test):
    # group by label_type
    t_to_ldsi = list_to_dict(ldtsi, 2)
    train_ldtsi = []
    test_ldtsi = []
    for label_type in sorted(t_to_ldsi.keys()):
        ldsi= t_to_ldsi[label_type]
        l_to_dsi = list_to_dict(ldsi, 0)
        for rootlabel in sorted(l_to_dsi.keys()):
            dsi = l_to_dsi[rootlabel]
            if len(dsi) < num_test:
                logging.info('rootlabel={}; label_type={}->less than {} images'.format(
                    rootlabel, label_type, len(dsi)))
            curr_num_test = min(num_test, int(len(dsi) / 2))
            random.shuffle(dsi)
            test_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i
                in dsi[:curr_num_test]])
            train_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i
                in dsi[curr_num_test:]])
    return train_ldtsi, test_ldtsi

def regularize_data_sources(data_infos):
    result = []
    default_splits = get_default_splits()
    for data_info in data_infos:
        if type(data_info) is str:
            r = {'name': data_info,
                    'cleaness': 10,
                    'valid_splits': default_splits,
                    'use_all': False}
            result.append(r)
        elif type(data_info) is tuple or type(data_info) is list:
            assert len(data_info) > 0
            r = {'name': data_info[0],
                    'cleaness': data_info[1] if len(data_info) > 1 else 10,
                    'valid_splits': data_info[2] if len(data_info) > 2 else
                                default_splits,
                    'use_all': data_info[3] if len(data_info) > 3 else False,
                    }
            assert len(data_info) < 5
            result.append(r)
        elif type(data_info) is dict:
            r = data_info
            result.append(r)
        else:
            raise Exception('unkwown data_info = {}'.format(data_info))
    for r in result:
        if 'data' in r:
            assert 'name' not in r
            r['name'] = r['data']
            del r['data']
    return result

def parse_data_clean_splits(data_infos):
    '''
    use regularize_data_sources
    '''
    datas, cleaness, all_valid_splits = [], [], []
    default_splits = get_default_splits()
    for data_info in data_infos:
        if type(data_info) is str:
            datas.append(data_info)
            cleaness.append(10)
            all_valid_splits.append(default_splits)
        elif type(data_info) is tuple or type(data_info) is list:
            assert len(data_info) > 0
            datas.append(data_info[0])
            cleaness.append(data_info[1] if len(data_info) > 1 else 10)
            all_valid_splits.append(data_info[2] if len(data_info) > 2 else default_splits)
        elif type(data_info) is dict:
            datas.append(data_info['data'])
            cleaness.append(data_info.get('cleaness', 10))
            all_valid_splits.append(data_info.get('valid_splits', default_splits))
        else:
            raise Exception('unkwown data_info = {}'.format(data_info))
    return datas, cleaness, all_valid_splits

def attach_properties(src_nodes, dst_tree):
    name_to_dst_node = {n.name: n for n in dst_tree.iter_search_nodes() if
            n != dst_tree}
    confusings = []
    for src_node in src_nodes:
        if src_node.name not in name_to_dst_node:
            continue
        dst_node = name_to_dst_node[src_node.name]
        for f in src_node.features:
            if f in ['support', 'name', 'dist', 'sub_group']:
                continue
            if f in dst_node.features:
                if dst_node.__getattribute__(f) == src_node.__getattribute__(f):
                    continue
                else:
                    confusings.append({'name': src_node.name,
                        'feature': f,
                        'value in tree': dst_node.__getattribute__(f),
                        'value in property list': src_node.__getattribute__(f)})
            else:
                dst_node.add_feature(f, src_node.__getattribute__(f))
    assert len(confusings) == 0, pformat(confusings)

def update_taxonomy_by_latest(ref_data, target_data):
    lift_train = False
    from .common import ensure_copy_folder
    # copy everything from ref_data to target_data and _no_bb
    ensure_copy_folder(op.join('data', ref_data),
            op.join('data', target_data))
    ensure_copy_folder(op.join('data', ref_data + '_no_bb'),
            op.join('data', target_data + '_no_bb'))

    # _with_bb
    ref_data = ref_data + '_with_bb'
    target_data = target_data + '_with_bb'
    update_bb_taxonomy_by_latest(ref_data, target_data)

def update_bb_taxonomy_by_latest(ref_data, target_data):
    ref_dataset = TSVDataset(ref_data)
    out_dataset = TSVDataset(target_data)

    split = 'train'
    splitX = '{}X'.format(split)
    all_idxsource_idxrow = [(int(s_idx_source), int(s_idx_row)) for s_idx_source, s_idx_row in
            tsv_reader(ref_dataset.get_shuffle_file(split))]
    pattern = 'data/(.*)/(train|trainval|test)\.label.*\.tsv'
    # e.g. source_origin_label = 'data/SeeingAISplit/train.label.tsv'
    source_data_splits = [re.match(pattern, source_origin_label).groups()
            for source_origin_label, in ref_dataset.iter_data(splitX, 'origin.label')]
    source_data_split_versions = [(d, s, -1) for d, s in source_data_splits]

    lift_train = False
    dump_to_taxonomy_dataset(ref_dataset, all_idxsource_idxrow,
            source_data_split_versions, lift_train, split, out_dataset)

def dump_to_taxonomy_dataset(ref_dataset, all_idxsource_idxrow,
        source_data_split_versions, lift_train, split, out_dataset):
    splitX = split + 'X'
    tax = Taxonomy(load_from_yaml_file(op.join(ref_dataset._data_root, 'root.yaml')))
    sources_label = []
    sources_origin_label = []
    for idxsource, (source_data, source_split, source_version) in enumerate(source_data_split_versions):
        idx = [idx_r for idx_s, idx_r in all_idxsource_idxrow if idxsource == idx_s]
        for n in tax.root.iter_search_nodes():
            if source_data in n.features:
                n.add_feature(source_data,
                        n.__getattribute__(source_data) + ',{}'.format(n.name))
        source_dataset = TSVDatasetSource(source_data, root=tax.root,
                split_infos=[{'split': source_split, 'version': -1}])
        source_dataset._ensure_initialized()
        source_origin_label = source_dataset.get_data(source_split, 'label',
                source_version)
        sources_origin_label.append(source_origin_label)
        converted_label = convert_label(source_origin_label,
                idx, source_dataset._sourcelabel_to_targetlabels,
                with_bb=True)
        for i in tqdm(idx):
            l = converted_label[i]
            if lift_train:
                l[1] = json.dumps(lift_one_image(l[1], tax))
            else:
                l[1] = json.dumps(delift_one_image(l[1], tax))
            l[0] = '{}_{}_{}'.format(source_dataset.name, source_split, l[0])
        out_split = '{}{}'.format(split, idxsource)
        label_file = out_dataset.get_data(out_split, 'label')
        tsv_writer(converted_label, label_file)
        sources_label.append(label_file)

    # the label version might be updated. Thus, the sources_label could be
    # different from the reference dataset
    write_to_file('\n'.join(sources_label),
            out_dataset.get_data(splitX, 'label'))

    write_to_file('\n'.join(sources_origin_label),
            out_dataset.get_data(splitX, 'origin.label'))

    # copy the image source file
    ensure_copy_file(ref_dataset.get_data(splitX),
            out_dataset.get_data(splitX))

    # copy the labelmap
    ensure_copy_file(ref_dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file())

    # copy the shuffle file
    # the idx could be different since we might add more images here
    tsv_writer(all_idxsource_idxrow,
            out_dataset.get_shuffle_file(split))

def test():
    data = 'LogosInTheWild-v2Clean'
    idx = 1237
    dataset = TSVDataset(data)
    tsv = TSVFile(dataset.get_data('train', 'label', 3))
    key, str_rects = tsv.seek(idx)
    key, _, str_im = dataset.seek_by_key(key, 'train')

    #im = img_from_base64(str_im)
    jpgbytestring = base64.b64decode(str_im)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_IGNORE_ORIENTATION);

    rects = json.loads(str_rects)
    draw_rects(im, rects)
    save_image(im, '/mnt/jianfw_desk/a.png')

def trim_rects_from_db(rects):
    keys = ['_id', 'create_time', 'idx_in_split',
            'key', 'split', 'data']
    for rect in rects:
        for k in keys:
            if k in rect:
                del rect[k]

class TSVDatasetDB(object):
    def __init__(self, name):
        self.name = name
        self._db = create_mongodb_client()
        self._gt = self._db['qd']['ground_truth']
        self._image = self._db['qd']['image']

    def iter_gt(self, split, version=None, version_by_time=None,
            idx=None, bb_type='with_bb'):
        # query the image db first and then attach the gt information
        match = {'data': self.name,
                 'split': split}
        if idx:
            match['idx_in_split'] = {'$in': list(idx)}
        lookup_pipeline = self._get_gt_rect_pipeline(
                split, version, version_by_time, bb_type,
                extra_filter={'$eq': ['$key', '$$key']},
                in_lookup=True)
        pipeline = [{'$match': match},
                    {'$lookup': {'from': 'ground_truth',
                                 'let': {'key': '$key'},
                                 'pipeline': lookup_pipeline,
                                 'as': 'rects'
                                 }}]
        for row in self._image.aggregate(pipeline, allowDiskUse=True):
            trim_rects_from_db(row['rects'])
            yield row['idx_in_split'], row['key'], row['rects']

    @deprecated('use iter_gt since this function will not return the image with no bb')
    def iter_gt_image(self, split, version=None, version_by_time=None,
            idx=None, bb_type='with_bb'):
        # deprecated. use iter_gt since this function will not return anything
        # if the image in idx has no box
        # if idx is not none, here we do not guarrentee the order is kept.
        # Thus, we also return the index of the image
        if idx:
            extra_filter = {'idx_in_split': {'$in': list(idx)}}
        else:
            extra_filter = None
        pipeline = self._get_gt_rect_pipeline(split,
                version, version_by_time, bb_type,
                extra_filter)
        pipeline.append({'$group': {'_id': '$idx_in_split',
                                    'key': {'$first': '$key'},
                                    'rects': {'$push': '$$ROOT'}}})
        logging.info(pformat(pipeline))
        for result in self._gt.aggregate(pipeline, allowDiskUse=True):
            for rect in result['rects']:
                if '_id' in rect:
                    del rect['_id']
                if 'create_time' in rect:
                    del rect['create_time']
            yield result['_id'], result['key'], result['rects']

    def _get_gt_rect_pipeline(self, split, version, version_by_time,
            bb_type, extra_filter, in_lookup=False):
        if not in_lookup:
            match = {'data': self.name,
                     'split': split}
        else:
            match_condition = []
            match_condition.append({'$eq': ['$data', self.name]})
            match_condition.append({'$eq': ['$split', split]})
        if extra_filter:
            if not in_lookup:
                match.update(extra_filter)
            else:
                match_condition.append(extra_filter)
        if version_by_time:
            if not in_lookup:
                match['create_time'] = {'$lte': version_by_time}
            else:
                match_condition.append({'$lte': ['$create_time', version_by_time]})
            sort_by = 'create_time'
        else:
            if version is None:
                version = 0
            if version != -1:
                if not in_lookup:
                    match['version'] = {'$lte': version}
                else:
                    match_condition.append({'$lte': ['$version', version]})
            sort_by = 'version'
        if in_lookup:
            match = {'$expr': {'$and': match_condition}}
        sort_value = OrderedDict() # use ordered since dict is non-ordered
        sort_value[sort_by] = -1
        sort_value['contribution'] = -1
        pipeline = [{'$match': match},
                    # we may change some properties at the same version or
                    # time. In this case, we prefer contribution=1
                    {'$sort': sort_value},
                    {'$group': {'_id':      '$action_target_id',
                                'rect_info': {'$first': '$$ROOT'}}},
                    {'$replaceRoot': {'newRoot': '$rect_info'}},
                    {'$match': {'contribution': 1}},
                    ]
        if bb_type == 'with_bb':
            cond1 = {'rect': {'$ne': None}}
            cond2_or = []
            for i in range(4):
                cond2_or.append({'rect.{}'.format(i): {'$ne': 0}})
            cond2 = {'$or': cond2_or}
            match_with_bb = {'$and': [cond1, cond2]}
            pipeline.append({'$match': match_with_bb})
        else:
            assert bb_type == 'no_bb'
        return pipeline

    def iter_gt_rect(self, split, version=None, version_by_time=None,
            bb_type='with_bb',
            extra_filter=None):
            # we do not add any filter here. Thus it is no_bb + with_bb
        pipeline = self._get_gt_rect_pipeline(split, version, version_by_time,
                bb_type, extra_filter)
        return self._gt.aggregate(pipeline, allowDiskUse=True)

    def num_rows(self, split):
        pipeline = [{'$match': {'data': self.name, 'split': split}},
                {'$group': {'_id': 1, 'm': {'$max': '$idx_in_split'}}}]
        result = next(self._image.aggregate(pipeline))
        return result['m'] + 1

    def get_data(self, split):
        return op.join('data', self.name, '{}.tsv'.format(split))

    def _get_unique_labels(self):
        # we ignore the version to make it simplier. it should cover all labels
        splits = [split_info['split'] for split_info in self._split_infos]
        pipeline = [{'$match': {'data': self.name,
                                'split': {'$in': splits}}},
                    {'$group': {'_id': '$class'}}]
        labelmap = sorted([r['_id'] for r in self._gt.aggregate(pipeline)])
        return labelmap

class TSVDatasetSourceDB(TSVDatasetDB):
    def __init__(self, name, root=None,
            split_infos=None,
            cleaness=10,
            use_all=False,
            use_negative_label=False,
            select_by_verified=False):
        super(TSVDatasetSourceDB, self).__init__(name)
        self._noffset_count = {}
        self._type = None
        self._root = root
        # the list of <datasetlabel, rootlabel>
        self._sourcelabel_to_targetlabels = None
        self._targetlabel_to_sourcelabels = None
        self._initialized = False
        self._type_to_datasetlabel_to_split_idx = None
        self._type_to_datasetlabel_to_count = None
        self._type_to_split_label_idx = None
        if split_infos is not None:
            self._split_infos = split_infos
        else:
            self._split_infos = [{'split': s, 'version': -1}
                    for s in get_default_splits()]
        self._split_to_info = {s['split']: s for s in self._split_infos}
        assert len(set([split_info['split'] for split_info in
            self._split_infos])) == len(self._split_infos)
        self._use_all = use_all
        self._select_by_verified = select_by_verified
        self.cleaness = cleaness
        self.use_negative_label = use_negative_label

    def populate_info(self, root):
        self._ensure_initialized()
        for node in root.iter_search_nodes():
            if root == node:
                continue
            if node.name in self._targetlabel_to_sourcelabels:
                sourcelabels = self._targetlabel_to_sourcelabels[node.name]
                node.add_feature(self.name, ','.join(sourcelabels))
                if any(is_noffset(l) for l in sourcelabels):
                    node.add_feature(self.name + '_readable',
                            ','.join(get_nick_name(noffset_to_synset(l)) if
                                is_noffset(l) else l for l in sourcelabels))
                total = 0
                for sourcelabel in sourcelabels:
                    for t in self._type_to_datasetlabel_to_count:
                        datasetlabel_to_count = self._type_to_datasetlabel_to_count[t]
                        c = datasetlabel_to_count.get(sourcelabel, 0)
                        total = total + c
                        key = 'images_{}'.format(t)
                        node.add_feature(key,
                                node.__getattribute__(key) + c)
                node.add_feature('{}_total'.format(self.name), total)

    def _ensure_initialized(self):
        if self._initialized:
            return

        self.update_label_mapper()

        self._load_inverted()

        self._initialized = True

    def _load_inverted(self):
        # make sure self.update_label_mapper() is called
        types = ['with_bb', 'no_bb']
        self._type_split_label_idx = []

        usefull_dataset_labels = self._sourcelabel_to_targetlabels.keys()
        usefull_dataset_labels = [l for l in usefull_dataset_labels if not l.startswith('-')]
        for split_info in self._split_infos:
            for bb_type in types:
                iter_merged = self.iter_gt_rect(split=split_info['split'],
                        version=split_info.get('version'),
                        version_by_time=split_info.get('version_by_time'),
                        bb_type=bb_type,
                        extra_filter={'class': {'$in': usefull_dataset_labels}})
                type_split_label_idx = ((bb_type, rect_info['split'],
                    rect_info['class'], rect_info['idx_in_split'])
                    for rect_info in iter_merged)
                self._type_split_label_idx.extend(type_split_label_idx)

        assert not self.use_negative_label, 'not supported'

        self._type_split_label_idx = list(set(self._type_split_label_idx))
        self._type_to_split_label_idx = list_to_dict(
                self._type_split_label_idx, 0)
        self._type_to_datasetlabel_to_split_idx = {}
        for t in self._type_to_split_label_idx:
            split_label_idx = self._type_to_split_label_idx[t]
            label_to_split_idx = list_to_dict(split_label_idx, 1)
            self._type_to_datasetlabel_to_split_idx[t] = label_to_split_idx
        self._type_to_datasetlabel_to_count = {}
        for t in self._type_to_datasetlabel_to_split_idx:
            datasetlabel_to_split_idx = self._type_to_datasetlabel_to_split_idx[t]
            datasetlabel_to_count = {l: len(datasetlabel_to_split_idx[l]) for l in
                    datasetlabel_to_split_idx}
            self._type_to_datasetlabel_to_count[t] = datasetlabel_to_count

        self._split_to_num_image = {split_info['split']: self.num_rows(split_info['split']) for split_info in
                self._split_infos}

    def update_label_mapper(self):
        root = self._root

        labelmap = self._get_unique_labels()
        hash_labelmap = set(labelmap)

        tree_noffsets = {}
        for node in root.iter_search_nodes():
            if node == root or not node.noffset:
                continue
            for s in node.noffset.split(','):
                tree_noffsets[s] = node.name
        name_to_targetlabels = {}
        targetlabel_has_whitelist = set()
        invalid_list = []
        any_source_key = 'label_names_in_all_dataset_source'
        for node in root.iter_search_nodes():
            if node == root:
                continue
            if hasattr(node, self.name) or hasattr(node, any_source_key):
                if hasattr(node, self.name):
                    # this is like a white-list
                    values = node.__getattribute__(self.name)
                else:
                    values = node.__getattribute__(any_source_key)
                if values is not None:
                    source_terms = values.split(',')
                    for t in source_terms:
                        t = t.strip()
                        if t not in name_to_targetlabels:
                            name_to_targetlabels[t] = set()
                        if t not in hash_labelmap:
                            invalid_list.append((t, self.name, node.name))
                            continue
                        name_to_targetlabels[t].add(node.name)
                # even if it is None, we will also add it to white-list so that
                # we will not automatically match the term.
                targetlabel_has_whitelist.add(node.name)
            else:
                # we will keep the lower case always for case-insensitive
                # comparison
                all_candidate_src_names = [node.name.lower()]
                if hasattr(node, 'alias_names'):
                    all_candidate_src_names.extend([s.strip() for s in
                        node.alias_names.split(',')])
                for t in set(all_candidate_src_names):
                    if t not in name_to_targetlabels:
                        name_to_targetlabels[t] = set()
                    name_to_targetlabels[t].add(node.name)

        sourcelabel_targetlabel = []
        if len(invalid_list) != 0:
            logging.warn('invalid white list information: {}'.format(pformat(invalid_list)))

        #result = {}
        label_to_synset = LabelToSynset()
        for l in labelmap:
            matched = False
            if l.lower() in name_to_targetlabels:
                matched = True
                for t in name_to_targetlabels[l.lower()]:
                    sourcelabel_targetlabel.append((l, t))
            if l in name_to_targetlabels:
                for t in name_to_targetlabels[l]:
                    sourcelabel_targetlabel.append((l, t))
                matched = True
            if not matched:
                succeed, ns = label_to_synset.convert(l)
                if not succeed:
                    continue
                for n in ns:
                    n = synset_to_noffset(n)
                    if n in tree_noffsets:
                        t = tree_noffsets[n]
                        if t in targetlabel_has_whitelist:
                            # if it has white list, we will not respect the
                            # noffset to do the autmatic matching
                            continue
                        sourcelabel_targetlabel.append((l, t))
                        #result[l] = t
        if self.use_negative_label:
            # we just add the mapping here. no need to check if -s is in the
            # source label list
            sourcelabel_targetlabel.extend([('-' + s, '-' + t) for s, t in
                sourcelabel_targetlabel])

        self._sourcelabel_to_targetlabels = list_to_dict_unique(sourcelabel_targetlabel,
                0)
        self._targetlabel_to_sourcelabels = list_to_dict_unique(sourcelabel_targetlabel,
                1)

        return self._sourcelabel_to_targetlabels

    def iter_gt_image(self, split, idx=None, bb_type='with_bb'):
        split_info = self._split_to_info[split]

        return super(TSVDatasetSourceDB, self).iter_gt_image(split,
                version=split_info.get('version'),
                version_by_time=split_info.get('version_by_time'),
                idx=idx, bb_type=bb_type)

    def select_tsv_rows(self, label_type):
        self._ensure_initialized()
        result = []
        if label_type in self._type_to_split_label_idx:
            split_label_idx = self._type_to_split_label_idx[label_type]
            datasetlabel_to_splitidx = list_to_dict(split_label_idx, 1)
            for datasetlabel in datasetlabel_to_splitidx:
                if datasetlabel in self._sourcelabel_to_targetlabels:
                    split_idxes = datasetlabel_to_splitidx[datasetlabel]
                    targetlabels = self._sourcelabel_to_targetlabels[datasetlabel]
                    for targetlabel in targetlabels:
                        result.extend([(targetlabel, split, idx) for split, idx in
                            split_idxes])
        # must_have_indices
        for split_info in self._split_infos:
            split = split_info['split']
            must_have_indices = split_info.get('must_have_indices', [])
            # we set the target label here as None so that the post-processing
            # will not ignore it. The real labels will also be converted
            # corrected since we do not depend on this target label only.
            result.extend((None, split, i) for i in must_have_indices)
        if self._use_all:
            split_to_targetlabel_idx = list_to_dict(result, 1)
            for s in split_to_targetlabel_idx:
                rootlabel_idxes = split_to_targetlabel_idx[s]
                idx_to_rootlabel = list_to_dict(rootlabel_idxes, 1)
                num_image = self._split_to_num_image[s]
                idxes = set(range(num_image)).difference(set(idx_to_rootlabel.keys()))
                for i in idxes:
                    # for these images, the root label is hard-coded as None
                    result.append((None, s, i))
            for split_info in self._split_infos:
                s = split_info['split']
                if s in split_to_targetlabel_idx:
                    continue
                if s not in self._split_to_num_image:
                    continue
                result.extend([(None, s, i) for i in
                    range(self._split_to_num_image[s])])
        return result

def build_tax_dataset_from_db(taxonomy_folder, **kwargs):
    random.seed(777)
    dataset_name = kwargs.get('data',
            op.basename(taxonomy_folder))
    overall_dataset = TSVDataset(dataset_name)
    if op.isfile(overall_dataset.get_labelmap_file()):
        logging.info('ignore to build taxonomy since {} exists'.format(
            overall_dataset.get_labelmap_file()))
        return
    init_logging()
    logging.info('building {}'.format(dataset_name))
    all_tax = load_all_tax(taxonomy_folder)
    tax = merge_all_tax(all_tax)
    tax.update()
    initialize_images_count(tax.root)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k_448', tax.root)
    if op.isfile(imagenet22k.get_labelmap_file()):
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('there is no imagenet22k_448 dataset to help identify the noffset')
    populate_url_for_offset(tax.root)

    ambigous_noffset_file = op.join(overall_dataset._data_root,
            'ambigous_noffsets.yaml')
    output_ambigous_noffsets(tax.root, ambigous_noffset_file)

    data_infos = regularize_data_sources(kwargs['datas'])

    data_sources = [TSVDatasetSourceDB(root=tax.root, **d)
            for d in data_infos]

    for s in data_sources:
        s.populate_info(tax.root)

    populate_cum_images(tax.root)

    labels, child_parent_sgs = child_parent_print_tree2(tax.root, 'name')

    label_map_file = overall_dataset.get_labelmap_file()
    write_to_file('\n'.join(labels), label_map_file)
    # save the parameter
    write_to_yaml_file((taxonomy_folder, kwargs), op.join(overall_dataset._data_root,
            'generate_parameters.yaml'))
    dest_taxonomy_folder = op.join(overall_dataset._data_root,
            'taxonomy_folder')
    if op.isdir(dest_taxonomy_folder):
        shutil.rmtree(dest_taxonomy_folder)
    shutil.copytree(taxonomy_folder, dest_taxonomy_folder)

    out_dataset = {'with_bb': TSVDataset(dataset_name + '_with_bb'),
            'no_bb': TSVDataset(dataset_name + '_no_bb')}

    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_labelmap_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(label_map_file, target_file)

    logging.info('cum_images_with_bb: {}'.format(tax.root.cum_images_with_bb))
    logging.info('cum_images_no_bb: {}'.format(tax.root.cum_images_no_bb))

    # write the simplified version of the tree
    dest = op.join(overall_dataset._data_root, 'root.simple.yaml')
    write_to_yaml_file(tax.dump(['images_with_bb']), dest)

    tree_file = overall_dataset.get_tree_file()
    write_to_file('\n'.join(['{} {}{}'.format(c, p, '' if sg < 0 else ' {}'.format(sg))
                             for c, p, sg in child_parent_sgs]),
            tree_file)
    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_tree_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(tree_file, target_file)

    node_should_have_images(tax.root, 200,
            op.join(overall_dataset._data_root, 'labels_with_few_images.yaml'))

    # get the information of all train val
    ldtsi = []
    logging.info('collecting all candidate images')
    for label_type in out_dataset:
        for dataset in data_sources:
            targetlabel_split_idxes = dataset.select_tsv_rows(label_type)
            for rootlabel, split, idx in targetlabel_split_idxes:
                ldtsi.append((rootlabel, dataset, label_type, split, idx))
    # we need to remove the duplicates. the duplicates could come from such
    # cases: for example, we have Laptop and laptop in the image. Both of the
    # labels are mapped to laptop, which is in the target domain. In this case,
    # the image could be in the list twice
    ldtsi = list(set(ldtsi))

    num_test = kwargs.get('num_test', 50)

    # for each label, let's duplicate the image or remove the image
    default_max_image = kwargs.get('max_image_per_label', 1000)
    label_to_max_image = {n.name: n.__getattribute__('max_image_extract_for_train')
            if 'max_image_extract_for_train' in n.features and n.__getattribute__('max_image_extract_for_train') > default_max_image
            else default_max_image for n in tax.root.iter_search_nodes() if n != tax.root}
    label_to_max_image = {l: max(label_to_max_image[l], num_test) for l in label_to_max_image}
    # negative images constraint
    labels = list(label_to_max_image.keys())
    for l in labels:
        label_to_max_image['-' + l] = label_to_max_image[l]
    label_to_max_image[None] = 10000000000
    min_image = kwargs.get('min_image_per_label', 200)

    logging.info('keep a small image pool to split')
    label_to_max_augmented_images = {l: label_to_max_image[l] * 3 for l in label_to_max_image}
    # reduce the computing cost
    ldtsi, extra_dtsi = remove_or_duplicate_each_type(ldtsi, 0,
            label_to_max_augmented_images)
    assert len(extra_dtsi) == 0

    logging.info('select the best test image')
    if num_test == 0:
        test_ldtsi = []
    else:
        # generate the test set from the best data source
        label_to_max_images_for_test = {l: num_test for l in
            label_to_max_image}
        test_ldtsi, extra_dtsi = remove_or_duplicate_each_type(ldtsi, 0,
                label_to_max_images_for_test)
        assert len(extra_dtsi) == 0

    logging.info('removing test images from image pool')
    train_ldtsi = remove_test_in_train(ldtsi, test_ldtsi)

    logging.info('select the final training images')
    train_ldtsi, extra_dtsi = remove_or_duplicate_each_type(train_ldtsi, min_image,
            label_to_max_image)

    logging.info('creating the train data')
    create_trainX_db(train_ldtsi, extra_dtsi, tax, out_dataset,
            lift_train=kwargs.get('lift_train', False))

    populate_output_num_images(test_ldtsi, 'toTest', tax.root)

    # dump the tree to yaml format
    dest = op.join(overall_dataset._data_root, 'root.yaml')
    d = tax.dump()
    write_to_yaml_file(d, dest)
    for label_type in out_dataset:
        target_file = op.join(out_dataset[label_type]._data_root, 'root.yaml')
        ensure_directory(op.dirname(target_file))
        shutil.copy(dest, target_file)

    create_testX_db(test_ldtsi, tax, out_dataset)

    logging.info('done')

def build_taxonomy_from_single_composite_source(source_data,
        source_split, source_version,
        min_image_per_label,
        out_data):
    info = get_frame_info()
    out_dataset = TSVDataset(out_data)
    if op.isdir(out_dataset._data_root):
        logging.info('ignore to build since exists')
        return
    populate_dataset_details(source_data)
    dataset = TSVDataset(source_data)
    label_to_idx = dataset.load_inverted_label(source_split,
            version=source_version)
    if min_image_per_label < 1:
        # this is the ratio beteen the real value and the average
        mean_count = calc_mean([len(idx) for _, idx in label_to_idx.items()])
        min_image_per_label = int(mean_count * min_image_per_label)
    all_idx = list(range(dataset.num_rows(source_split)))
    idx_to_labels = list_to_dict(dict_to_list(label_to_idx, 0), 1)

    while len(label_to_idx) > 0:
        # remove the labels if the len(idx) >= min_image_per_label
        to_remove = [l for l in label_to_idx if len(label_to_idx[l]) >= min_image_per_label]
        for l in to_remove:
            del label_to_idx[l]
        if len(label_to_idx) == 0:
            break
        min_label = min(label_to_idx, key=lambda x: len(label_to_idx[x]))
        min_idx = copy.deepcopy(label_to_idx[min_label])
        min_count = len(min_idx)
        copies = (min_image_per_label + min_count - 1)  // min_count - 1
        info['duplicate_info${}'.format(min_label)] = copies
        assert copies > 0
        # add these extra images
        all_idx.extend(copies * min_idx)
        for i in min_idx:
            for l in idx_to_labels[i]:
                if l in label_to_idx:
                    # otherwise, it means that label is enough
                    for _ in range(copies):
                        label_to_idx[l].append(i)
    random.seed(6)
    random.shuffle(all_idx)
    ensure_copy_file(dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file())

    shuffle = TSVFile(dataset.get_shuffle_file(source_split))
    tsv_writer((shuffle[i] for i in all_idx), out_dataset.get_shuffle_file('train'))
    tsv_copy(dataset.get_data(source_split + 'X'),
            out_dataset.get_data('trainX'))

    out_dataset.write_data(
            dataset.iter_data('train', 'hw', filter_idx=all_idx),
            'train', 'hw')
    out_dataset.write_data(
            dataset.iter_data('train', 'label', version=source_version,
                filter_idx=all_idx),
            'train', 'label')
    out_dataset.write_data(list(info.items()), 'train', 'generate_info')

def build_taxonomy_from_single_source(source_data,
        source_split, source_version,
        min_image_per_label,
        out_data):
    populate_dataset_details(source_data)
    populate_dataset_hw(source_data, ['train'])
    dataset = TSVDataset(source_data)
    label_to_idx = dataset.load_inverted_label(source_split,
            version=source_version)
    all_idx = list(range(dataset.num_rows(source_split)))
    idx_to_labels = list_to_dict(dict_to_list(label_to_idx, 0), 1)

    while len(label_to_idx) > 0:
        # remove the labels if the len(idx) >= min_image_per_label
        to_remove = [l for l in label_to_idx if len(label_to_idx[l]) >= min_image_per_label]
        for l in to_remove:
            del label_to_idx[l]
        if len(label_to_idx) == 0:
            break
        min_label = min(label_to_idx, key=lambda x: len(label_to_idx[x]))
        min_idx = copy.deepcopy(label_to_idx[min_label])
        min_count = len(min_idx)
        copies = (min_image_per_label + min_count - 1)  // min_count - 1
        assert copies > 0
        # add these extra images
        all_idx.extend(copies * min_idx)
        for i in min_idx:
            for l in idx_to_labels[i]:
                if l in label_to_idx:
                    # otherwise, it means that label is enough
                    for _ in range(copies):
                        label_to_idx[l].append(i)
    random.seed(6)
    random.shuffle(all_idx)
    out_dataset = TSVDataset(out_data)
    shuffle = [(0, r) for r in all_idx]
    tsv_writer(shuffle, out_dataset.get_train_shuffle_file())
    ensure_copy_file(dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file())

    write_to_file(dataset.get_data('train'),
            out_dataset.get_data('trainX'))
    write_to_file(dataset.get_data('train', 'hw'),
            out_dataset.get_data('trainX', 'hw'))
    write_to_file(dataset.get_data('train', 'label', version=source_version),
            out_dataset.get_data('trainX', 'label'))

def build_taxonomy_impl(taxonomy_folder,
        data, datas,
        taxonomy=None,
        num_test=50, max_image_per_label=1000,
        min_image_per_label=200,
        lift_train=False,
        use_original_label=False,
        **kwargs):
    random.seed(777)
    dataset_name = data
    overall_dataset = TSVDataset(dataset_name)
    if op.isfile(overall_dataset.get_labelmap_file()):
        logging.info('ignore to build taxonomy since {} exists'.format(
            overall_dataset.get_labelmap_file()))
        return
    init_logging()
    logging.info('building {}'.format(dataset_name))
    if taxonomy_folder is not None:
        all_tax = load_all_tax(taxonomy_folder)
        tax = merge_all_tax(all_tax)
    else:
        tax = Taxonomy(taxonomy)
    tax.update()
    initialize_images_count(tax.root)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k_448', tax.root)
    if op.isfile(imagenet22k.get_labelmap_file()):
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('there is no imagenet22k_448 dataset to help identify the noffset')
    populate_url_for_offset(tax.root)

    ambigous_noffset_file = op.join(overall_dataset._data_root,
            'ambigous_noffsets.yaml')
    output_ambigous_noffsets(tax.root, ambigous_noffset_file)

    if isinstance(datas, str):
        datas = load_from_yaml_file(datas)
    data_infos = regularize_data_sources(datas)

    data_sources = [TSVDatasetSource(root=tax.root, **d)
            for d in data_infos]

    for s in data_sources:
        s.populate_info(tax.root)

    populate_cum_images(tax.root)

    labels, child_parent_sgs = child_parent_print_tree2(tax.root, 'name')

    label_map_file = overall_dataset.get_labelmap_file()
    write_to_file('\n'.join(labels), label_map_file)
    # save the parameter
    write_to_yaml_file(get_frame_info(),
            op.join(overall_dataset._data_root,
            'generate_parameters.yaml'))
    dest_taxonomy_folder = op.join(overall_dataset._data_root,
            'taxonomy_folder')
    if op.isdir(dest_taxonomy_folder):
        shutil.rmtree(dest_taxonomy_folder)
    if taxonomy_folder is not None:
        shutil.copytree(taxonomy_folder, dest_taxonomy_folder)
    else:
        write_to_yaml_file(taxonomy, op.join(dest_taxonomy_folder, 'root.yaml'))

    out_dataset = {'with_bb': TSVDataset(dataset_name + '_with_bb'),
            'no_bb': TSVDataset(dataset_name + '_no_bb')}

    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_labelmap_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(label_map_file, target_file)

    logging.info('cum_images_with_bb: {}'.format(tax.root.cum_images_with_bb))
    logging.info('cum_images_no_bb: {}'.format(tax.root.cum_images_no_bb))

    # write the simplified version of the tree
    dest = op.join(overall_dataset._data_root, 'root.simple.yaml')
    write_to_yaml_file(tax.dump(['images_with_bb']), dest)

    tree_file = overall_dataset.get_tree_file()
    write_to_file('\n'.join(['{} {}{}'.format(c, p, '' if sg < 0 else ' {}'.format(sg))
                             for c, p, sg in child_parent_sgs]),
            tree_file)
    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_tree_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(tree_file, target_file)

    node_should_have_images(tax.root, 200,
            op.join(overall_dataset._data_root, 'labels_with_few_images.yaml'))

    # get the information of all train val
    ldtsi = []
    logging.info('collecting all candidate images')
    for label_type in out_dataset:
        for dataset in data_sources:
            targetlabel_split_idxes = dataset.select_tsv_rows(label_type)
            for rootlabel, split, idx in targetlabel_split_idxes:
                ldtsi.append((rootlabel, dataset, label_type, split, idx))
    # we need to remove the duplicates. the duplicates could come from such
    # cases: for example, we have Laptop and laptop in the image. Both of the
    # labels are mapped to laptop, which is in the target domain. In this case,
    # the image could be in the list twice
    ldtsi = list(set(ldtsi))

    # for each label, let's duplicate the image or remove the image
    default_max_image = max_image_per_label
    label_to_max_image = {n.name: n.__getattribute__('max_image_extract_for_train')
            if 'max_image_extract_for_train' in n.features and n.__getattribute__('max_image_extract_for_train') > default_max_image
            else default_max_image for n in tax.root.iter_search_nodes() if n != tax.root}
    label_to_max_image = {l: max(label_to_max_image[l], num_test) for l in label_to_max_image}
    # negative images constraint
    labels = list(label_to_max_image.keys())
    for l in labels:
        label_to_max_image['-' + l] = label_to_max_image[l]
    label_to_max_image[None] = 10000000000

    label_to_min_image = {n.name: n.__getattribute__('min_image_extract_for_train')
            if 'min_image_extract_for_train' in n.features
            else min_image_per_label for n in tax.root.iter_search_nodes() if n != tax.root}

    logging.info('keep a small image pool to split')
    label_to_max_augmented_images = {l: label_to_max_image[l] * 3 for l in label_to_max_image}
    # reduce the computing cost
    ldtsi, extra_dtsi = remove_or_duplicate_each_type(ldtsi, 0,
            label_to_max_augmented_images)
    assert len(extra_dtsi) == 0

    logging.info('select the best test image')
    if num_test == 0:
        test_ldtsi = []
    else:
        # generate the test set from the best data source
        label_to_max_images_for_test = {l: num_test for l in
            label_to_max_image}
        test_ldtsi, extra_dtsi = remove_or_duplicate_each_type(ldtsi, 0,
                label_to_max_images_for_test)
        assert len(extra_dtsi) == 0

    logging.info('removing test images from image pool')
    train_ldtsi = remove_test_in_train(ldtsi, test_ldtsi)

    logging.info('select the final training images')
    train_ldtsi, extra_dtsi = remove_or_duplicate_each_type(
        train_ldtsi,
        label_to_min_image,
        label_to_max_image)

    logging.info('creating the train data')
    create_trainX(train_ldtsi, extra_dtsi, tax, out_dataset,
            lift_train=lift_train,
            use_original_label=use_original_label)

    populate_output_num_images(test_ldtsi, 'toTest', tax.root)

    # dump the tree to yaml format
    dest = op.join(overall_dataset._data_root, 'root.yaml')
    d = tax.dump()
    write_to_yaml_file(d, dest)
    for label_type in out_dataset:
        target_file = op.join(out_dataset[label_type]._data_root, 'root.yaml')
        ensure_directory(op.dirname(target_file))
        shutil.copy(dest, target_file)

    create_testX(test_ldtsi, tax, out_dataset)

    logging.info('done')

def delift_one_image(rects, tax):
    # currently, for the training, we need to delift the label. That is, if it
    # is man, we should remove the person label
    rects2 = []
    for curr_r in rects:
        if curr_r['class'].startswith('-'):
            # if it is a background, we do nothing here. In the future, we
            # might change this logic
            rects2.append(curr_r)
            continue
        curr_label = curr_r['class']
        if 'rect' not in curr_r:
            same_place_rects = rects2
        else:
            ious = [calculate_iou(r['rect'], curr_r['rect']) for r in rects2]
            same_place_rects = [r for i, r in zip(ious, rects2) if i > 0.9]
        # if current label is one of parent of the same_place rects, ignore it
        ignore = False
        for same_place_r in same_place_rects:
            ancestors = tax.name_to_ancestors[same_place_r['class']]
            if curr_label in ancestors or same_place_r['class'] == curr_label:
                ignore = True
                break
        if ignore:
            continue
        ancestors = tax.name_to_ancestors[curr_label]
        to_removed = []
        for same_place_r in same_place_rects:
            if same_place_r['class'] in ancestors:
                to_removed.append(same_place_r)
        for t in to_removed:
            rects2.remove(t)
        rects2.append(curr_r)
    return rects2

def lift_one_image(rects, tax):
    rects2 = []
    for curr_r in rects:
        if curr_r['class'].startswith('-'):
            rects2.append(curr_r)
            continue
        label = curr_r['class']
        all_label = tax.name_to_ancestors[label]
        all_label.add(label)
        for l in all_label:
            same_label_rects = [r for r in rects2 if r['class'] == l]
            if 'rect' in curr_r:
                ious = [calculate_iou(r['rect'], curr_r['rect']) for r in
                    same_label_rects if 'rect' in r]
                if len(ious) > 0 and max(ious) > 0.9:
                    continue
                else:
                    r = copy.deepcopy(curr_r)
                    r['class'] = l
                    rects2.append(r)
            else:
                if len(same_label_rects) == 0:
                    r = copy.deepcopy(curr_r)
                    r['class'] = l
                    rects2.append(r)
    return rects2

def populate_output_num_images(ldtX, suffix, root):
    label_to_node = {n.name: n for n in root.iter_search_nodes() if n != root}
    targetlabel_to_dX = list_to_dict(ldtX, 0)
    for targetlabel in targetlabel_to_dX:
        if not targetlabel or targetlabel.startswith('-'):
            # currently, we ignore this background case. In the future, we
            # might change this logic
            continue
        if not targetlabel:
            # it means background images
            continue
        dtX = targetlabel_to_dX[targetlabel]
        dataset_to_X = list_to_dict(dtX, 0)
        for dataset in dataset_to_X:
            X = dataset_to_X[dataset]
            if len(X) == 0:
                continue
            key = '{}_{}'.format(dataset.name, suffix)
            value = len(X)
            label_to_node[targetlabel].add_feature(key, value)
        labeltype_to_dX = list_to_dict(dtX, 1)
        for labeltype in labeltype_to_dX:
            dX = labeltype_to_dX[labeltype]
            key = '{}_{}'.format(labeltype, suffix)
            value = len(dX)
            label_to_node[targetlabel].add_feature(key, value)

def output_ambigous_noffsets(root, ambigous_noffset_file):
    ambigous = []
    for node in root.iter_search_nodes():
        if hasattr(node, 'noffsets') and node.noffset is None:
            noffsets = node.noffsets.split(',')
            d = create_info_for_ambigous_noffset(node.name, noffsets)
            d['parent_name'] = ','.join([n.name for n in
                        node.get_ancestors()[:-1]])
            ambigous.append(d)
    if len(ambigous) > 0:
        logging.info('output ambigous terms to {}'.format(ambigous_noffset_file))
        write_to_yaml_file(ambigous, ambigous_noffset_file)
    else:
        logging.info('Congratulations on no ambigous terms.')

def output_ambigous_noffsets_main(tax_input_folder, ambigous_file_out):
    all_tax = load_all_tax(tax_input_folder)
    tax = merge_all_tax(all_tax)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k')
    if op.isfile(imagenet22k.get_labelmap_file()):
        logging.info('remove the noffset if it is not in imagenet22k')
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('no imagenet22k data used to help remove noffset ambiguities')

    populate_url_for_offset(tax.root)

    output_ambigous_noffsets(tax.root, ambigous_file_out)

def convert_inverted_file(name):
    d = TSVDataset(name)
    splits = get_default_splits()
    for split in splits:
        logging.info('loading {}-{}'.format(name, split))
        x = d.load_inverted_label(split)
        def gen_rows():
            for label in x:
                idx = x[label]
                yield label, ' '.join(map(str, idx))

        inverted_file = d.get_data(split, 'inverted.label')
        target_file = op.splitext(inverted_file)[0] + '.tsv'
        if not op.isfile(inverted_file):
            if op.isfile(target_file):
                os.remove(target_file)
            continue
        if op.isfile(target_file):
            continue
        tsv_writer(gen_rows(), target_file)

def standarize_crawled(tsv_input, tsv_output):
    rows = tsv_reader(tsv_input)
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info(i)
            image_str = row[-1]
            image_label = row[0]
            rects = [{'rect': [0, 0, 0, 0], 'class': image_label}]
            image_name = '{}_{}'.format(op.basename(tsv_input), i)
            yield image_name, json.dumps(rects), image_str
    tsv_writer(gen_rows(), tsv_output)

def get_data_sources(version):
    return load_from_yaml_file('./aux_data/data_sources/{}.yaml'.format(version))

def get_img_url2(img_key):
    # don't use get_image_url
    clean_name = _map_img_key_to_name2(img_key)
    url = _get_url_from_name(clean_name)
    return url

def get_img_url(img_key):
    # use version 2
    clean_name = _map_img_key_to_name(img_key)
    url = _get_url_from_name(clean_name)
    return url

def map_image_key_to_url_key(data, split, key):
    return hash_sha1(str((data, split, key)))

@deprecated(reason='need to incoroprate the data split info')
def _map_img_key_to_name2(key):
    assert len(key) > 0
    ext = '.jpg'
    pattern = '^([0-9]|-|[a-z]|[A-Z]|\.|_)*$'
    if not re.match(pattern, key):
        key = hash_sha1(key)
    key = key.lower()
    if key.endswith(ext):
        return key
    else:
        return key + ext

def _map_img_key_to_name(key):
    # use version 2
    _EXT = ".jpg"
    if key.startswith("brand"):
        return "brand" + str(hash(key)) + _EXT
    else:
        key = key.lower()
        if key.endswith(_EXT):
            return key
        else:
            return key + _EXT

def _get_url_from_name(name):
    _SITE = 'https://cogsimagestorage.blob.core.windows.net/'
    _CONTAINER_NAME = "detectortrain"
    return _SITE + _CONTAINER_NAME + "/" + name

def parse_combine(key):
    pattern = '(.*?)_(train|trainval|test)_(.*)'
    result = re.match(pattern, key)
    if result is None:
        return None, None, key
    else:
        return result.groups()

def convert_to_uhrs_with_url(data):
    dataset = TSVDataset(data)
    for split in get_default_splits():
        if not dataset.has(split, 'label'):
            continue
        v = dataset.get_latest_version(split, 'label')
        def gen_rows():
            for row in dataset.iter_data(split, 'label', version=v):
                key = row[0]
                _, _, key = parse_combine(key)
                row.append(get_img_url(key))
                yield row
        dataset.write_data(gen_rows(), split,
                'url', version=v)

def find_same_location_rects(target, rects, iou=0.95):
    return [r for r in rects if
        calculate_iou(target['rect'], r['rect']) > iou]

def find_same_rects(target, rects, iou=0.95):
    same_class_rects = [r for r in rects if r['class'] == target['class']]
    return [r for r in same_class_rects if
        calculate_iou(target['rect'], r['rect']) > iou]

def rects_diff(lefts, rights, iou=0.95):
    result = []
    for l in lefts:
        if len(find_same_rects(l, rights, iou)) == 0:
            result.append(l)
    return result

def rects_inter(lefts, rights, iou=0.95):
    result = []
    for l in lefts:
        if len(find_same_rects(l, rights, iou)) > 0:
            result.append(l)
    return result


def rect_in_rects(target, rects, iou=0.95):
    same_class_rects = [r for r in rects if r['class'] == target['class']]
    if 'rect' not in target:
        return len(same_class_rects) > 0
    else:
        return any(r for r in same_class_rects if 'rect' in r and
            calculate_iou(target['rect'], r['rect']) > iou)

def strict_rect_in_rects(target, rects):
    return any(float_tolorance_equal(target, r) for r in rects)

def load_key_rects(iter_data):
    assert type(iter_data) is not str
    result = []
    logging.info('loading key rects')
    for row in tqdm(iter_data):
        assert len(row) == 2
        rects = json.loads(row[1])
        if isinstance(rects, int):
            # in this case, it is imagenet2012 dataset, which is the class
            # index.
            rects = [{'class': str(rects)}]
        result.append([row[0], rects])
    return result

def convert_uhrs_result_back_to_sources(in_tsv, debug=True, tree_file=None):
    rows = tsv_reader(in_tsv)
    key_rects3 = []
    num_yes, num_no, num_un = 0, 0, 0
    for row in rows:
        # key, yes, no, uncertain
        assert len(row) == 4
        rects_yes = json.loads(row[1])
        rects_no = json.loads(row[2])
        rects_un = json.loads(row[3])
        num_yes = num_yes + len(rects_yes)
        num_no = num_no + len(rects_no)
        num_un = num_un + len(rects_un)
        key_rects3.append([row[0], [rects_yes, rects_no, rects_un]])

    logging.info('#yes={}; #no={}; #un={}; #yes/(#yes+#no+#un)={}'.format(
        num_yes, num_no, num_un, 1.*num_yes/(num_yes+num_no+num_un)))

    if tree_file:
        tax = Taxonomy(load_from_yaml_file(tree_file))
        mapper = LabelToSynset()
        mapper.populate_noffset(tax.root)

    datas = get_data_sources()
    datas = [data for data, _ in datas]
    datasplitkey_rects3 = [[parse_combine(key), rects3]
            for key, rects3 in key_rects3]
    data_split_key_rects3 = [(data, split, key, rects3)
            for (data, split, key), rects3 in datasplitkey_rects3]

    data_to_split_key_rects3 = list_to_dict(data_split_key_rects3, 0)

    for data in data_to_split_key_rects3:
        logging.info(data)
        if tree_file:
            source_dataset = TSVDatasetSource(data, tax.root)
            source_dataset._ensure_initialized()
        else:
            source_dataset = TSVDataset(data)
        split_key_rects3 = data_to_split_key_rects3[data]
        split_to_key_rects3 = list_to_dict(split_key_rects3, 0)
        for split in split_to_key_rects3:
            logging.info(split)
            key_rects3 = split_to_key_rects3[split]
            key_to_rects3 = list_to_dict(key_rects3, 0)
            v = source_dataset.get_latest_version(split, 'label')
            logging.info('{} - {}'.format(data, split))
            source_key_rects = load_key_rects(source_dataset.iter_data(split, 'label', version=v))
            is_equal = True
            num_added, num_removed = 0, 0
            meta = {'in_tsv': in_tsv}
            for i, (key, origin_rects) in tqdm(enumerate(source_key_rects)):
                if debug:
                    old_origin_rects = copy.deepcopy(origin_rects)
                yes_rects, no_rects, un_rects = key_to_rects3.get(key, [[[], [], []]])[0]
                # if no_rects are in original, remove it. remove first
                for r in no_rects:
                    delete_rects = find_same_rects(r, origin_rects)
                    if tree_file:
                        if r['class'] in source_dataset._targetlabel_to_sourcelabels:
                            for reverse_class in source_dataset._targetlabel_to_sourcelabels[r['class']]:
                                r2 = copy.deepcopy(r)
                                r2['class'] = reverse_class
                                delete_rects2 = find_same_rects(r2, origin_rects)
                                delete_rects.extend(delete_rects2)
                    if 'class_from' in r:
                        r2 = copy.deepcopy(r)
                        r2['class'] = r2['class_from']
                        delete_rects2 = find_same_rects(r2, origin_rects)
                        delete_rects.extend(delete_rects2)
                    for d in delete_rects:
                        # delete rects may include duplicate terms
                        if d in origin_rects:
                            origin_rects.remove(d)
                            num_removed = num_removed + 1
                            is_equal = False
                # if yes_rects are not in original, add it
                for r in yes_rects:
                    same_rects = find_same_rects(r, origin_rects)
                    if len(same_rects) > 0:
                        for s in same_rects:
                            if s.get('uhrs_confirm', 0) == 0:
                                is_equal = False
                            s['uhrs_confirm'] = s.get('uhrs_confirm', 0) + 1
                    else:
                        r['uhrs_confirm'] = r.get('uhrs_confirm', 0) + 1
                        origin_rects.append(copy.deepcopy(r))
                        is_equal = False
                        num_added = num_added + 1
                for r in un_rects:
                    same_rects = find_same_rects(r, origin_rects)
                    for s in same_rects:
                        if s.get('uhrs_uncertain', 0) == 0:
                            is_equal = True
                        s['uhrs_uncertain'] = s.get('uhrs_uncertain', 0) + 1
                if debug:
                    if len(origin_rects) != len(old_origin_rects):
                        for _, _, im_str in source_dataset.iter_data(split, filter_idx=[i]):
                            im = img_from_base64(im_str)
                            old_im = im.copy()
                            draw_bb(old_im, [r['rect'] for r in old_origin_rects],
                                    [r['class'] for r in old_origin_rects])
                            new_im = im.copy()
                            draw_bb(new_im, [r['rect'] for r in origin_rects],
                                    [r['class'] for r in origin_rects])
                            yes_im = im.copy()
                            draw_bb(yes_im, [r['rect'] for r in yes_rects],
                                    [r['class'] for r in yes_rects])
                            no_im = im.copy()
                            draw_bb(no_im, [r['rect'] for r in no_rects],
                                    [r['class'] for r in no_rects])

                            logging.info(pformat(old_origin_rects))
                            logging.info(pformat(origin_rects))
                            logging.info(pformat(yes_rects))
                            logging.info(pformat(no_rects))
                            show_images([old_im, new_im, yes_im, no_im], 2, 2)

            assert not source_dataset.has(split, 'label', v + 1)
            if not is_equal:
                source_dataset.write_data(((key, json.dumps(rects)) for key, rects in source_key_rects),
                        split, 'label', version=v+1)
                meta_file = source_dataset.get_data(split, 'label.metadata', version=v+1) + '.yaml'
                meta['num_added_rects'] = num_added
                meta['num_removed_rects'] = num_removed
                meta['total_number_images'] = len(source_key_rects)

                meta['avg_added_rects'] = 1. * num_added / meta['total_number_images']
                meta['avg_removed_rects'] = 1. * num_removed / meta['total_number_images']
                write_to_yaml_file(meta, meta_file)
                logging.info(pformat(meta))
            else:
                logging.info('equal - {} - {}'.format(data, split))
        populate_dataset_details(data)

def merge_multi_by_key(tsv_file1, all_tsv_file2, out_tsv,
        from_flag1, all_from_flag2):
    files = [tsv_file1]
    files.extend(all_tsv_file2)
    all_key_rects = [load_key_rects(tsv_reader(f)) for f in files]
    all_key_to_rects = [{key: rects for key, rects in key_rects}
        for key_rects in all_key_rects]
    keys = [key for key, _ in all_key_rects[0]]
    flags = [from_flag1]
    flags.extend(all_from_flag2)
    def gen_rows():
        for key in keys:
            all_rects = [key_to_rects[key] for key_to_rects in all_key_to_rects if key in key_to_rects]
            for rects, f in zip(all_rects, flags):
                for r in rects:
                    r['from'] = f
            rects = all_rects[0]
            for r in all_rects[1:]:
                rects.extend(r)
            yield key, json_dump(rects)
    tsv_writer(gen_rows(), out_tsv)

def merge_by_key(tsv_file1, tsv_file2, out_tsv,
        from_flag1='', from_flag2=''):
    files = [tsv_file1, tsv_file2]
    all_key_rects = [load_key_rects(tsv_reader(f)) for f in files]
    all_key_to_rects = [{key: rects for key, rects in key_rects}
        for key_rects in all_key_rects]
    keys = [key for key, _ in all_key_rects[0]]
    flags = [from_flag1, from_flag2]
    def gen_rows():
        for key in keys:
            all_rects = [key_to_rects[key] for key_to_rects in all_key_to_rects if key in key_to_rects]
            for rects, f in zip(all_rects, flags):
                for r in rects:
                    r['from'] = f
            rects = all_rects[0]
            for r in all_rects[1:]:
                rects.extend(r)
            yield key, json_dump(rects)
    tsv_writer(gen_rows(), out_tsv)

def threshold_merged_prediction(pred_tsv_file, from_to_class_to_threshold,
        output_file):
    def gen_rows():
        for key, str_rects in tsv_reader(pred_tsv_file):
            rects = json.loads(str_rects)
            all_rect = []
            for r in rects:
                if type(from_to_class_to_threshold[r['from']]) is dict:
                    th = from_to_class_to_threshold[r['from']][r['class']]
                else:
                    th = from_to_class_to_threshold[r['from']]
                assert type(th) is float or type(th) is int
                if r['conf'] > th:
                    all_rect.append(r)
            yield key, json_dump(all_rect)
    tsv_writer(gen_rows(), output_file)

def remove_has_confirmed_colocation_labels(all_data, iou_threshold=0.75):
    all_split = get_default_splits()
    debug = False
    for data in all_data:
        dataset = TSVDataset(data)
        for split in all_split:
            if not dataset.has(split):
                continue
            logging.info('{} -> {}'.format(data, split))
            info = {'total': 0, 'removed': 0}
            def gen_rows():
                for i, (key, en_rects) in tqdm(enumerate(dataset.iter_data(split,
                    'label', version=-1))):
                    rects = json.loads(en_rects)
                    info['total'] = info['total'] + len(rects)
                    if debug:
                        _, _, im = next(dataset.iter_data(split, filter_idx=[i]))
                        im = img_from_base64(im)
                        im_origin = np.copy(im)
                        draw_bb(im_origin, [r['rect'] for r in rects], [r['class'] for r in rects])
                    all_removed = []
                    # some labels has no rect
                    good_rects = [r for r in rects if 'rect' in r and
                            ('conf' not in r or ('conf' in r and 'uhrs_confirm' in r))]
                    pro_no_verify = [r for r in rects if 'conf' in r and
                            'uhrs_confirm' not in r and 'rect' in r]
                    no_rects = [r for r in rects if 'rect' not in r]
                    assert len(good_rects) + len(pro_no_verify) + len(no_rects)==len(rects)
                    to_remove = []
                    for r in pro_no_verify:
                        if any((s for s in good_rects if
                                calculate_iou(r['rect'], s['rect']) >
                                iou_threshold)):
                            to_remove.append(r)
                    for t in to_remove:
                        rects.remove(t)
                    info['removed'] = info['removed'] + len(to_remove)
                    all_removed.extend(to_remove)
                    if debug and len(all_removed) > 0:
                        logging.info(len(all_removed))
                        im_removed = np.copy(im)
                        logging.info(pformat(all_removed))
                        draw_bb(im_removed, [r['rect'] for r in all_removed],
                                [r['class'] for r in all_removed])
                        show_images([im_origin, im_removed], 1, 2)
                    yield key, json_dump(rects)
            def gen_info():
                yield 'remove auto-pro and has good labels in same locatoin', iou_threshold
                yield 'total', info['total']
                yield 'removed', info['removed']
                ratio = 1. * info['removed'] / info['total']
                yield 'ratio', ratio
                logging.info(pformat(info))
                logging.info(ratio)

            dataset.update_data(gen_rows(), split, 'label', generate_info=gen_info())

def get_taxonomy_path(data):
    pattern = 'Tax(.*)V([0-9]*)_(.*)'
    result = re.match(pattern, data)
    assert result is not None
    major, minor, revision = result.groups()
    return './aux_data/taxonomy10k/Tax{0}/Tax{0}V{1}'.format(major, minor)

def uhrs_verify_db_merge_to_tsv(collection_name='uhrs_logo_verification',
        extra_match=None):
    set_interpretation_result_for_uhrs_result(collection_name)
    c = create_bbverification_db(collection_name=collection_name)
    data_split_to_key_rects, all_id = c.get_completed_uhrs_result(
            extra_match=extra_match)
    merge_uhrs_result_to_dataset(data_split_to_key_rects)
    size = 1000
    i = 0
    while True:
        start = i * size
        end = i * size + size
        if start >= len(all_id):
            break
        if end >= len(all_id):
            end = len(all_id)
        c.set_status_as_merged(all_id[start:end])
        i += 1
    y = [c.set_status_as_merged([i]) for i in tqdm(all_id)]
    y = list(map(lambda i: c.set_status_as_merged([i]), all_id))

def uhrs_merge_one(uhrs_rect, target_rects, tag_only=False):
    info = {'num_added': 0,
            'num_removed': 0,
            'verified_confirmed': 0,
            'verified_removed': 0,
            'non_verified_confirmed': 0,
            'non_verified_removed': 0}
    same_rect, iou = find_best_matched_rect(uhrs_rect, target_rects, check_iou=(not tag_only))
    if iou < 0.8:
        if is_positive_uhrs_verified(uhrs_rect):
            target_rects.append(uhrs_rect)
            info['num_added'] = 1
        return info

    if is_verified_rect(same_rect):
        if is_positive_uhrs_verified(uhrs_rect):
            info['verified_confirmed'] = 1
        elif is_negative_uhrs_verified(uhrs_rect):
            info['verified_removed'] = 1
            target_rects.remove(same_rect)
    else:
        if is_positive_uhrs_verified(uhrs_rect):
            info['non_verified_confirmed'] = 1
        elif is_negative_uhrs_verified(uhrs_rect):
            info['non_verified_removed'] = 1
            target_rects.remove(same_rect)

    same_rect['uhrs'] = {}
    for t, v in viewitems(uhrs_rect['uhrs']):
        same_rect['uhrs'][t] = v

    return info

def merge_uhrs_result_to_dataset(data_split_to_key_rects, tag_only=False):
    from .common import list_to_dict
    from .common import json_dump
    for (data, split), uhrs_key_rects in viewitems(data_split_to_key_rects):
        logging.info((data, split))
        dataset = TSVDataset(data)
        uhrs_key_to_rects = list_to_dict(uhrs_key_rects, 0)
        logging.info('number of image will be affected: {}'.format(len(uhrs_key_rects)))
        info = {}
        def gen_rows():
            for key, str_rects in dataset.iter_data(split, 'label', -1,
                    progress=True):
                rects = json.loads(str_rects)
                if key in uhrs_key_to_rects:
                    uhrs_rects = uhrs_key_to_rects[key]
                    del uhrs_key_to_rects[key]
                else:
                    uhrs_rects = []
                for uhrs_rect in uhrs_rects:
                    sub_info = uhrs_merge_one(uhrs_rect, rects, tag_only=tag_only)
                    for k, v in viewitems(sub_info):
                        info[k] = v + info.get(k, 0)
                yield key, json_dump(rects)
            assert len(uhrs_key_to_rects) == 0
        def generate_info():
            for k, v in viewitems(info):
                yield k, v
            for key, rects in viewitems(uhrs_key_to_rects):
                yield key, json_dump(rects)
        dataset.update_data(gen_rows(), split, 'label',
                generate_info=generate_info())

def set_interpretation_result_for_uhrs_result(collection_name='uhrs_logo_verification'):
    c = create_bbverification_db(collection_name=collection_name)
    query = {'status': {'$in': [c.status_merged, c.status_completed]},
            'interpretation_result': None}
    positive_ids, negative_ids, uncertain_ids = [], [], []
    pos, neg, un = 0, 0, 0
    for rect_info in tqdm(c.collection.find(query)):
        rect = rect_info['rect']
        rect.update({'uhrs': rect_info['uhrs_completed_result']})
        if is_positive_uhrs_verified(rect):
            positive_ids.append(rect_info['_id'])
            if len(positive_ids) > 1000:
                logging.info('positive = {}'.format(pos))
                q = {'_id': {'$in': positive_ids}}
                c.collection.update_many(filter=q,
                        update={'$set': {'interpretation_result': 1}})
                pos += len(positive_ids)
                positive_ids.clear()
        elif is_negative_uhrs_verified(rect):
            negative_ids.append(rect_info['_id'])
            if len(negative_ids) > 1000:
                logging.info('neg = {}'.format(neg))
                q = {'_id': {'$in': negative_ids}}
                c.collection.update_many(filter=q,
                        update={'$set': {'interpretation_result': -1}})
                neg += len(negative_ids)
                negative_ids.clear()
        else:
            uncertain_ids.append(rect_info['_id'])
            if len(uncertain_ids) > 1000:
                logging.info('un = {}'.format(un))
                query = {'_id': {'$in': uncertain_ids}}
                c.collection.update_many(filter=query,
                        update={'$set': {'interpretation_result': 0}})
                un += len(uncertain_ids)
                uncertain_ids.clear()


    if len(positive_ids) > 0:
        query = {'_id': {'$in': positive_ids}}
        c.collection.update_many(filter=query,
                update={'$set': {'interpretation_result': 1}})
    if len(negative_ids) > 0:
        query = {'_id': {'$in': negative_ids}}
        c.collection.update_many(filter=query,
                update={'$set': {'interpretation_result': -1}})
    if len(uncertain_ids) > 0:
        query = {'_id': {'$in': uncertain_ids}}
        c.collection.update_many(filter=query,
                update={'$set': {'interpretation_result': 0}})

def uhrs_verify_db_closest_rect(collection, test_data, test_split, gt_key, p):
    rect_infos = list(collection.find({'data': test_data,
        'split': test_split,
        'key': gt_key}))
    rects = [rect_info['rect'] for rect_info in rect_infos]
    best_idx, best_iou = find_best_matched_rect_idx(p, rects, check_class=True)

    if best_idx is not None:
        rect_info = rect_infos[best_idx]
    else:
        rect_info = None

    return rect_info, best_iou

def verify_prediction_by_db(pred_file, test_data, test_split, conf_th=0.3,
        priority_tier=1, collection_name='uhrs_bounding_box_verification'):
    from qd.process_tsv import ensure_upload_image_to_blob
    from qd.process_tsv import parse_combine
    ensure_upload_image_to_blob(test_data, test_split)

    dataset = TSVDataset(test_data)
    gt_iter = dataset.iter_data(test_split, 'label', version=-1)
    pred_iter = tsv_reader(pred_file)
    key_url_iter = dataset.iter_data(test_split, 'key.url')

    db_task = []
    num_task, num_exists, num_matched_gt, num_change_pri = 0, 0, 0, 0
    c = create_bbverification_db(collection_name=collection_name)
    for gt_row, pred_row, url_row in tqdm(zip(gt_iter, pred_iter,
        key_url_iter)):
        gt_key, gt_str_rects = gt_row
        pred_key, pred_str_rects = pred_row
        assert gt_key == pred_key == url_row[0]
        source_data, source_split, source_key = parse_combine(gt_key)
        if source_data is None and source_split is None:
            source_data = test_data
            source_split = test_split
        gt_rects = json.loads(gt_str_rects)
        pred_rects = json.loads(pred_str_rects)
        for p in pred_rects:
            if p['conf'] < conf_th:
                continue
            # check with the gt
            _, best_iou = find_best_matched_rect(p, gt_rects)
            if best_iou > 0.7:
                num_matched_gt = 0
                continue
            best_rect_info, best_iou = uhrs_verify_db_closest_rect(c.collection, source_data,
                    source_split, source_key, p)
            if best_iou > 0.95:
                num_exists = num_exists + 1
                if best_rect_info['priority_tier'] != c.urgent_priority_tier and \
                        best_rect_info['status'] == c.status_requested:
                    c.collection.update_one(
                            {'_id': best_rect_info['_id']},
                            update={'$set': {'priority_tier': c.urgent_priority_tier}})
                    num_change_pri = num_change_pri + 1
                continue
            p['from'] = pred_file
            url = url_row[1]
            task = {'url': url,
                'data': source_data,
                'split': source_split,
                'key': source_key,
                'rect': p,
                'priority_tier': priority_tier,
                'priority': 0.5}
            num_task = num_task + 1
            db_task.append(task)
            if len(db_task) > 1000:
                c.request_by_insert(db_task)
                db_task = []
    if len(db_task) > 0:
        c.request_by_insert(db_task)
        db_task = []
    logging.info('#task = {}; #exists in db = {}; #matched gt = {}; #num pri change = {}'.format(
        num_task, num_exists, num_matched_gt, num_change_pri))

def convertcomposite_to_standard(data,
        split='train', ignore_image=False, **kwargs):
    dataset = TSVDataset(data)
    if op.isfile(dataset.get_data(split, t='label')) and \
            op.isfile(dataset.get_data(split)):
        logging.info('file exists')
        return
    rows_image = dataset.iter_composite(split, t=None, version=None)
    num_image = dataset.num_rows(split)
    out_tsv = dataset.get_data(split)
    if not ignore_image:
        logging.info('writing image tsv: {}'.format(out_tsv))
        rows_label = dataset.iter_data(split, t='label', version=None)
        tsv_writer(((row_label[0], row_label[1], row_image[2]) for row_image, row_label
                in tqdm(zip(rows_image, rows_label), total=num_image)),
                out_tsv)
    if not op.isfile(dataset.get_data(split, t='label')):
        logging.info('writing image label tsv')
        rows = dataset.iter_composite(split, t='label', version=None)
        tsv_writer(tqdm(rows), dataset.get_data(split, t='label'))

def gen_merged_candidate(prediction_file,
        gts, srclabel_to_destlabel=None, **kwargs):
    # we will only consider the bounding box if the prob is larger than 0.7
    prob_th = kwargs['merge_min_prob']
    prob_th_max = kwargs['merge_max_prob']
    # we will add the bounding box only if the iou is lower than 0.7 to avoid
    # duplicate addition
    iou_th = kwargs['merge_max_iou']
    aug_labels = kwargs['aug_labels']
    prediction = tsv_reader(prediction_file)
    is_aligned = kwargs.get('is_aligned', False)
    logging.info('loading prediction: {}'.format(prediction_file))
    if not is_aligned:
        key_to_pred = {}
        for p in tqdm(prediction):
            assert len(p) == 2
            key = p[0]
            rects = json.loads(p[1])
            key_to_pred[key] = rects
    num_image = 0
    num_added = 0
    for gt in tqdm(gts):
        assert len(gt) == 2
        num_image = num_image + 1
        key = gt[0]
        g_rects = json.loads(gt[1])
        origin_g_rects = copy.deepcopy(g_rects)
        add_rects = []
        if not is_aligned:
            if key not in key_to_pred:
                yield key, origin_g_rects, add_rects, g_rects
                continue
            p_rects = key_to_pred[key]
        else:
            p_key, p_rects = next(prediction)
            p_rects = json.loads(p_rects)
            assert key == p_key
        p_rects = [r for r in p_rects if r['conf'] > prob_th and
                r['conf'] <= prob_th_max]
        if aug_labels is not None:
            p_rects = [r for r in p_rects if r['class'] in aug_labels]
        if srclabel_to_destlabel:
            p_rects = [r for r in p_rects if r['class'] in srclabel_to_destlabel]
            for r in p_rects:
                r['class'] = srclabel_to_destlabel[r['class']]
        p_rects = sorted(p_rects, key=lambda x: -x['conf'])
        for p_rect in p_rects:
            is_add = True
            ious = [calculate_iou(p_rect['rect'], g_rect['rect'])
                    for g_rect in g_rects]
            if len(ious) > 0 and max(ious) > iou_th:
                is_add = False
            if is_add:
                num_added = num_added + 1
                p_rect['from'] = prediction_file
                # add p_rect
                add_rects.append(p_rect)
                g_rects.append(p_rect)
        yield key, origin_g_rects, add_rects, g_rects
    logging.info('num_image = {}'.format(num_image))
    logging.info('num_added = {}'.format(num_added))


def merge_prediction_to_gt(prediction_file, gts, im_rows, dataset, split,
        **param):
    dataset.iter_data(split, 'label')
    gts2 = gen_merged_candidate(prediction_file, gts, **param)
    if im_rows is not None:
        if False:
            # show image
            for gt2, im_row in zip(gts2, im_rows):
                assert gt2[0] == im_row[0]
                im = img_from_base64(im_row[-1])
                origin_rects = gt2[1]
                add_rects = gt2[2]
                merged_rects = gt2[3]
                if len(add_rects) == 0:
                    continue
                im_origin_gt = np.copy(im)
                draw_bb(im_origin_gt, [r['rect'] for r in origin_rects],
                        [r['class'] for r in origin_rects])
                im_add = np.copy(im)
                draw_bb(im_add, [r['rect'] for r in add_rects],
                        [r['class'] for r in add_rects],
                        [r['conf'] for r in add_rects])
                im_merged = np.copy(im)
                draw_bb(im_merged, [r['rect'] for r in merged_rects],
                        [r['class'] for r in merged_rects])
                logging.info(pformat(add_rects))
                show_images([im_origin_gt, im_add, im_merged], 1, 3)
        else:
            # save image
            for i, (gt2, im_row) in enumerate(zip(gts2, im_rows)):
                assert gt2[0] == im_row[0]
                im = img_from_base64(im_row[-1])
                origin_rects = gt2[1]
                add_rects = gt2[2]
                merged_rects = gt2[3]
                if len(add_rects) == 0:
                    continue
                add_class = list(set([r['class'] for r in add_rects]))
                for c in add_class:
                    c_rects = [r for r in add_rects if r['class'] == c]
                    im_add = np.copy(im)
                    draw_rects(c_rects, im_add, add_label=False)
                    im_origin = np.copy(im)
                    draw_rects([x for x in origin_rects if x['class'] == c], im_origin)
                    im_merged = np.copy(im)
                    draw_rects([x for x in merged_rects if x['class'] == c], im_merged)
                    top = np.concatenate((im, im_add), 1)
                    bottom = np.concatenate((im_origin, im_merged), 1)
                    x = np.concatenate((top, bottom), 0)
                    save_image(x, './output/add_image/{}_{}.jpg'.format(
                        c, im_row[0]))
    else:
        meta = {'num_image': 0,
                'num_added_total': 0,
                'from': prediction_file}
        all_added = []
        def gen_rows():
            for key, origin_rects, add_rects, merged_rects in gts2:
                all_added.append((key, add_rects))
                meta['num_added_total'] = meta['num_added_total']+len(add_rects)
                meta['num_image'] = meta['num_image'] + 1
                yield key, json_dump(merged_rects)
        # we will not change the files if they are the same
        def gen_info():
            meta['num_added_each'] = 1. * meta['num_added_total'] / meta['num_image']
            for k, v in viewitems(meta):
                yield k, v
            for k, rects in all_added:
                yield k, json_dump(rects)
        dataset.update_data(gen_rows(), split, 'label',
                generate_info=gen_info())

def generate_labels_to_verify(predict_file, data, split, th,
        pred_to_gt, output_file):
    dataset = TSVDataset(data)
    if op.isfile(op.join(dataset._data_root, 'root.yaml')):
        tax = Taxonomy(load_from_yaml_file(op.join(dataset._data_root, 'root.yaml')))
    else:
        tax = None
    gt_key_rects = [(row[0], json.loads(row[1])) for row in tqdm(dataset.iter_data(split,
        'label', version=-1))]
    logging.info('loading pred')
    pred_key_to_rects = {}
    for row in tqdm(tsv_reader(predict_file)):
        key = row[0]
        rects = json.loads(row[1])
        pred_key_to_rects[key] = [r for r in rects if r['conf'] > th]

    def gen_rows(pred_to_gt=True):
        logging.info('start to writting')
        for key, gt_rects in tqdm(gt_key_rects):
            if tax is not None:
                rects2 = lift_one_image(gt_rects, tax)
                del gt_rects[:]
                gt_rects.extend(rects2)
            pred_rects = pred_key_to_rects.get(key)
            if pred_rects is None:
                continue
            need_confirm = []
            if pred_to_gt:
                for pr in pred_rects:
                    if not onebb_in_list_of_bb(pr, gt_rects):
                        need_confirm.append(pr)
            else:
                for g in gt_rects:
                    if len(g) == 2:
                        assert 'class' in g and 'rect' in g
                        continue
                    if not onebb_in_list_of_bb(g, pred_rects):
                        need_confirm.append(g)
            yield key, json.dumps(need_confirm)

    tsv_writer(gen_rows(pred_to_gt=pred_to_gt), output_file)

def onebb_in_list_of_bb(bb, bbs):
    bbs = [b for b in bbs if b['class'] == bb['class']]
    return any(calculate_iou(b['rect'], bb['rect']) > 0.5 for b in bbs)

def inject_accuracy():
    all_full_expid = os.listdir('./output')
    for full_expid in tqdm(all_full_expid):
        inject_accuracy_one(full_expid)

def inject_accuracy_one(full_expid):
    from qd.db import create_annotation_db
    c = create_annotation_db()
    all_predict = glob.glob(op.join('output', full_expid, 'snapshot',
        '*.predict'))
    all_predict.extend(glob.glob(op.join('output', full_expid, 'snapshot',
        '*.predict.tsv')))

    all_report = glob.glob(op.join('output', full_expid, 'snapshot', '*.report'))
    for report_file in all_report:
        from .common import parse_test_data_with_version
        try:
            test_data, test_split, test_version = parse_test_data_with_version(op.basename(report_file))
        except:
            continue
        info = {'full_expid': full_expid,
                'report_file': op.basename(report_file),
                'test_data': test_data,
                'test_split': test_split,
                'test_version': test_version,
                }
        if 'coco_box' in report_file or '.randomness.' in report_file:
            acc = load_from_yaml_file(report_file)
        elif '.neg_aware_gmap.' in report_file:
            acc = load_from_yaml_file(report_file)
            if 'neg_aware_gmap.noNMSGt.noNMSDet.noExpandDet.report' in report_file:
                key = 'nngmAP'
            elif '.neg_aware_gmap.noNMSGt.report' in report_file:
                key = 'ngmAP'
            else:
                key = 'gmAP'
            acc = {key: acc['map']}
        elif '.top1.' in report_file:
            acc = load_from_yaml_file(report_file)
        elif report_file.endswith('predict.tsv.speed.yaml'):
            all_meter = load_from_yaml_file(report_file)['meters']
            all_meter = [m for m in all_meter if m['name'] == '']
            assert len(all_meter) == 1
            acc = all_meter[0]
            del acc['name']
        elif report_file.endswith('caption.report'):
            acc = json.loads(read_to_buffer(report_file))
        elif any(report_file.endswith(k) for k in [
            'ir_acc.report',
            'cap_acc.report',
            'cap_acc.nospace.report',
            'vqa_acc.report',
            'vqa2.report',
            'stvqa_acc.report',
            'stvqa_nopro_acc',
            'stvqa_anls.report',
            'nlvr2.report',
            'acc.report',
            'vizwiz_vqa.report',
            'textvqa_acc.report',
        ]):
            acc = load_from_yaml_file(report_file)
            from .common import get_all_path, dict_get_path_value
            all_path = get_all_path(acc)
            acc = {p: dict_get_path_value(acc, p) for p in all_path}
        elif report_file.endswith('attr.report'):
            x = json.loads(read_to_buffer(report_file))
            acc = {'attr': x['overall']['0.5']['map']}
        else:
            map_json_file = report_file + '.map.json'
            if op.isfile(map_json_file):
                x = json.loads(read_to_buffer(map_json_file))
                acc = {}
                for k1 in x:
                    for k2 in x[k1]:
                        for k3 in x[k1][k2]:
                            k = '{}${}${}'.format(k1, k2, k3)
                            acc[k] = x[k1][k2][k3]
            else:
                continue
        for k in list(acc.keys()):
            curr = copy.deepcopy(info)
            curr['metric_name'] = k
            exist = False
            for found in c.iter_acc(**curr):
                exist = True
                if found['metric_value'] == acc[k]:
                    continue
                else:
                    # the newer one will overwrite the old one. Sometimes, the
                    # code has issues and the experiment results are not
                    # correct
                    c.update_one_acc(query=curr,
                            update={'$set': {'metric_value': acc[k]}})
            if not exist:
                curr['metric_value'] = acc[k]
                c.insert_acc(**curr)

def find_predict_file(report_file, all_predict):
    for k in ['.ir_acc.report',
              '.vqa_acc.report',
              '.stvqa_acc.report',
              '.stvqa_anls.report',
              '.caption.report',
              '.attr.report'
              ]:
        if report_file.endswith(k):
            pred = report_file[:-len(k)] + '.tsv'
            if pred in all_predict:
                return pred
    #model_iter_0200182.pt.TaxImageNet2012CapDef.test.max_token100.vphoto_def.predict.top1.vphoto_def.report
    #model_iter_0200182.pt.TaxImageNet2012CapDef.test.max_token100.vphoto_def.predict.tsv
    r_pattern = '.*\.(v[^\.]*)\..*(v[^\.]*)\..*'
    result = re.match(r_pattern, op.basename(report_file))
    if result is not None:
        v1, v2 = result.groups()
        if v1 == v2:
           pattern = '(.*)\.top1\.{}\.report'.format(v1)
           result = re.match(pattern, op.basename(report_file))
           if result is not None:
               prefix, = result.groups()
               p = prefix + '.tsv'
               p = op.join(op.dirname(report_file), p)
               for p1 in all_predict:
                   if p1 == p:
                       return p


    found = False
    for p in all_predict:
        if p + '.report' == report_file:
            assert not found
            found = True
            result = p
            return result
    for p in all_predict:
        if p.endswith('.predict.tsv'):
            ps = p[: -len('.predict.tsv')]
        elif p.endswith('.predict'):
            ps = p[: -len('.predict')]
        else:
            continue
        if report_file.startswith(ps):
            rs = report_file[len(ps):]
            logging.info(rs)
            eval_keys = ['predict', 'tsv', 'coco_box', 'neg_aware_gmap', 'top1',
                    'noNMSGt', 'noNMSDet', 'noExpandDet', 'speed', 'MaxDet.*']
            p0 = ''.join(['(\.{})?'.format(k) for k in eval_keys])
            pattern = '{}(\.v[0-9]*)?(\.0)?\.(report|yaml)'.format(p0)
            if re.match(pattern, rs):
                assert not found
                found = True
                result = p
    if not found:
        matched = [p for p in all_predict if report_file.startswith(p)]
        if len(matched) > 0:
            result = sorted(matched, key=lambda x: -len(x))[0]
            found = True
    assert found
    return result

def softnms_row_process(row, sigma=0.5, method=2, Nt=0.5):
    from .common import softnms_c
    key, str_rects = row
    rects = json.loads(str_rects)
    all_class_rect = [(r['class'], r) for r in rects]
    class_to_rects = list_to_dict(all_class_rect, 0)
    rects2 = []
    for c, rs in class_to_rects.items():
        rs = softnms_c(rs, sigma=sigma, method=method, Nt=Nt)
        for r in rs:
            r['class'] = c
        rects2.extend(rs)
    return key, json_dump(rects2)

def tsv_subset_process_NtoN(info):
    row_processor = info['row_processor']
    idx_process = info['idx_process']
    tmp_out = info['tmp_out']
    idx_range_start = info['idx_range_start']
    idx_range_end = info['idx_range_end']
    head = info['head']
    sep = info['out_sep']
    if all(op.isfile(t) and all(op.isfile(s) for s in get_tsv_associates(t)) for t in tmp_out):
        logging.info('skip since exist: {}'.format(tmp_out))
        return
    if 'in_tsvs' in info:
        tsvs = info['in_tsvs']
    else:
        tsvs = [TSVFile(t) for t in info['in_tsv_files']]
    def gen_rows():
        if idx_process == 0 and head is not None:
            yield head
        for i in tqdm(range(idx_range_start, idx_range_end)):
            r = row_processor([t[i] for t in tsvs])
            if r is None:
                continue
            yield r
    tsv_writers(gen_rows(), tmp_out, sep=sep)
    logging.info('done to generate {}'.format(tmp_out))

def tsv_subset_process_1toN(info, with_row_idx):
    row_processor = info['row_processor']
    idx_process = info['idx_process']
    tmp_out = info['tmp_out']
    idx_range_start = info['idx_range_start']
    idx_range_end = info['idx_range_end']
    head = info['head']
    sep = info['out_sep']
    if all(op.isfile(t) for t in tmp_out):
        logging.info('skip since exist: {}'.format(tmp_out))
        return
    if 'in_tsv' in info:
        tsv = info['in_tsv']
    else:
        in_tsv_file = info['in_tsv_file']
        tsv = TSVFile(in_tsv_file)
    def gen_rows():
        if idx_process == 0 and head is not None:
            yield head
        for i in tqdm(range(idx_range_start, idx_range_end)):
            try:
                if with_row_idx:
                    r = row_processor(tsv[i], i)
                else:
                    r = row_processor(tsv[i])
                if r is None:
                    continue
                if isinstance(r, (list, tuple)):
                    yield r
                else:
                    while True:
                        try:
                            x = next(r)
                            yield x
                        except StopIteration:
                            break
            except:
                logging.info(f'{tsv}-{i}')
                raise
    tsv_writers(gen_rows(), tmp_out, sep=sep)

def tsv_subset_process(info, with_row_idx=False):
    row_processor = info['row_processor']
    idx_process = info['idx_process']
    tmp_out = info['tmp_out']
    idx_range_start = info['idx_range_start']
    idx_range_end = info['idx_range_end']
    head = info['head']
    sep = info['out_sep']
    if op.isfile(tmp_out):
        logging.info('skip since exist: {}'.format(tmp_out))
        return
    if 'in_tsv' in info:
        tsv = info['in_tsv']
    else:
        in_tsv_file = info['in_tsv_file']
        tsv = TSVFile(in_tsv_file)
    def gen_rows():
        if idx_process == 0 and head is not None:
            yield head
        for i in tqdm(range(idx_range_start, idx_range_end)):
            # this is a breaking change to add i as option, but we have to do
            # this as row_processor has no way to figure it out which row it is
            try:
                if with_row_idx:
                    r = row_processor(tsv[i], i)
                else:
                    r = row_processor(tsv[i])
                if r is None:
                    continue
                if isinstance(r, (list, tuple)):
                    yield r
                else:
                    while True:
                        try:
                            x = next(r)
                            yield x
                        except StopIteration:
                            break
            except:
                logging.info(f'{tsv}/{i}')
                raise
    tsv_writer(gen_rows(), tmp_out, sep=sep)

def tsv_subset_aggregator(info, with_row_idx=False):
    row_processor = info['row_processor']
    tmp_out = info['tmp_out']
    idx_range_start = info['idx_range_start']
    idx_range_end = info['idx_range_end']
    if File.isfile(tmp_out):
        return load_from_yaml_file(tmp_out)
    if 'in_tsv' in info:
        tsv = info['in_tsv']
    else:
        in_tsv_file = info['in_tsv_file']
        tsv = TSVFile(in_tsv_file)
    ret = {}
    for i in tqdm(range(idx_range_start, idx_range_end)):
        # this is a breaking change to add i as option, but we have to do
        # this as row_processor has no way to figure it out which row it is
        try:
            if with_row_idx:
                row_processor(ret, tsv[i], i)
            else:
                row_processor(ret, tsv[i])
        except:
            logging.info(f'{tsv}/{i}')
            raise
    write_to_yaml_file(ret, tmp_out)
    return ret

def multi_tsv_subset_process(info):
    row_processor = info['row_processor']
    idx_process = info['idx_process']
    in_tsv_files = info['in_tsv_files']
    tmp_out = info['tmp_out']
    idx_range = info['idx_range']
    head = info['head']
    out_sep = info['out_sep']
    if op.isfile(tmp_out):
        logging.info('skip to create {}'.format(tmp_out))
        return
    def gen_rows():
        if idx_process == 0 and head is not None:
            yield head
        tsvs = [TSVFile(f) for f in in_tsv_files]
        for i in tqdm(idx_range):
            row_result = row_processor([tsv[i] for tsv in tsvs])
            if row_result is not None:
                # we can return None to remove rows
                yield row_result
    tsv_writer(gen_rows(), tmp_out, sep=out_sep)

def multi_tsv_row_merger(rows):
    all_key = [row[0] for row in rows]
    assert all(all_key[0] == key for key in all_key[1:])
    all_rect = [json.loads(row[1]) for row in rows]
    str_rect = json_dump([r for rs in all_rect for r in rs])
    return all_key[0], str_rect

def parallel_tsv_process_NtoN(row_processor, in_tsv_files,
        out_tsv_files, num_process, num_jobs=None, head=None, out_sep='\t'):
    in_tsvs = [TSVFile(in_tsv_file) if isinstance(in_tsv_file, str) else
               in_tsv_file for in_tsv_file in in_tsv_files]
    total = len(in_tsvs[0])
    assert all(len(t) == total for t in in_tsvs[1:])
    for t in in_tsvs:
        t.close()
    if num_jobs is None:
        num_jobs = max(1, num_process)
    rows_each_rank = (total + num_jobs - 1) // num_jobs
    all_task = []
    folder = get_tmp_folder()
    for i in range(num_jobs):
        start = i * rows_each_rank
        end = start + rows_each_rank
        end = min(end, total)
        tmp_out = ['{}/{}'.format(folder, out_tsv_file) + '.{}.{}.tsv'.format(i, num_jobs)
            for out_tsv_file in out_tsv_files]
        info = {
            'row_processor': row_processor,
            'idx_process': i,
            #'in_tsv_files': in_tsv_files,
            'in_tsvs': in_tsvs,
            'tmp_out': tmp_out,
            'idx_range_start': start,
            'idx_range_end': end,
            'head': head,
            'out_sep': out_sep,
        }
        all_task.append(info)
    from .common import parallel_map
    parallel_map(tsv_subset_process_NtoN, all_task,
                 num_worker=num_process)
    all_out = [task['tmp_out'] for task in all_task]
    for i, out_tsv_file in enumerate(out_tsv_files):
        tmp_out = [out[i] for out in all_out]
        concat_tsv_files(tmp_out, out_tsv_file, gen_lineidx=False)
        delete_tsv_files(tmp_out)

def parallel_multi_tsv_process(row_processor, in_tsv_files,
        out_tsv_file, num_process, num_jobs=None, head=None, out_sep='\t'):
    in_tsvs = [TSVFile(in_tsv_file) for in_tsv_file in in_tsv_files]
    total = len(in_tsvs[0])
    assert all(len(t) == total for t in in_tsvs[1:])
    if num_jobs is None:
        num_jobs = max(1, num_process)
    rows_each_rank = (total + num_jobs - 1) // num_jobs
    all_task = []
    for i in range(num_jobs):
        start = i * rows_each_rank
        end = start + rows_each_rank
        end = min(end, total)
        if num_jobs == 1:
            tmp_out = out_tsv_file
        else:
            tmp_out = out_tsv_file + '.{}.{}.tsv'.format(i, num_jobs)
        info = {'row_processor': row_processor,
                'idx_process': i,
                'in_tsv_files': in_tsv_files,
                'tmp_out': tmp_out,
                'idx_range': list(range(start, end)),
                'head': head,
                'out_sep': out_sep
                }
        all_task.append(info)
    from .common import parallel_map
    parallel_map(multi_tsv_subset_process, all_task,
            num_worker=num_process)
    if num_jobs > 1:
        all_out = [task['tmp_out'] for task in all_task]
        concat_tsv_files(all_out, out_tsv_file, gen_lineidx=False)
        delete_tsv_files(all_out)

def parallel_tsv_process_1toN(row_processor, in_tsv_file,
                              out_tsv_files, num_process, num_jobs=None, head=None, out_sep='\t',
                              with_row_idx=False,
                              ):
    # in_tsv_file can be a string, interpreted as a tsv file
    # or a list
    if isinstance(in_tsv_file, str):
        in_tsv = TSVFile(in_tsv_file)
    else:
        in_tsv = in_tsv_file
    total = len(in_tsv)
    if num_jobs is None:
        if num_process == 0:
            num_jobs = 1
        else:
            num_jobs = num_process
    rows_each_rank = (total + num_jobs - 1) // num_jobs
    all_task = []
    if isinstance(in_tsv, (TSVFile, CompositeTSVFile)):
        # we need to clear all the cache in TSVFile. otherwise, the process
        # might need a lot of time to copy the cache in in_tsv, e.g. lineidx
        # when the number of data are huge.
        in_tsv.close()
    folder = get_tmp_folder()
    for i in range(num_jobs):
        start = i * rows_each_rank
        end = start + rows_each_rank
        end = min(end, total)
        tmp_out = ['{}/{}'.format(folder, out_tsv_file) + '.{}.{}.tsv'.format(i, num_jobs)
            for out_tsv_file in out_tsv_files]
        info = {'row_processor': row_processor,
                'idx_process': i,
                #'in_tsv_file': in_tsv_file,
                'in_tsv': in_tsv,
                'tmp_out': tmp_out,
                'idx_range_start': start,
                'idx_range_end': end,
                'head': head,
                'out_sep': out_sep
                }
        all_task.append(info)
    from .common import parallel_map
    parallel_map(lambda x: tsv_subset_process_1toN(x, with_row_idx), all_task,
            num_worker=num_process)
    all_out = [task['tmp_out'] for task in all_task]
    if isinstance(in_tsv, TSVFile):
        # explicitly close the file
        in_tsv.close()
    for i, out_tsv_file in enumerate(out_tsv_files):
        tmp_out = [out[i] for out in all_out]
        concat_tsv_files(tmp_out, out_tsv_file, gen_lineidx=False)
        delete_tsv_files(tmp_out)

def parallel_tsv_process(row_processor, in_tsv_file,
                         out_tsv_file,
                         num_process=64,
                         num_jobs=None,
                         head=None,
                         out_sep='\t',
                         with_row_idx=False,
                         ):
    if isinstance(in_tsv_file, str):
        in_tsv = TSVFile(in_tsv_file)
    else:
        in_tsv = in_tsv_file
    total = len(in_tsv)
    if num_jobs is None:
        if num_process == 0:
            num_jobs = 1
        else:
            num_jobs = num_process
    rows_each_rank = (total + num_jobs - 1) // num_jobs
    all_task = []
    if isinstance(in_tsv, TSVFile):
        # we need to clear all the cache in TSVFile. otherwise, the process
        # might need a lot of time to copy the cache in in_tsv, e.g. lineidx
        # when the number of data are huge.
        in_tsv.close()
    folder = get_tmp_folder()
    for i in range(num_jobs):
        start = i * rows_each_rank
        end = start + rows_each_rank
        end = min(end, total)
        #tmp_out = out_tsv_file + '.{}.{}.tsv'.format(i, num_jobs)
        tmp_out = '{}/{}'.format(folder, out_tsv_file) + '.{}.{}.tsv'.format(i, num_jobs)
        info = {'row_processor': row_processor,
                'idx_process': i,
                #'in_tsv_file': in_tsv_file,
                'in_tsv': in_tsv,
                'tmp_out': tmp_out,
                'idx_range_start': start,
                'idx_range_end': end,
                'head': head,
                'out_sep': out_sep
                }
        all_task.append(info)
    from .common import parallel_map
    parallel_map(lambda x: tsv_subset_process(x, with_row_idx), all_task,
            num_worker=num_process)
    all_out = [task['tmp_out'] for task in all_task]
    concat_tsv_files(all_out, out_tsv_file, gen_lineidx=False)
    delete_tsv_files(all_out)

def parallel_tsv_aggregator(
    mapper,
    reducer,
    in_tsv_file,
    num_process=64,
    num_jobs=None,
    head=None,
    out_sep='\t',
    with_row_idx=False,
):
    if isinstance(in_tsv_file, str):
        in_tsv = TSVFile(in_tsv_file)
    else:
        in_tsv = in_tsv_file
    total = len(in_tsv)
    if num_jobs is None:
        if num_process == 0:
            num_jobs = 1
        else:
            num_jobs = num_process
    rows_each_rank = (total + num_jobs - 1) // num_jobs
    all_task = []
    if isinstance(in_tsv, TSVFile):
        # we need to clear all the cache in TSVFile. otherwise, the process
        # might need a lot of time to copy the cache in in_tsv, e.g. lineidx
        # when the number of data are huge.
        in_tsv.close()
    folder = get_tmp_folder()
    basename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + str(hash(in_tsv_file))
    for i in range(num_jobs):
        start = i * rows_each_rank
        end = start + rows_each_rank
        end = min(end, total)
        #tmp_out = out_tsv_file + '.{}.{}.tsv'.format(i, num_jobs)
        tmp_out = '{}/{}'.format(folder, basename) + '.{}.{}.yaml'.format(i, num_jobs)
        info = {'row_processor': mapper,
                'idx_process': i,
                #'in_tsv_file': in_tsv_file,
                'in_tsv': in_tsv,
                'tmp_out': tmp_out,
                'idx_range_start': start,
                'idx_range_end': end,
                'head': head,
                'out_sep': out_sep
                }
        all_task.append(info)
    from .common import parallel_map
    rets = parallel_map(lambda x: tsv_subset_aggregator(x, with_row_idx), all_task,
            num_worker=num_process)
    rets = reducer(rets)
    all_out = [task['tmp_out'] for task in all_task]
    for f in all_out:
        ensure_remove_file(f)
    return rets

def mpi_tsv_process(row_processor, in_tsv_file, out_tsv_file):
    mpi_size = get_mpi_size()
    mpi_rank = get_mpi_rank()
    in_tsv = TSVFile(in_tsv_file)
    total = len(in_tsv)
    rows_each_rank = (total + mpi_size - 1) // mpi_size
    logging.info('size for each rank = {}'.format(rows_each_rank))
    start = mpi_rank * rows_each_rank
    end = min(start + rows_each_rank, total)
    logging.info('rank = {}; start = {}; end = {}'.format(mpi_rank,
        start, end))
    out_tsv_rank = out_tsv_file + '.{}.{}.tsv'.format(mpi_rank, mpi_size)
    def gen_rows():
        for i in tqdm(range(start, end)):
            row = in_tsv[i]
            row = row_processor(row)
            yield row
    tsv_writer(gen_rows(), out_tsv_rank)

    all_out = [out_tsv_file + '.{}.{}.tsv'.format(i, mpi_size) for i in
            range(mpi_size)]

    while True:
        if all(op.isfile(f) for f in all_out):
            break
        logging.info('waiting {}'.format('; '.join(f for f in all_out if not
            op.isfile(f))))
        time.sleep(5)

    if mpi_rank == 0:
        concat_tsv_files(all_out, out_tsv_file)
        delete_tsv_files(all_out)
    else:
        while not op.isfile(out_tsv_file):
            logging.info('waiting {}'.format(out_tsv_file))
            time.sleep(5)

def create_focus_dataset(data, source_version, target_labels, out_data):
    out_dataset = TSVDataset(out_data)
    write_to_file('\n'.join(target_labels),
            out_dataset.get_labelmap_file())

    # train
    dataset = TSVDataset(data)
    write_to_file(dataset.get_data('train'), out_dataset.get_data('trainX'))
    write_to_file(dataset.get_data('train', t='hw'),
            out_dataset.get_data('trainX', t='hw'))

    def gen_rows():
        target_label_set = set(target_labels)
        for key, str_rects in dataset.iter_data('train', t='label',
                version=source_version, progress=True):
            rects = json.loads(str_rects)
            rects = [r for r in rects if r['class'] in target_label_set]
            yield '_'.join([data, 'train', key]), json_dump(rects)
    out_dataset.write_data(gen_rows(), 'train', t='label')
    num_rows = dataset.num_rows('train')
    tsv_writer(((0, i) for i in range(num_rows)),
            op.join(out_dataset._data_root, 'train.shuffle.txt'))

class CogAPI(object):
    def __init__(self):
        self.remote_image_features = ['adult']
        self.run_ocr = False
        self.computervision_client = None

    def ensure_init(self):
        from msrest.authentication import CognitiveServicesCredentials
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        if self.computervision_client is None:
            x = load_from_yaml_file('aux_data/configs/cognitive_credential.yaml')
            self.subscription_key = x['subscription_key']
            self.endpoint = x['endpoint']
            self.computervision_client = ComputerVisionClient(self.endpoint,
                    CognitiveServicesCredentials(self.subscription_key))

    def call(self, image_url):
        self.ensure_init()
        result = {}
        if self.remote_image_features:
            remote_image_analysis = self.computervision_client.analyze_image(image_url,
                    self.remote_image_features)
            from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
            if 'adult' in self.remote_image_features:
                result.update(self.parse_adult(remote_image_analysis))
            if 'description' in self.remote_image_features:
                result.update(self.parse_description(remote_image_analysis))
            if VisualFeatureTypes.image_type in self.remote_image_features:
                result.update(self.parse_type(remote_image_analysis))
        if self.run_ocr:
            recognize_printed_results = self.computervision_client.batch_read_file(image_url,  raw=True)
            result.update(self.parse_ocr(recognize_printed_results))

        return result

    def parse_ocr(self, recognize_printed_results):
        from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
        # Get the operation location (URL with an ID at the end) from the response
        operation_location_remote = recognize_printed_results.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = operation_location_remote.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results 
        while True:
            get_printed_text_results = self.computervision_client.get_read_operation_result(operation_id)
            if get_printed_text_results.status not in ['NotStarted', 'Running']:
                break
            time.sleep(1)

        lines = []
        if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
            for text_result in get_printed_text_results.recognition_results:
                for line in text_result.lines:
                    lines.append({'text': line.text, 'bounding_box':
                        line.bounding_box})
        return {'ocr': lines}

    def parse_description(self, result):
        return {'captions': [{'text': cap.text,
                              'conf': cap.confidence}
                              for cap in result.description.captions]}

    def parse_type(self, remote_image_analysis):
        return {'clip_art_type': remote_image_analysis.image_type.clip_art_type,
                'line_drawing_type': remote_image_analysis.image_type.line_drawing_type,
                }

    def parse_adult(self, remote_image_analysis):
        result = {'is_adult': remote_image_analysis.adult.is_adult_content,
                'adult_score': remote_image_analysis.adult.adult_score,
                'is_racy': remote_image_analysis.adult.is_racy_content,
                'racy_score': remote_image_analysis.adult.racy_score}
        return result

def adult_filtering_by_cogapi(data, split):
    dataset = TSVDataset(data)
    api = CogAPI()
    in_tsv = dataset.get_data(split, 'key.url')
    out_tsv = dataset.get_data(split, 'key.adult')
    def row_processor(row):
        key, url = row
        try:
            result = api.call(url)
            return key, json_dump(result)
        except:
            from .common import print_trace
            print_trace()
            return key, json_dump(-1)
    if op.isfile(out_tsv):
        return
    parallel_tsv_process(row_processor, in_tsv,
            out_tsv, num_process=16, num_jobs=1024)

def scale_ocr_result(result, h_scale, w_scale):
    for r in result:
        r['rect'][::2] = [x * w_scale for x in r['rect'][::2]]
        r['rect'][1::2] = [x * h_scale for x in r['rect'][1::2]]
        r['lines'][::2] = [x * w_scale for x in r['lines'][::2]]
        r['lines'][1::2] = [x * h_scale for x in r['lines'][1::2]]

def ocr_engine_result_to_rects(lines):
    rects = []
    for info in lines:
        r = {}
        xs = info['boundingBox'][::2]
        ys = info['boundingBox'][1::2]
        r['rect'] = [min(xs), min(ys), max(xs), max(ys)]
        r['class'] = 'text'
        r['lines'] = info['boundingBox']
        r['text'] = info['text']
        rects.append(r)
    return rects

def run_ocr_on_content(input_file, urls):
    headers = {
        'Content-Type':'image/jpg'
    }
    url = urls[int(random.random() * 9999) % len(urls)]
    import requests
    r = requests.post(url, headers=headers, data=input_file)

    if (r.ok):
        x = json.loads(r.text)
        if 'analyzeResult' not in x or 'readResults' not in x['analyzeResult']:
            return []
        rects = ocr_engine_result_to_rects(x['analyzeResult']['readResults'][0]['lines'])
        return rects

class OCRRowProcessor(object):
    def __init__(self, urls):
        self.urls = urls

    def __call__(self, row):
        if len(row) == 3:
            key, str_rects, str_im = row
        else:
            key, str_im = row
        im = img_from_base64(str_im)
        def proper_image_scale(im):
            h, w = im.shape[:2]
            if h <= w:
                ratio = 800. / h
                w = ratio * w
                h = 800
                if w > 8000:
                    w = 8000
            else:
                ratio = 800. / w
                w = 800
                h = ratio * h
                if h > 8000:
                    h = 8000
            im = cv2.resize(im, (int(w), int(h)))
            return im

        if max(im.shape[:2]) <= 200 or max(im.shape[:2]) > 8000:        
            im_scale = proper_image_scale(im)
            h_scale, w_scale = (1. * im_scale.shape[0] / im.shape[0], 1. *
                    im_scale.shape[1] / im.shape[1])
        else:
            h_scale, w_scale = 1, 1
            im_scale = im

        content = cv2.imencode('.jpg', im_scale)[1]
        try:
            result = run_ocr_on_content(content.tobytes(), self.urls)
        except:
            logging.info(pformat(im_scale.shape))
            raise
        if result is None:
            logging.info('unable to process {}'.format(key))
            return key, -1
        else:
            scale_ocr_result(result, 1./h_scale, 1./w_scale)
            return key, json_dump(result)

def ocr_tsv(in_tsv, out_tsv, urls):
    processor = OCRRowProcessor(urls)
    parallel_tsv_process(processor, in_tsv, out_tsv,
            num_process=5, num_jobs=10240)

def create_dataset_from_pred(pred_tsv, conf_th, src_data, out_data):
    # create label
    def gen_rows():
        for key, str_rects in tqdm(tsv_reader(pred_tsv)):
            rects = json.loads(str_rects)
            high_rects = [r for r in rects if r['conf'] > conf_th]
            if len(high_rects) > 0:
                yield '_'.join([src_data, 'train', key]), json_dump(high_rects)
    tsv_writer(gen_rows(), './data/{}/train.label.tsv'.format(out_data))

    create_image_X_by_key(out_data, 'train', 'label')
    ensure_copy_file(TSVDataset('coco2017Full').get_labelmap_file(),
        TSVDataset(out_data).get_labelmap_file())
    populate_dataset_details(out_data)
    populate_dataset_hw(out_data, ['train'])

def create_image_X_by_key(target_data, target_split, t):
    # given train.label.tsv where key is a composed key, we derive the
    # trainX.tsv and the shuffle file
    dataset = TSVDataset(target_data)
    data_split_keys = [parse_combine(key) for key, _ in
            dataset.iter_data(target_split, t)]
    data_to_split_keys = list_to_dict(data_split_keys, 0)
    data_splits = []
    for data, split_keys in data_to_split_keys.items():
        split_to_keys = list_to_dict(split_keys, 0)
        for split in split_to_keys:
            data_splits.append((data, split))
    data_to_dataset = {}
    def get_dataset(src_data):
        if src_data not in data_to_dataset:
            data_to_dataset[src_data] = TSVDataset(src_data)
        return data_to_dataset[src_data]
    src_tsv = [get_dataset(data).get_data(split) for data, split in data_splits]
    data_split_to_idx = {data_split: i for i, data_split in
            enumerate(data_splits)}
    def gen_shuffle():
        for src_data, src_split, src_key in data_split_keys:
            idx_source = data_split_to_idx[(src_data, src_split)]
            idx_key = get_dataset(src_data).get_idx_by_key(src_key,
                    src_split)
            yield idx_source, idx_key

    tsv_writer(gen_shuffle(),
            get_dataset(target_data).get_shuffle_file(target_split))
    write_to_file('\n'.join(src_tsv),
            get_dataset(target_data).get_data(target_split + 'X'))

def create_tiny_set(data, out_data):
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)

    num_image = dataset.num_rows('train')
    all_idx = list(range(num_image))
    random.shuffle(all_idx)

    out_num_train = int(0.1 * len(all_idx))
    out_num_test = int(0.05 * len(all_idx))
    train_idx = all_idx[:out_num_train]
    test_idx = all_idx[out_num_train: (out_num_train + out_num_test)]

    ensure_copy_file(dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file())

    for t in [None, 'label']:
        out_dataset.write_data(dataset.iter_data('train',
                                                 t=t,
                                                 filter_idx=train_idx,
                                                 progress=True),
                               split='train',
                               t=t)
        out_dataset.write_data(dataset.iter_data('train',
                                                 t=t,
                                                 filter_idx=test_idx,
                                                 progress=True),
                               split='test',
                               t=t)

def create_tsv_by_image_folder(folder, tsv_file,
        max_short=1200, key_func=None):
    from .common import is_valid_image
    from .common import insensitive_glob
    from .process_image import load_image
    if key_func is None:
        key_func = lambda f: op.basename(f)
    def gen_rows():
        for ext in ['png', 'jpg', 'jpeg', 'JPG']:
            pattern = '*.{}'.format(ext)
            pattern = op.join(folder, pattern)
            im_files = insensitive_glob(pattern)
            for f in tqdm(im_files):
                im = load_image(f)
                if not is_valid_image(im):
                    continue
                if min(im.shape[:2]) > max_short:
                    ratio = 1. * max_short / min(im.shape[:2])
                    h = int(im.shape[0] * ratio)
                    w = int(im.shape[1] * ratio)
                    im = cv2.resize(im, (w, h))
                    yield key_func(f), encoded_from_img(im)
                else:
                    yield key_func(f), base64.b64encode(read_to_buffer(f))
    tsv_writer(gen_rows(), tsv_file)

def create_versions_from_uncertain_label(data, split):
    dataset = TSVDataset(data)
    info = defaultdict(int)
    assert not dataset.has(split, 'label', version=1)
    assert not dataset.has(split, 'label', version=2)
    def gen_rows():
        iter_label = dataset.iter_data(split, 'label')
        for key, str_rects in tqdm(iter_label):
            rects = json.loads(str_rects)
            for r in rects:
                if r['class'].endswith('_?'):
                    r['class'] = r['class'][:-2]
                    info[r['class']] += 1
            yield key, json_dump(rects)
    def gen_info():
        yield 'remove _? in class name', 'keep the boxes'
        for k, v in info.items():
            yield k, v
    dataset.update_data(gen_rows(), split, 'label',
            generate_info=gen_info())

    info = defaultdict(int)
    def gen_rows2():
        iter_label = dataset.iter_data(split, 'label')
        for key, str_rects in iter_label:
            rects = json.loads(str_rects)
            origin_num = len(rects)
            info['original box'] += origin_num
            if split == 'train':
                rects = [r for r in rects if not r['class'].endswith('_?')]
                info['removed'] += origin_num - len(rects)
            else:
                assert split == 'test'
                for r in rects:
                    if r['class'].endswith('_?'):
                        r['class'] = r['class'][:-2]
                        r['diff'] = 1
                        info['new diff set'] += 1
            yield key, json_dump(rects)
    def gen_info2():
        if split == 'train':
            yield 'remove the box whose label ends with _?',
        else:
            assert split == 'test'
            yield 'remove _? if the class ends with _?.', \
                    'keep the box', \
                    'add diff = 1'
        for k, v in info.items():
            yield k, v
    dataset.update_data(gen_rows2(), split, 'label',
            generate_info=gen_info2())

def remove_size_in_label(data, split, version):
    dataset = TSVDataset(data)
    def gen_rows():
        for key, str_rects in tqdm(dataset.iter_data(split, 'label',
            version=version)):
            rects = json.loads(str_rects)
            assert len(rects) == 1
            c = rects[0]['class']
            assert not all(s in c for s in ['_s_', '_S_'])
            for s in ['_s_', '_S_']:
                parts = c.split('_s_')
                assert len(parts) in [1, 2]
                rects[0]['class'] = parts[0]
                if len(parts) == 2:
                    break
            yield key, json_dump(rects)
    dataset.update_data(gen_rows(), split, 'label',
        generate_info=[('remove size in the label', )])

def crop_detbox(data, split, version, out_data=None, save_type=None):
    dataset = TSVDataset(data)
    iter_image = dataset.iter_data(split)
    iter_label = dataset.iter_data(split, 'label', version=version)
    labels = []
    def gen_rows():
        for (image_key, _, image_str), (label_key, str_rects) in tqdm(
                zip(iter_image, iter_label)):
            assert image_key == label_key
            im = img_from_base64(image_str)
            h, w = im.shape[:2]
            rects = json.loads(str_rects)
            for i, r in enumerate(rects):
                x0, y0, x1, y1 = r['rect']
                x0, y0 = int(x0), int(y0)
                x1, y1 = math.ceil(x1), math.ceil(y1)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(x1, w), min(y1, h)
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = im[y0:y1, x0:x1, :]
                del r['rect']
                res_key = image_key + '${}'.format(i)
                str_label = json_dump([r])
                labels.append((res_key, str_label))
                yield res_key, str_label, encoded_from_img(sub,
                        save_type=save_type)
    if out_data is None:
        out_data = '{}_crop_{}{}'.format(data, split, version)
    out_dataset = TSVDataset(out_data)
    if out_dataset.has(split):
        logging.info('{} - {} exists'.format(out_dataset.name,
            split))
        return out_data
    #assert not out_dataset.has(split)
    out_dataset.write_data(gen_rows(), split)
    out_dataset.write_data(labels, split, 'label')
    populate_dataset_details(out_data)
    return out_data

def create_single_class_detection_taxonomy(data, split, version, root_name,
        excludes=None):
    populate_dataset_details(data)
    dataset = TSVDataset(data)
    labels = [l for l, in dataset.iter_data(split, 'labelmap', version=version)]
    if excludes is not None:
        lower_label = [(l.lower(), l) for l in labels]
        lower_to_labels = list_to_dict(lower_label, 0)
        lower_excludes = [e.lower() for e in excludes]
        for e in lower_excludes:
            if e in lower_to_labels:
                del lower_to_labels[e]
        labels = [l for _, ls in lower_to_labels.items() for l in ls]
    info = [{'name': root_name,
            'alias_names': ','.join(labels)}]
    return info

def unflatten_to_detection(data, split, pred, output,
        threshold=0.01):
    dataset = TSVDataset(data)
    iter_label = dataset.iter_data(split, 'label')
    iter_pred = tsv_reader(pred)
    def gen_rows():
        pre_key = None
        pre_rects = []
        for (key, str_rects), (pred_key, pred_rects) in tqdm(zip(
                iter_label, iter_pred)):
            assert key == pred_key
            idx = key.rfind('$')
            real_key = key[:idx]
            if real_key != pre_key:
                if pre_key is not None:
                    yield pre_key, json_dump(pre_rects)
                pre_key = real_key
                pre_rects = []
            gt = json.loads(str_rects)
            assert len(gt) == 1
            gt = gt[0]
            pred = json.loads(pred_rects)
            for p in pred:
                sub_gt = copy.deepcopy(gt)
                sub_gt['class'] = p['class']
                sub_gt['conf'] *= p['conf']
                if sub_gt['conf'] > threshold:
                    pre_rects.append(sub_gt)
        yield pre_key, json_dump(pre_rects)
    tsv_writer(gen_rows(), output)

def flatten_labels_for_classification(image_data, image_split, label_tsv,
        out_data, out_split):
    input_argument = get_frame_info()
    input_argument['func'] = 'flatten_labels_for_classification'

    dataset = TSVDataset(image_data)

    populate_dataset_hw(image_data, [image_split])

    iter_hw = dataset.iter_data(image_split, 'hw')
    if label_tsv is not None:
        iter_label = tsv_reader(label_tsv)
    else:
        iter_label = dataset.iter_data(image_split, 'label')

    all_idx = []
    def gen_rows():
        for idx_image, ((label_key, str_rects), (hw_key, str_hw)) in enumerate(
                zip(iter_label, iter_hw)):
            assert label_key == hw_key
            h, w = map(int, str_hw.split(' '))
            rects = json.loads(str_rects)
            for i, r in enumerate(rects):
                x1, y1, x2, y2 = r['rect']

                # since it is for classification, it has to be a valid box.
                # Here we do the fewest pre-processing/filtering to have a
                # valid box coordinates
                x1 = int(x1)
                y1 = int(y1)
                x2 = math.ceil(x2)
                y2 = math.ceil(y2)

                x1 = min(w, max(0, x1))
                x2 = min(w, max(0, x2))
                y1 = min(h, max(0, y1))
                y2 = min(h, max(0, y2))

                if y2 <= y1 or x2 <= x1:
                    continue
                r['rect'] = [x1, y1, x2, y2]
                all_idx.append(idx_image)
                yield label_key + '${}'.format(i), json_dump([r])
    if out_data == None:
        out_data = '{}_{}'.format(image_data,
                hash_sha1(label_tsv)[-5:])
    out_dataset = TSVDataset(out_data)
    if op.isdir(out_dataset._data_root):
        logging.info('{} exists'.format(out_dataset._data_root))
        return out_data
    out_dataset.write_data(gen_rows(), out_split, 'label')
    tsv_writer(((0, i) for i in all_idx),
            out_dataset.get_shuffle_file(out_split))
    write_to_file(dataset.get_data(image_split), out_dataset.get_data(out_split
        + 'X'))
    readme_file = op.join(out_dataset._data_root, 'readme.yaml')

    write_to_yaml_file(input_argument, readme_file)

    expand_nested_splitX(out_data, out_split)

    return out_data

def download_images_to_tsv(info, tsv_file, key='key'):
    from .common import url_to_bytes
    from .common import bytes_to_image
    def processor(i):
        bs = url_to_bytes(i['url'])
        if bs is None:
            return None
        try:
            im = bytes_to_image(bs)
        except:
            return None
        if im is None:
            return None
        return i[key], json_dump([]), base64.b64encode(bs)
    parallel_tsv_process(processor, info, tsv_file, num_process=16,
            num_jobs=1024)

def smart_resize_dataset(data, input_size, out_data, resize_type='cv2',
        save_type=None):
    if resize_type is None:
        resize_type = 'cv2'
    assert resize_type in ['cv2', 'pil']
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)

    for split in ['train', 'test']:
        if not dataset.has(split):
            continue
        infos = []
        def gen_rows():
            if resize_type == 'cv2':
                for key, str_rects, str_im in dataset.iter_data(split,
                        progress=True):
                    im = img_from_base64(str_im)
                    im2, left, top, scale = smart_resize(im, input_size)
                    im2 = im2.astype(np.uint8)
                    infos.append((left, top, scale))
                    yield key, json_dump([]), encoded_from_img(im2,
                            save_type=save_type)
            else:
                assert resize_type == 'pil'
                for key, str_rects, str_im in dataset.iter_data(split,
                        progress=True):
                    im = pilimg_from_base64(str_im)
                    im2, left, top, scale = smart_resize(im, input_size)
                    infos.append((left, top, scale))
                    yield key, json_dump([]), encoded_from_img(im2,
                            save_type=save_type)
        if not out_dataset.has(split):
            out_dataset.write_data(gen_rows(), split)
        v = 0
        while dataset.has(split, 'label', version=v):
            def gen_label_rows():
                for i, (key, str_rects) in enumerate(dataset.iter_data(split, 'label',
                        version=v, progress=True)):
                    rects = json.loads(str_rects)
                    def transform_box(r, info):
                        left, top, scale = info
                        x1, y1, x2, y2 = r['rect']
                        out_x1 = x1 * scale + left
                        out_y1 = y1 * scale + top
                        out_x2 = x2 * scale + left
                        out_y2 = y2 * scale + top
                        r['rect'] = [out_x1, out_y1, out_x2, out_y2]
                        return r
                    rects = [transform_box(r, infos[i]) for r in rects]
                    yield key, json_dump(rects)
            if not out_dataset.has(split, 'label', version=v):
                out_dataset.write_data(gen_label_rows(), split, 'label', version=v)
            v += 1
    ensure_copy_file(dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file())
    populate_dataset_hw(out_data)

def smart_resize_lefttop(im, target_size, lower=True):
    from .process_image import is_pil_image
    if is_pil_image(im):
        w, h = im.size
    else:
        h, w, c = im.shape
    alpha = target_size / np.sqrt(h * w)
    if lower:
        height2 = int(alpha * h)
        width2 = int(alpha * w)
    else:
        height2 = int(np.round(alpha * h))
        width2 = int(np.round(alpha * w))
    if h > w:
        if lower:
            input_height = height2 // 32 * 32
            input_width = (input_height * w) // h // 32 * 32
            im_scale = input_width / w
        else:
            input_height = (height2 + 31) // 32 * 32
            input_width = ((input_height * w + h - 1) // h +
                                        31) // 32 * 32
    else:
        if lower:
            input_width = (width2) // 32 * 32
            input_height = (input_width * h) // w // 32 * 32
            im_scale = input_height / h
        else:
            input_width = (width2 + 31) // 32 * 32
            input_height = ((input_width * h + w - 1) // w +
                                        31) // 32 * 32

    im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
    if is_pil_image(im_resized):
        new_w, new_h = im_resized.size
    else:
        new_h, new_w = im_resized.shape[0:2]
    left = 0
    right = input_width - new_w - left
    top = 0
    bottom = input_height - new_h - top
    from .process_image import copy_make_border
    im_squared = copy_make_border(im_resized, top=top,
            bottom=bottom, left=left, right=right)
    return im_squared, im_scale

def smart_resize(im, target_size):
    from .process_image import is_pil_image
    if is_pil_image(im):
        w, h = im.size
    else:
        h, w, c = im.shape
    alpha = target_size / np.sqrt(h * w)
    height2 = int(np.round(alpha * h))
    width2 = int(np.round(alpha * w))
    if h > w:
        input_height = (height2 + 31) // 32 * 32
        input_width = ((input_height * w + h - 1) // h +
                                    31) // 32 * 32
    else:
        input_width = (width2 + 31) // 32 * 32
        input_height = ((input_width * h + w - 1) // w +
                                    31) // 32 * 32

    if float(h) / input_height > float(w) / input_width:
        rescale_size = input_height
    else:
        rescale_size = input_width
    from .process_image import im_rescale
    im_resized, im_scale = im_rescale(im, rescale_size)
    if is_pil_image(im_resized):
        new_w, new_h = im_resized.size
    else:
        new_h, new_w = im_resized.shape[0:2]
    left = (input_width - new_w) // 2
    right = input_width - new_w - left
    top = (input_height - new_h) // 2
    bottom = input_height - new_h - top
    from .process_image import copy_make_border
    im_squared = copy_make_border(im_resized, top=top,
            bottom=bottom, left=left, right=right)
    return im_squared, left, top, im_scale

def kmeans(np_feature, k, seed=6):
    from sklearn.preprocessing import normalize
    from libKMCUDA import kmeans_cuda
    np_feature = normalize(np_feature, axis=1)
    init = 'kmeans++'
    while True:
        centroid, assignments = kmeans_cuda(np_feature,
                                            k,
                                            verbosity=1,
                                            seed=seed,
                                            device=1,
                                            init=init,
                                            metric='cos',
                                            )
        unique_assignment, cluster_count = np.unique(assignments, return_counts=True)
        if len(unique_assignment) == k:
            logging.info('reached non-empty')
            break
        num_new = k - len(unique_assignment)
        logging.info('adding {}'.format(num_new))
        new_from_cluster_idx = np.argsort(cluster_count)[-num_new:]
        cluster_idx = unique_assignment[new_from_cluster_idx]
        feat_idx = [np.where(assignments == i)[0] for i in cluster_idx]
        feat_idx = np.concatenate(feat_idx)
        np.random.seed(6)
        feat_idx = np.random.permutation(feat_idx)
        new_cluster = np_feature[feat_idx[:num_new]]
        new_cluster_idx = 0
        for c in range(len(centroid)):
            if np.isnan(centroid[c][0]):
                centroid[c] = new_cluster[new_cluster_idx]
                new_cluster_idx += 1
        assert new_cluster_idx == len(new_cluster)
        init = centroid
    return centroid, assignments

def kmeans_pred_to_dataX(data, split, k, feature_fname, out_data,
                         num_kmeans=1):
    out_dataset = TSVDataset(out_data)
    if op.isfile(out_dataset.get_labelmap_file()) and \
            op.isfile(out_dataset.get_data(split + 'X')):
        return out_data
    dataset = TSVDataset(data)
    iter_label = dataset.iter_data(split, 'label')
    iter_pred = tsv_reader(feature_fname)
    np_feature = None
    num_rows = dataset.num_rows(split)
    for i, (label_row, pred_row) in tqdm(enumerate(zip(iter_label, iter_pred))):
        assert label_row[0] == pred_row[0]
        f = json.loads(pred_row[-1])[0]['feature']
        if np_feature is None:
            np_feature = np.zeros((num_rows, len(f)), dtype=np.float32)
        np_feature[i] = f
    # there are some issues with multi-gpu. Thus setting device=1 which means to use
    # gpu 0
    all_centroid, all_assignment = [], []
    for i in range(num_kmeans):
        centroid, assignments = kmeans(np_feature, k, seed=i + 9)
        all_centroid.append(centroid)
        all_assignment.append(assignments)
    disk_assign = [int(i) for _, i in out_dataset.iter_data(split, 'label')]
    for a, da in zip(assignments, disk_assign):
        assert a == da
    for i, centroid in zip(range(num_kmeans), all_centroid):
        out_dataset.write_data(centroid, split, 'kmeans_center', version=i)
    write_to_file(dataset.get_data(split),
                  out_dataset.get_data(split + 'X'))
    tsv_writer(((0, i) for i in range(num_rows)),
               out_dataset.get_shuffle_file(split))
    if op.isfile(out_dataset.get_labelmap_file()):
        labelmap = out_dataset.load_labelmap()
        assert len(labelmap) == k
        for i, l in enumerate(labelmap):
            assert l == str(i)
    else:
        tsv_writer(((i,) for i in range(k)), out_dataset.get_labelmap_file())
    iter_label = dataset.iter_data(split, 'label')
    def gen_label():
        for row in zip(iter_label, *all_assignment):
            key = row[0][0]
            label = '_'.join(map(str, [int(a) for a in row[1:]]))
            yield key, json_dump([{'class': label}])
    tsv_writer(gen_label(), out_dataset.get_data(split, 'label'))
    tsv_writer([(feature_fname,)], out_dataset.get_data(split, 'feature_ptr'))
    return out_data

def create_compare_dataset(pred1, pred2, suffix1, suffix2, data, split, out_data):
    out_dataset = TSVDataset(out_data)
    dataset = TSVDataset(data)
    tsv_writer([(dataset.get_data(split),)], out_dataset.get_data(split + 'X'))
    tsv_writer(((0, i)  for i in range(dataset.num_rows(split)) for _ in
                range(2)),
               out_dataset.get_shuffle_file(split))
    iter_pred1 = tsv_reader(pred1)
    iter_pred2 = tsv_reader(pred2)
    iter_gt = dataset.iter_data(split, 'label')
    def gen_label_rows():
        for row_gt, row_pred1, row_pred2 in zip(iter_gt, iter_pred1, iter_pred2):
            assert row_gt[0] == row_pred1[0] == row_pred2[0]
            rects_pred1 = json.loads(row_pred1[1])
            rects_pred1 = [r for r in rects_pred1 if r['conf'] > 0.2]
            rects_pred2 = json.loads(row_pred2[1])
            rects_pred2 = [r for r in rects_pred2 if r['conf'] > 0.2]
            yield row_gt[0] + suffix1, json_dump(rects_pred1)
            yield row_gt[0] + suffix2, json_dump(rects_pred2)
    out_dataset.write_data(gen_label_rows(), split, 'label')
    expand_nested_splitX(out_data, split)
    populate_dataset_details(out_data)

def duplicate_balance_fg_classes(
    data, split, version,
    bkg_class, fg_ratio, fg_rel_class_ratio, out_data):
    # some recommendation: fg_ratio = 0.3
    # fg_rel_class_ratio: 0.1
    info = get_frame_info()
    populate_dataset_details(data)
    out_dataset = TSVDataset(out_data)
    if op.isfile(out_dataset.get_labelmap_file()):
        logging.info('{} exists'.format(out_dataset))
        return
    dataset = TSVDataset(data)

    # {label: list_of_idx}
    label_to_idx = dataset.load_inverted_label(split, version)
    # the number of images with unknown categories
    unknown_count = len(label_to_idx[bkg_class])
    # total number of images with product categories
    product_count = sum([len(idx) for l, idx in label_to_idx.items()
         if l != bkg_class])
    # current ratio
    curr_ratio = 1. * product_count / (product_count + unknown_count)
    # we need to duplicate product images by dup_factor times
    dup_factor = int(math.ceil(fg_ratio / curr_ratio))
    # duplicate operations by duplicating the index
    dup_label_to_idx = {l: idx * dup_factor for l, idx in label_to_idx.items()
                        if l != bkg_class}
    info['dup_factor'] = dup_factor
    # calculate the average number of images for each category
    avg_count = calc_mean([len(idx) for l, idx in dup_label_to_idx.items()])
    # based on the parameters, each category should have at_least_count images.
    at_least_count = int(avg_count * fg_rel_class_ratio)
    for l, idx in dup_label_to_idx.items():
        if len(idx) < at_least_count:
            info[l] = (len(idx), at_least_count)
    # duplicate the images by duplicating the index
    dup_label_to_idx = {l: idx
                        if len(idx) >= at_least_count else idx * int(math.ceil(1. * at_least_count / len(idx)))
                        for l, idx in dup_label_to_idx.items()
                        }
    # make the index as a list
    final_idx = [i for _, idxs in dup_label_to_idx.items() for i in idxs]
    final_idx.extend(label_to_idx[bkg_class])
    # random shuffling
    random.shuffle(final_idx)
    # save it
    tsv_writer(((0, i) for i in final_idx), out_dataset.get_shuffle_file(split))
    assert op.isfile(dataset.get_data(split))
    assert op.isfile(dataset.get_data(split, t='label', version=version))
    tsv_writer([(dataset.get_data(split),)], out_dataset.get_data(split + 'X'))
    tsv_writer([(dataset.get_data(split, t='label', version=version),)],
               out_dataset.get_data(split + 'X', t='label'))
    ensure_copy_file(dataset.get_labelmap_file(),
                     out_dataset.get_labelmap_file())
    out_dataset.write_data(info.items(), split, t='generate_info')

    populate_dataset_details(out_data)

def split_dataset_one(
    data, num_split, out_data,
    split_type, split, version
):
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)
    num_rows = dataset.num_rows(split, t=split_type, version=version)
    each_size = (num_rows + num_split - 1) //  num_split
    shuffle = []
    t_to_tsvs = defaultdict(list)
    for i in range(num_split):
        start = i * each_size
        end = start + each_size
        end = min(end, num_rows)
        t = split_type
        logging.info('split = {}; i = {}; t = {}'.format(
            split, i, t))
        def gen_rows():
            for j, row in enumerate(dataset.iter_data(
                split, t=t,
                version=version,
                filter_idx=range(start,
                                 end),
                    progress=True)):
                if t is None:
                    shuffle.append((i, j))
                yield row
        ssplit = '{}_{}_{}'.format(split, i, num_split)
        if out_dataset.has(ssplit, t, version=version):
            iter_in_file = out_dataset.iter_data(ssplit, t, version=version)
            for in_file, curr in zip(iter_in_file, gen_rows()):
                assert in_file == curr
        else:
            out_dataset.safe_write_data(
                gen_rows(),
                split=ssplit, t=t,
                version=version)
        t_to_tsvs[str(t)].append(out_dataset.get_data(ssplit, t, version))
    if op.isfile(out_dataset.get_shuffle_file(split)):
        iter_in_file = tsv_reader(out_dataset.get_shuffle_file(split))
        for x, y in zip(iter_in_file, shuffle):
            assert x == y
    else:
        tsv_writer(shuffle, out_dataset.get_shuffle_file(split))
    t = split_type
    split_x = '{}X'.format(split)
    target_file = out_dataset.get_data(split_x, t=t, version=version)
    if op.isfile(target_file):
        assert '\n'.join(t_to_tsvs[str(t)]) == read_to_buffer(target_file).decode()
    else:
        write_to_file('\n'.join(t_to_tsvs[str(t)]), target_file)
    ensure_copy_file(dataset.get_labelmap_file(), out_dataset.get_labelmap_file())

def split_dataset(
    data, num_split, out_data,
    split_types=(None, 'label', 'hw', 'caption'),
    splits=('train', 'trainval', 'test'),
):
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)
    for split in splits:
        num_rows = dataset.num_rows(split)
        each_size = (num_rows + num_split - 1) //  num_split
        shuffle = []
        t_to_tsvs = defaultdict(list)
        for i in range(num_split):
            start = i * each_size
            end = start + each_size
            end = min(end, num_rows)
            for t in split_types:
                logging.info('split = {}; i = {}; t = {}'.format(
                    split, i, t))
                def gen_rows():
                    for j, row in enumerate(dataset.iter_data(
                        split, t=t,
                        filter_idx=range(start,
                                         end),
                            progress=True)):
                        if t is None:
                            shuffle.append((i, j))
                        yield row
                ssplit = '{}_{}_{}'.format(split, i, num_split)
                if out_dataset.has(ssplit, t):
                    iter_in_file = out_dataset.iter_data(ssplit, t)
                    for in_file, curr in zip(iter_in_file, gen_rows()):
                        assert in_file == curr
                else:
                    out_dataset.write_data(gen_rows(), split=ssplit, t=t)
                t_to_tsvs[str(t)].append(out_dataset.get_data(ssplit, t))
        if op.isfile(out_dataset.get_shuffle_file(split)):
            iter_in_file = tsv_reader(out_dataset.get_shuffle_file(split))
            for x, y in zip(iter_in_file, shuffle):
                assert x == y
        else:
            tsv_writer(shuffle, out_dataset.get_shuffle_file(split))
        for t in split_types:
            split_x = '{}X'.format(split)
            target_file = out_dataset.get_data(split_x, t=t)
            if op.isfile(target_file):
                assert '\n'.join(t_to_tsvs[str(t)]) == read_to_buffer(target_file).decode()
            else:
                write_to_file('\n'.join(t_to_tsvs[str(t)]), target_file)
    ensure_copy_file(dataset.get_labelmap_file(), out_dataset.get_labelmap_file())

def merge_dataset_parallel(*args, **kwargs):
    merge_dataset(*args, **kwargs)

#@deprecated('use merge_dataset_parallel')
#def merge_dataset(source_infos, target_info, ps):
    #target_dataset = TSVDataset(target_info['data'])
    #for p in ps:
        #tsv_list = []
        #for source in source_infos:
            #dataset = TSVDataset(source['data'])
            #if 'version_info' in source and p['type'] in source['version_info']:
                #version = source['version_info'][p['type']]
            #elif 'version' in p:
                #version = p['version']
            #else:
                #version = None
            #tsv_file = dataset.get_data(source['split'], t=p['type'],
                                #version=version)
            #if op.isfile(tsv_file):
                #tsv_list.append(tsv_file)
            #else:
                #tsv_x = dataset.get_data(source['split'] + 'X', t=p['type'],
                                         #version=version)
                #curr_list = load_list_file(tsv_x)
                #curr_shuffle = dataset.get_shuffle_file(source['split'])
                #logging.info('loading {}'.format(curr_shuffle))
                #tsv_list.extend(curr_list)
        #out_tsvx = target_dataset.get_data(
            #target_info['split'] + 'X', p['type'],
            #version=target_info.get('version'))
        #if not op.isfile(out_tsvx):
            #write_to_file('\n'.join(tsv_list), out_tsvx)
        #else:
            #es = load_list_file(out_tsvx)
            #assert len(tsv_list) == len(es)
            #for e, c in zip(es, tsv_list):
                #assert e == c, (e, c)

    #for p in ps:
        #if p.get('type') is not None:
            #continue
        #tsv_list = []
        #def gen_shuffle():
            #for source in source_infos:
                #dataset = TSVDataset(source['data'])
                #if 'version_info' in source and p['type'] in source['version_info']:
                    #version = source['version_info'][p['type']]
                #elif 'version' in p:
                    #version = p['version']
                #else:
                    #version = None
                #tsv_file = dataset.get_data(source['split'], t=p['type'],
                                    #version=version)
                #tsv_offset = len(tsv_list)
                #if op.isfile(tsv_file):
                    #num = dataset.num_rows(source['split'])
                    #for i in range(num):
                        #yield tsv_offset, i
                    #tsv_list.append(tsv_file)
                #else:
                    #tsv_x = dataset.get_data(source['split'] + 'X', t=p['type'],
                                             #version=version)
                    #curr_list = load_list_file(tsv_x)
                    #curr_shuffle = dataset.get_shuffle_file(source['split'])
                    #logging.info('loading {}'.format(curr_shuffle))
                    #for i, j in tqdm(tsv_reader(curr_shuffle)):
                        #yield int(i) + tsv_offset, j
                    #tsv_list.extend(curr_list)
        #shuffle_f = target_dataset.get_shuffle_file(target_info['split'])
        #if not op.isfile(shuffle_f):
            #tsv_writer(gen_shuffle(), shuffle_f)
        #else:
            #for in_file, curr in zip(tsv_reader(shuffle_f), gen_shuffle()):
                #assert in_file[0] == str(curr[0])
                #assert in_file[1] == str(curr[1])



#@deprecated('try to switch to merge_dataset2, tested once')
#def merge_dataset(source_infos, target_info, ps):
    #target_dataset = TSVDataset(target_info['data'])
    #for p in ps:
        #tsv_list = []
        #shuffle = []
        #for source in source_infos:
            #dataset = TSVDataset(source['data'])
            #if 'version_info' in source and p['type'] in source['version_info']:
                #version = source['version_info'][p['type']]
            #elif 'version' in p:
                #version = p['version']
            #else:
                #version = None
            #tsv_file = dataset.get_data(source['split'], t=p['type'],
                                #version=version)
            #tsv_offset = len(tsv_list)
            #if op.isfile(tsv_file):
                #num = dataset.num_rows(source['split'])
                #shuffle.extend([(tsv_offset, i) for i in range(num)])
                #tsv_list.append(tsv_file)
            #else:
                #tsv_x = dataset.get_data(source['split'] + 'X', t=p['type'],
                                         #version=version)
                #curr_list = load_list_file(tsv_x)
                #curr_shuffle = dataset.get_shuffle_file(source['split'])
                #logging.info('loading {}'.format(curr_shuffle))
                #shuffle.extend([(int(i) + tsv_offset, j) for i, j in
                                #tqdm(tsv_reader(curr_shuffle))])
                #tsv_list.extend(curr_list)
        #out_tsvx = target_dataset.get_data(
            #target_info['split'] + 'X', p['type'],
            #version=target_info.get('version'))
        #if not op.isfile(out_tsvx):
            #write_to_file('\n'.join(tsv_list), out_tsvx)
        #else:
            #es = load_list_file(out_tsvx)
            #assert len(tsv_list) == len(es)
            #for e, c in zip(es, tsv_list):
                #assert e == c, (e, c)
        #if p.get('type') is None:
            #shuffle_f = target_dataset.get_shuffle_file(target_info['split'])
            #if not op.isfile(shuffle_f):
                #tsv_writer(shuffle, shuffle_f)
            #else:
                #for in_file, curr in zip(tsv_reader(shuffle_f), shuffle):
                    #assert in_file[0] == str(curr[0])
                    #assert in_file[1] == str(curr[1])

    ## labelmap
    ##target_labelmap = target_dataset.get_labelmap_file()
    ##lss = [TSVDataset(source['data']).load_labelmap() for source in source_infos]
    ##labelmap = list(set([l for ls in lss for l in ls]))
    ##labelmap = sorted(labelmap)
    ##if not op.isfile(target_labelmap):
        ##write_to_file('\n'.join(labelmap), target_labelmap)
    ##else:
        ##existing = load_list_file(target_labelmap)
        ##assert len(existing) == len(labelmap)
        ##for e, l in zip(existing, labelmap):
            ##assert e == l

def merge_dataset_row_processor(row, offset):
    i, j = row
    return int(i) + offset, j

def merge_dataset(source_infos, target_info, ps):
    target_dataset = TSVDataset(target_info['data'])
    for p in ps:
        tsv_list = []
        for source in source_infos:
            dataset = TSVDataset(source['data'])
            if 'version_info' in source and p['type'] in source['version_info']:
                version = source['version_info'][p['type']]
            elif 'version' in p:
                version = p['version']
            else:
                version = None
            tsv_file = dataset.get_data(source['split'], t=p['type'],
                                version=version)
            if File.isfile(tsv_file):
                tsv_list.append(tsv_file)
            else:
                tsv_x = dataset.get_data(source['split'] + 'X', t=p['type'],
                                         version=version)
                curr_list = load_list_file(tsv_x)
                curr_shuffle = dataset.get_shuffle_file(source['split'])
                logging.info('loading {}'.format(curr_shuffle))
                tsv_list.extend(curr_list)
        out_tsvx = target_dataset.get_data(
            target_info['split'] + 'X', p['type'],
            version=target_info.get('version'))
        if not File.isfile(out_tsvx):
            tsv_writer(((_t, ) for _t in tsv_list), out_tsvx)
            #write_to_file('\n'.join(tsv_list), out_tsvx)
        else:
            es = load_list_file(out_tsvx)
            assert len(tsv_list) == len(es)
            for e, c in zip(es, tsv_list):
                assert e == c, (e, c)

    all_num_tsv = []
    for source in source_infos:
        dataset = TSVDataset(source['data'])
        curr_shuffle = dataset.get_shuffle_file(source['split'])
        if File.isfile(curr_shuffle):
            tsv_x = dataset.get_data(source['split'] + 'X', t=p['type'],
                                     version=version)
            curr_list = load_list_file(tsv_x)
            all_num_tsv.append(len(curr_list))
        else:
            assert File.isfile(dataset.get_data(source['split']))
            all_num_tsv.append(1)

    s = 0
    tsv_offset = [0]
    for n in all_num_tsv[:-1]:
        s += n
        tsv_offset.append(s)

    all_shuffle = []
    target_shuffle = target_dataset.get_shuffle_file(target_info['split'])
    folder = get_tmp_folder()
    for idxsource, source in enumerate(source_infos):
        dataset = TSVDataset(source['data'])
        out_shuffle = op.join(folder,
                              '{}'.format(target_shuffle) +
                              '.{}.{}.tsv'.format(idxsource, len(source_infos)),
                              )
        src_shuffle_tsv = dataset.get_shuffle_file(source['split'])
        parallel_tsv_process(lambda row: merge_dataset_row_processor(row, tsv_offset[idxsource]),
                             src_shuffle_tsv,
                             out_shuffle,
                             64,
                             )
        all_shuffle.append(out_shuffle)
    concat_tsv_files(all_shuffle,
                     target_dataset.get_shuffle_file(target_info['split']),
                     gen_lineidx=False
                     )
    delete_tsv_files(all_shuffle)

def tuple_merge_pred_label_to_gt(x):
    pred, data, split, in_version, out_version = x
    merge_pred_label_to_gt(pred, data, split, in_version, out_version)

def merge_pred_label_to_gt(pred, data, split, in_version, out_version):
    dataset = TSVDataset(data)
    def gen_rows():
        iter_pred = tsv_reader(pred)
        iter_gt = dataset.iter_data(split, 'label', version=in_version)
        for row_pred, row_gt in tqdm(zip(iter_pred, iter_gt)):
            assert row_pred[0] == row_gt[0]
            pred_rects = json.loads(row_pred[1])
            gt = json.loads(row_gt[1])
            for p in pred_rects:
                if 'zlib_feature' in p:
                    del p['zlib_feature']
                if 'attr_labels' in p:
                    del p['attr_labels']
                if 'attr_scores' in p:
                    del p['attr_scores']
            gt.extend(pred_rects)
            yield row_pred[0], json_dump(gt)
    #assert not dataset.has(split, 'label', version=out_version)
    dataset.write_data(gen_rows(), split, 'label', version=out_version)

def generate_caption_linelist(data, split, version=None):
    dataset = TSVDataset(data)
    if not dataset.has(split, 'caption', version=None):
        logging.info('caption does not exist')
        return
    iter_caption = dataset.iter_data(split, 'caption', version=version)
    iter_shuffle = tsv_reader(dataset.get_shuffle_file(split))

    def gen_rows():
        for (idx_source, idx_row), (key, s) in tqdm(zip(iter_shuffle,
                                                        iter_caption)):
            cs = json.loads(s)
            for idx_caption in range(len(cs)):
                yield idx_source, idx_row, idx_caption
    if not dataset.has(split, 'caption_linelist', version=version):
        dataset.safe_write_data(gen_rows(), split, 'caption_linelist',
                           version=version)

def generate_key_idximage_idxcaption_from_num(data, split, version):
    dataset = TSVDataset(data)
    if not dataset.has(split, 'caption', version):
        logging.info('ignore to generate as caption file does not exist')
        return
    iter_caption = dataset.iter_data(split, 'num_caption', version=version)

    def gen_rows():
        for idx_img, (key, n) in tqdm(enumerate(iter_caption)):
            n = int(n)
            for i in range(n):
                yield key, idx_img, i

    t = 'key_idximage_idxcaption'
    if not dataset.has(split, t, version=version):
        dataset.write_data(gen_rows(), split, t, version=version)

def derive_key_idximage_idxcaption_from_merge_dataset(source_infos, target_info):
    for s in source_infos:
        assert s.get('version_info') is None
        generate_key_idximage_idxcaption(s['data'], s['split'])

    split = target_info['split']
    def gen_rows():
        offset = 0
        for s in source_infos:
            dataset = TSVDataset(s['data'])
            for key, idximage, idxcaption in dataset.iter_data(s['split'],
                                                               t='key_idximage_idxcaption'):
                yield key, int(idximage) + offset, idxcaption
            added = False
            for t in [None, 'caption', 'hw']:
                if dataset.has(s['split'], t):
                    offset += dataset.num_rows(s['split'])
                    added = True
                    break
            assert added
    data = target_info['data']
    out = TSVDataset(data)
    if not out.has(split, 'key_idximage_idxcaption'):
        out.write_data(gen_rows(), split, t='key_idximage_idxcaption')

def derive_key_idximage_idxcaption_from_merge_dataset_parallel_multi_source(source_infos, target_info):
    for s in source_infos:
        assert s.get('version_info') is None
        generate_key_idximage_idxcaption(s['data'], s['split'])

    out = TSVDataset(target_info['data'])
    if out.has(target_info['split'], 'key_idximage_idxcaption'):
        return

    all_size = [len(TSVSplitProperty(s['data'], s['split'], 'caption')) for s in
                source_infos]
    cum_size = 0
    all_offset = [cum_size]
    for s in all_size[:-1]:
        cum_size += s
        all_offset.append(cum_size)

    in_tsvs = list(zip(all_offset, source_infos))
    from .cloud_storage import create_cloud_fuse
    c = create_cloud_fuse()
    tsvs = [TSVDataset(s['data']).get_data(s['split'], 'key_idximage_idxcaption') for s
            in source_infos]
    c.ensure_cache(tsvs)
    c.ensure_cache([get_tsv_associates(t)[0] for t in tsvs])

    def row_processor(row):
        offset, s = row
        for key, idximage, idxcaption in TSVSplitProperty(s['data'],
                                                          s['split'],
                                                          'key_idximage_idxcaption'):
            yield key, int(idximage) + offset, idxcaption
    out_tsv = out.get_data(target_info['split'], 'key_idximage_idxcaption')
    parallel_tsv_process(
        row_processor,
        in_tsvs,
        out_tsv,
        64
    )

def derive_key_idximage_idxcaption_from_merge_dataset_parallel_each(source_infos, target_info):
    for s in source_infos:
        assert s.get('version_info') is None
        generate_key_idximage_idxcaption(s['data'], s['split'])

    out = TSVDataset(target_info['data'])
    if out.has(target_info['split'], 'key_idximage_idxcaption'):
        return

    all_size = [len(TSVSplitProperty(s['data'], s['split'], 'caption')) for s in
                source_infos]
    cum_size = 0
    all_offset = [cum_size]
    for s in all_size[:-1]:
        cum_size += s
        all_offset.append(cum_size)

    from .cloud_storage import create_cloud_fuse
    c = create_cloud_fuse()
    tsvs = [TSVDataset(s['data']).get_data(s['split'], 'key_idximage_idxcaption') for s
            in source_infos]
    c.ensure_cache(tsvs)
    c.ensure_cache([get_tsv_associates(t)[0] for t in tsvs])

    tmp_folder = get_tmp_folder()
    out_tsv = out.get_data(target_info['split'], 'key_idximage_idxcaption')
    all_tmp = []
    for offset, s in zip(all_offset, source_infos):
        curr_in_tsv = TSVDataset(s['data']).get_data(s['split'], 'key_idximage_idxcaption')
        curr_out_tsv = op.join(tmp_folder, out_tsv + f'.{offset}.tsv')

        def row_processor(row):
            key, idximage, idxcaption = row
            return key, int(idximage) + offset, idxcaption
        parallel_tsv_process(
            row_processor,
            curr_in_tsv,
            curr_out_tsv,
            64
        )
        all_tmp.append(curr_out_tsv)
    concat_tsv_files(all_tmp, out_tsv, gen_lineidx=False)
    delete_tsv_files(all_tmp)

def derive_key_idximage_idxcaption_idxsource_from_merge_dataset_parallel_multi_source(source_infos, target_info):
    for s in source_infos:
        assert s.get('version_info') is None
        generate_key_idximage_idxcaption_idxsource(s['data'], s['split'])

    out = TSVDataset(target_info['data'])
    if out.has(target_info['split'], 'key_idximage_idxcaption_idxsource'):
        return

    all_size = [len(TSVSplitProperty(s['data'], s['split'], 'caption')) for s in
                source_infos]
    cum_size = 0
    all_img_offset = [cum_size]
    for s in all_size[:-1]:
        cum_size += s
        all_img_offset.append(cum_size)

    all_source_offset = [0]
    cum = 0
    for s in source_infos[:-1]:
        curr_dataset = TSVDataset(s['data'])
        if File.isfile(curr_dataset.get_data(s['split'])):
            num = 1
        else:
            num = len(load_list_file(curr_dataset.get_data(s['split'] + 'X')))
        cum += num
        all_source_offset.append(cum)

    in_tsvs = list(zip(all_img_offset, all_source_offset, source_infos))
    from .cloud_storage import create_cloud_fuse
    c = create_cloud_fuse()
    tsvs = [TSVDataset(s['data']).get_data(s['split'],
                                           'key_idximage_idxcaption_idxsource') for s
            in source_infos]
    c.ensure_cache(tsvs)
    c.ensure_cache([get_tsv_associates(t)[0] for t in tsvs])

    def row_processor(row):
        img_offset, source_offset, s = row
        for idxsource, in TSVSplitProperty(s['data'], s['split'], 'key_idximage_idxcaption_idxsource'):
            yield int(idxsource) + source_offset,
    out_tsv = out.get_data(target_info['split'], 'key_idximage_idxcaption_idxsource')
    parallel_tsv_process(row_processor, in_tsvs, out_tsv, 64)

def derive_key_idximage_idxcaption_idxsource_from_merge_dataset_parallel_each_source(source_infos, target_info):
    for s in source_infos:
        assert s.get('version_info') is None
        generate_key_idximage_idxcaption_idxsource(s['data'], s['split'])

    out = TSVDataset(target_info['data'])
    if out.has(target_info['split'], 'key_idximage_idxcaption_idxsource'):
        return

    all_size = [len(TSVSplitProperty(s['data'], s['split'], 'caption')) for s in
                source_infos]
    cum_size = 0
    all_img_offset = [cum_size]
    for s in all_size[:-1]:
        cum_size += s
        all_img_offset.append(cum_size)

    all_source_offset = [0]
    cum = 0
    for s in source_infos[:-1]:
        curr_dataset = TSVDataset(s['data'])
        if File.isfile(curr_dataset.get_data(s['split'])):
            num = 1
        else:
            num = len(load_list_file(curr_dataset.get_data(s['split'] + 'X')))
        cum += num
        all_source_offset.append(cum)

    c = create_cloud_fuse()
    tsvs = [TSVDataset(s['data']).get_data(s['split'],
                                           'key_idximage_idxcaption_idxsource') for s
            in source_infos]
    c.ensure_cache(tsvs)
    c.ensure_cache([get_tsv_associates(t)[0] for t in tsvs])

    out_tsv = out.get_data(target_info['split'], 'key_idximage_idxcaption_idxsource')

    tmp_folder = get_tmp_folder()
    all_tmp = []
    for img_offset, source_offset, s in zip(all_img_offset, all_source_offset, source_infos):
        def row_processor(row):
            idxsource, = row
            return int(idxsource) + source_offset,
        in_tsv = TSVDataset(s['data']).get_data(s['split'], 'key_idximage_idxcaption_idxsource')
        curr_out = '{}/{}'.format(tmp_folder, out_tsv + '.{}.tsv'.format(img_offset))
        parallel_tsv_process(row_processor, in_tsv, curr_out, 64)
        all_tmp.append(curr_out)
    concat_tsv_files(all_tmp, out_tsv, gen_lineidx=False)
    delete_tsv_files(all_tmp)

def assert_dataset_each_length_same(data):
    dataset = TSVDataset(data)
    img_tsvs = load_list_file(dataset.get_data('trainX'))
    cap_tsvs = load_list_file(dataset.get_data('trainX', 'caption'))
    assert len(img_tsvs) == len(cap_tsvs)
    #c = create_cloud_fuse()
    #c.ensure_cache(fs)
    for i, c in tqdm(zip(img_tsvs, cap_tsvs)):
        assert len(TSVFile(i)) == len(TSVFile(c)), (i, c)


def generate_key_idximage_idxcaption(data, split, version=None):
    dataset = TSVDataset(data)
    if not dataset.has(split, 'caption', version):
        logging.info('ignore to generate as caption file does not exist')
        return
    #from qd.data_layer.loader import iter_data_loader
    #iter_caption = iter_data_loader(
        #data, split, 'caption', version=version,
        #transform=lambda x: (x[0], json.loads(x[1])),
        #num_workers=8,
    #)
    info = defaultdict(int)
    def row_processor(row, idx_img):
        key, s = row
        try:
            cs = json.loads(s)
        except:
            logging.info('{}/{}'.format(s, idx_img))
            raise
        for i in range(len(cs)):
            if ('caption' in cs[i] and isinstance(cs[i]['caption'], str)) or ('caption' not in cs[i]):
                yield (key, idx_img, i)
                info['valid'] += 1
            else:
                info['non_str_caption'] += 1

    t = 'key_idximage_idxcaption'
    if not dataset.has(split, t, version=version):
        tsv = TSVSplitProperty(data=data, split=split, t='caption', version=version)
        fs = dataset.get_data(split + 'X', 'caption', version=version)
        if File.isfile(fs):
            tsvs = load_list_file(fs)
            File.prepare(tsvs)
            File.prepare([get_tsv_associates(t)[0] for t in tsvs])
        parallel_tsv_process(
            row_processor,
            tsv,
            dataset.get_data(split, t, version=version),
            64,
            with_row_idx=True
        )
    logging.info(info)

def generate_idximage_idxcaption(
        data, split, version=None):
    dataset = TSVDataset(data)
    if not dataset.has(split, 'caption', version):
        logging.info('ignore to generate as caption file does not exist')
        return
    info = defaultdict(int)
    def row_processor(row, idx_img):
        key, s = row
        cs = json.loads(s)
        for i in range(len(cs)):
            if ('caption' in cs[i] and isinstance(cs[i]['caption'], str)) or ('caption' not in cs[i]):
                yield idx_img, i
                info['valid'] += 1
            else:
                info['non_str_caption'] += 1

    t = 'idximage_idxcaption'
    if not dataset.has(split, t, version=version):
        tsv = TSVSplitProperty(data=data, split=split, t='caption', version=version)
        c = create_cloud_fuse()
        fs = dataset.get_data(split + 'X', 'caption', version=version)
        if File.isfile(fs):
            tsvs = load_list_file(fs)
            c.ensure_cache(tsvs)
            c.ensure_cache([get_tsv_associates(t)[0] for t in tsvs])
        parallel_tsv_process(
            row_processor,
            tsv,
            dataset.get_data(split, t, version=version),
            64,
            with_row_idx=True
        )
    logging.info(info)

@deprecated('use generate_idximage_idxcaption')
def generate_idximage_idxcaption_parallel(*args, **kwargs):
    generate_idximage_idxcaption(*args, **kwargs)

#def generate_key_idximage_idxcaption(data, split, version=None):
    #dataset = TSVDataset(data)
    #if not dataset.has(split, 'caption', version):
        #logging.info('ignore to generate as caption file does not exist')
        #return
    ##from qd.data_layer.loader import iter_data_loader
    ##iter_caption = iter_data_loader(
        ##data, split, 'caption', version=version,
        ##transform=lambda x: (x[0], json.loads(x[1])),
        ##num_workers=8,
    ##)
    #info = defaultdict(int)
    #def gen_rows():
        ##iter_caption = dataset.iter_data(split, 'caption', version=version)
        ##total = dataset.num_rows(split, 'caption')
        #iter_caption = TSVSplitProperty(data=data, split=split, t='caption', version=version)
        ##for idx_img, (key, s) in tqdm(enumerate(iter_caption), total=total):
        #for idx_img, (key, s) in tqdm(enumerate(iter_caption)):
            #cs = json.loads(s)
            #for i in range(len(cs)):
                #if ('caption' in cs[i] and isinstance(cs[i]['caption'], str)) or ('caption' not in cs[i]):
                    #yield key, idx_img, i
                    #info['valid'] += 1
                #else:
                    #info['non_str_caption'] += 1

    #t = 'key_idximage_idxcaption'
    #if not dataset.has(split, t, version=version):
        #dataset.write_data(gen_rows(), split, t, version=version)
    #logging.info(info)

def generate_idximage_idxcaption_from_with_key(data, split, version=None):
    dataset = TSVDataset(data)
    def gen_rows():
        #iter_caption = dataset.iter_data(split, 'caption', version=version)
        #total = dataset.num_rows(split, 'caption')
        iter_old = TSVSplitProperty(data=data, split=split, t='key_idximage_idxcaption', version=version)
        #for idx_img, (key, s) in tqdm(enumerate(iter_caption), total=total):
        for key, idximage, idxcaption in tqdm(iter_old):
            yield idximage, idxcaption

    t = 'idximage_idxcaption'
    if not dataset.has(split, t, version=version):
        dataset.write_data(gen_rows(), split, t, version=version)


#def generate_idximage_idxcaption(data, split, version=None):
    #dataset = TSVDataset(data)
    #if not dataset.has(split, 'caption', version):
        #logging.info('ignore to generate as caption file does not exist')
        #return
    ##from qd.data_layer.loader import iter_data_loader
    ##iter_caption = iter_data_loader(
        ##data, split, 'caption', version=version,
        ##transform=lambda x: (x[0], json.loads(x[1])),
        ##num_workers=8,
    ##)
    #info = defaultdict(int)
    #def gen_rows():
        ##iter_caption = dataset.iter_data(split, 'caption', version=version)
        ##total = dataset.num_rows(split, 'caption')
        #iter_caption = TSVSplitProperty(data=data, split=split, t='caption', version=version)
        ##for idx_img, (key, s) in tqdm(enumerate(iter_caption), total=total):
        #for idx_img, (key, s) in tqdm(enumerate(iter_caption)):
            #cs = json.loads(s)
            #for i in range(len(cs)):
                #if ('caption' in cs[i] and isinstance(cs[i]['caption'], str)) or ('caption' not in cs[i]):
                    #yield idx_img, i
                    #info['valid'] += 1
                #else:
                    #info['non_str_caption'] += 1

    #t = 'idximage_idxcaption'
    #if not dataset.has(split, t, version=version):
        #dataset.write_data(gen_rows(), split, t, version=version)
    #logging.info(info)

def generate_num_caption(data, split, version=None):
    dataset = TSVDataset(data)
    if not dataset.has(split, 'caption', version):
        logging.info('ignore to generate as caption file does not exist')
        return
    iter_caption = dataset.iter_data(split, 'caption', version=version)

    def gen_rows():
        for (key, s) in tqdm(iter_caption):
            cs = json.loads(s)
            yield key, len(cs)

    if not dataset.has(split, 'num_caption', version=version):
        dataset.write_data(gen_rows(), split, 'num_caption', version=version)

def create_n_class_label(data_sources, out_data, out_split):
    frame_info = get_frame_info()
    out_dataset = TSVDataset(out_data)
    if out_dataset.has(out_split, 'label'):
        logging.info('skip for {}'.format(pformat(frame_info)))
        return
    all_ds = []
    for ds in data_sources:
        expand = ds.get('expand')
        if expand:
            curr_dataset = TSVDataset(ds['data'])
            data_splits = curr_dataset.load_composite_source_data_split(
                ds['split'])
            for d, s in data_splits:
                all_ds.append({'data': d, 'split': s, 'version': ds['version']})
        else:
            # in expand, we only support data/split/version. If we add other property, we should also update this code
            assert len(ds) == 3
            assert all(k in ds for k in ['data', 'split', 'version'])
            all_ds.append(ds)
    data_sources = all_ds
    # with expand, data_sources will be different. We save this information
    frame_info['data_sources'] = data_sources

    def gen_rows():
        for idx_source, data_source in enumerate(data_sources):
            curr_dataset = TSVDataset(data_source['data'])
            iter_label = curr_dataset.iter_data(
                data_source['split'],
                'label',
                version=data_source['version'],
            )
            total = curr_dataset.num_rows(data_source['split'])
            for i, (key, str_rects) in tqdm(enumerate(iter_label), total=total):
                rects = json.loads(str_rects)
                if isinstance(rects, int):
                    rects = [{'class': str(rects)}]
                if len(rects) == 0:
                    continue
                for r in rects:
                    if '$' not in r['class']:
                        if not r['class'].startswith('-'):
                            r['class'] = '$'.join([data_source['data'], r['class']])
                        else:
                            r['class'] = '$'.join([data_source['data'], r['class'][1:]])
                            r['class'] = '-{}'.format(r['class'])
                yield (key, json_dump(rects)), (idx_source, i)
    def gen_info():
        yield 'create_n_classs_label',
        ps = dict_get_all_path(frame_info)
        for p in ps:
            v = dict_get_path_value(frame_info, p)
            yield p, v

    label_file = out_dataset.get_data(out_split, 'label')
    shuffle_file = out_dataset.get_shuffle_file(out_split)
    if not op.isfile(label_file):
        tsv_writers(gen_rows(), (label_file, shuffle_file))
        gen_tsv = out_dataset.get_gen_info_data(out_split, 'label')
        tsv_writer(gen_info(), gen_tsv)
    labelmap = []
    #domain_info = []
    for ds_data in set(d['data'] for d in data_sources):
        curr_labelmap = TSVDataset(ds_data).load_labelmap()
        labelmap.extend(['{}${}'.format(
            ds_data, l) for l in curr_labelmap])
        #domain_info.append({'data': ds_data, 'labelmap': curr_labelmap})
    write_to_file('\n'.join(set(labelmap)), out_dataset.get_labelmap_file())
    #write_to_yaml_file(domain_info, out_dataset.get_file('domainmap.yaml'))
    image_tsvs = [TSVDataset(source['data']).get_data(source['split']) for source in
                  data_sources]
    write_to_file('\n'.join(image_tsvs), out_dataset.get_data(out_split + 'X'))
    expand_nested_splitX(out_data, out_split)

def merge_prediction_to_data(
    data, split, in_version, predict_file, conf_th, out_version,
    merge_colocated_iou=0.8,
):
    frame_info = get_frame_info()
    dataset = TSVDataset(data)
    assert dataset.has(split, 'label', version=in_version), (
        dataset, split, in_version
    )
    if dataset.has(split, 'label', version=out_version):
        logging.info('out version esists {}'.format(out_version))
        return
    iter_gt = dataset.iter_data(split, 'label', version=in_version)
    info = defaultdict(int)
    def gen_rows():
        debug = False
        iter_pred = tsv_reader(predict_file)
        for i, (row_gt, row_pred) in tqdm(enumerate(
            zip(iter_gt, iter_pred))):
            key = row_gt[0]
            assert row_pred[0] == key
            gt = json.loads(row_gt[1])
            pred = json.loads(row_pred[1])
            pred = [p for p in pred if p['conf'] >= conf_th]
            if debug:
                origin_gt = copy.deepcopy(gt)
                extra = []
            info['num_label_pre'] += len(gt)
            for g in gt:
                if 'location_id' not in g:
                    g['location_id'] = hash_sha1(g['rect'])
            for p in pred:
                if 'location_id' not in p:
                    p['location_id'] = hash_sha1(p['rect'])
            for p in pred:
                idx, iou = find_best_matched_rect_idx(p, gt, check_class=False)
                if iou >= merge_colocated_iou:
                    p['rect'] = gt[idx]['rect']
                    p['location_id'] = gt[idx]['location_id']
                    info['merge_colocated'] += 1
                p['predict_file'] = predict_file
                info['added'] += 1
                if debug:
                    extra.append(p)
                gt.append(p)
            if debug and len(extra) > 0:
                logging.info('debugging')
                im = img_from_base64(dataset.seek_by_idx(i, split='train')[-1])
                im_gt = im.copy()
                draw_rects(origin_gt, im_gt)
                im_extra = im.copy()
                draw_rects(extra, im_extra)
                save_image(np.concatenate((im_gt, im_extra)),
                           '/mnt/gpu02_raid/jianfw/work/tmp9/{}.jpg'.format(key))
            info['num_label_after'] += len(gt)
            yield key, json_dump(gt)
    def gen_info():
        for k, v in frame_info.items():
            yield k, v
        for k, v in info.items():
            yield k, v
    dataset.write_data(gen_rows(), split, 'label', version=out_version,
                       generate_info=gen_info())
    logging.info(pformat(info))

def create_1_class_label(data_sources, out_data, out_split,
                         cls_name):
    frame_info = get_frame_info()
    out_dataset = TSVDataset(out_data)
    if out_dataset.has(out_split, 'label'):
        logging.info('skip for {}'.format(pformat(frame_info)))
        return
    def gen_rows():
        for idx_source, data_source in enumerate(data_sources):
            curr_dataset = TSVDataset(data_source['data'])
            iter_label = curr_dataset.iter_data(
                data_source['split'],
                'label',
                version=data_source['version'],
            )
            for i, (key, str_rects) in tqdm(enumerate(iter_label)):
                rects = json.loads(str_rects)
                if len(rects) == 0:
                    continue
                for r in rects:
                    assert 'from_class' not in r
                    r['from_class'] = r['class']
                    r['class'] = 'object'
                yield (key, json_dump(rects)), (idx_source, i)
    def gen_info():
        yield 'create_1_classs_label',
        for k, v in frame_info.items():
            yield k, v

    label_file = out_dataset.get_data(out_split, 'label')
    shuffle_file = out_dataset.get_shuffle_file(out_split)
    assert not op.isfile(label_file)
    tsv_writers(gen_rows(), (label_file, shuffle_file))
    gen_tsv = out_dataset.get_gen_info_data(out_split, 'label')
    tsv_writer(gen_info(), gen_tsv)
    write_to_file(cls_name, out_dataset.get_labelmap_file())

    image_tsvs = [TSVDataset(source['data']).get_data(source['split']) for source in
                  data_sources]
    write_to_file('\n'.join(image_tsvs), out_dataset.get_data(out_split + 'X'))
    expand_nested_splitX(out_data, out_split)

def open_each_tsv(data, split, hold=False):
    dataset = TSVDataset(data)
    files = load_list_file(dataset.get_data(split + 'X'))
    all_tsv = []
    for f in files:
        logging.info('open and read {}'.format(f))
        tsv = TSVFile(f)
        tsv[0]
        if hold:
            all_tsv.append(tsv)
        else:
            tsv.close()
        from .common import print_opened_files
        print_opened_files()

def iter_caption_to_json(iter_caption, json_file):
    # save gt caption to json format so thet we can call the api
    key_captions = [(key, json.loads(p)) for key, p in iter_caption]

    info = {
        'info': 'dummy',
        'licenses': 'dummy',
        'type': 'captions',
    }
    info['images'] = [{'file_name': k, 'id': k} for k, _ in key_captions]
    n = 0
    annotations = []
    for k, cs in key_captions:
        for c in cs:
            annotations.append({
                'image_id': k,
                'caption': c['caption'],
                'id': n
            })
            n += 1
    info['annotations'] = annotations
    from .common import write_to_file
    write_to_file(json.dumps(info), json_file)

def compare_2_way_save(iter_pred1, iter_pred2, iter_image, out_data,
                       diff_iou=0.95, inter_iou=0.95, force=False, hint1='1', hint2='2'):
    labels = []
    def gen_rows():
        for _, ((key1, rects1), (key2, rects2), (im_key, _, str_im)) in tqdm(enumerate(zip(
                iter_pred1, iter_pred2, iter_image))):
            assert key1 == key2 == im_key
            unique_rects1 = rects_diff(rects1, rects2, iou=diff_iou)
            unique_rects2 = rects_diff(rects2, rects1, iou=diff_iou)
            k, l = key1 + '_{}'.format(hint1), json_dump(rects1)
            labels.append((k, l))
            yield k, l, str_im
            k, l = key2 + '_{}'.format(hint2), json_dump(rects2)
            labels.append((k, l))
            yield k, l, str_im
            k, l = key1 + '_{}_unique'.format(hint1), json_dump(unique_rects1)
            labels.append((k, l))
            yield k, l, str_im
            k, l = key2 + '_{}_unique'.format(hint2), json_dump(unique_rects2)
            labels.append((k, l))
            yield k, l, str_im
    if force:
        from .common import ensure_remove_dir
        ensure_remove_dir(op.join('data', out_data))
    out_dataset = TSVDataset(out_data)
    assert not op.isdir(out_dataset._data_root) and \
            not op.islink(out_dataset._data_root)
    out_dataset.write_data(gen_rows(), 'test')
    out_dataset.write_data(labels, 'test', 'label')
    populate_dataset_details(out_data)

def create_caption_map(data, split, version):
    dataset = TSVDataset(data)
    all_cap = []
    for k, ss in tqdm(dataset.iter_data(
        split,
        'caption',
        version=version,
    )):
        rects = json.loads(ss)
        for r in rects:
            all_cap.append(r['caption'])
    all_cap = list(set(all_cap))
    dataset.write_data(((c,)for c in all_cap), split, 'captionmap',
                       version=version)

def create_filter_dataset(src_data, excludes, dst_data):
    filters = []
    data, split = src_data, 'train'
    for e in excludes:
        filter_file = 'data/remove_duplicate/target_{}_{}_exclude_{}_{}_0.995_filter_by_pixel.tsv'.format(
            data, split, e['data'], e['split']
        )
        filters.append(filter_file)

    ensure_directory(op.join('data', dst_data))
    for f in glob.glob(op.join('data', src_data, 'trainX*')):
        dest_f = op.join('data', dst_data, op.basename(f))
        ensure_copy_file(f, dest_f)

    shuffle = [(int(i), int(j)) for i, j in tsv_reader(op.join('data', src_data, 'train.shuffle.txt'))]

    idx = []
    for f in filters:
        idx.extend([int(idx_train) for idx_test, idx_train in tsv_reader(f)])
    all_idx = set(range(len(shuffle)))
    logging.info('before {}'.format(len(all_idx)))
    all_idx = all_idx.difference(idx)
    logging.info('after {}'.format(len(all_idx)))
    if len(idx) > 0:
        assert max(idx) < len(shuffle)
    dest_shuffle = [shuffle[i] for i in all_idx]

    tsv_writer(dest_shuffle, op.join('data', dst_data, 'train.shuffle.txt'))

    #generate_key_idximage_idxcaption(dst_data, 'train', None)

def remove_duplicate_loader(exclude, train_data, train_split):
    #dataset = TSVDataset('TaxFlickr30K')
    #dataset = TSVDataset('TaxVQA')
    num_workers = 64
    batch_size = 32
    use_hist = False
    threshold = 0.995
    out_file = 'data/remove_duplicate/target_{}_{}_exclude_{}_{}_{}_filter_by_hist.tsv'.format(
        train_data,
        train_split,
        exclude['data'],
        exclude['split'],
        threshold)
    out_file2 = 'data/remove_duplicate/target_{}_{}_exclude_{}_{}_{}_filter_by_pixel.tsv'.format(
        train_data,
        train_split,
        exclude['data'],
        exclude['split'],
        threshold)
    if File.isfile(out_file2):
        return
    def get_hist(data):
        image = data['image']
        all_hist = []
        h, w = image.shape[:2]
        if use_hist:
            for i in range(3):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                hist = torch.tensor(hist)
                all_hist.append(hist)

                hist = cv2.calcHist([image[:h//2]], [i], None, [256], [0, 256])
                hist = torch.tensor(hist)
                all_hist.append(hist)

                hist = cv2.calcHist([image[h//2:]], [i], None, [256], [0, 256])
                hist = torch.tensor(hist)
                all_hist.append(hist)

                hist = cv2.calcHist([image[:, :w//2, :]], [i], None, [256], [0, 256])
                hist = torch.tensor(hist)
                all_hist.append(hist)

                hist = cv2.calcHist([image[:, w//2:, :]], [i], None, [256], [0, 256])
                hist = torch.tensor(hist)
                all_hist.append(hist)

            hist = torch.cat(all_hist, dim=0)
            hist = torch.nn.functional.normalize(hist, dim=0)
            #ipdb> pp hist.shape
            #torch.Size([256, 1])
            data['hist'] = hist.squeeze()
        else:
            image = torch.tensor(cv2.resize(image, (64, 64)))
            image = image.view(1, -1)
            image = torch.nn.functional.normalize(image.float(), dim=1)
            image = image.squeeze()
            data['hist'] = image
        del data['image']
        return data

    hists = []
    info = exclude
    loader = create_tsv_dataset_loader(
        info['data'], info['split'],
        batch_size=batch_size,
        num_workers=num_workers,
        extra_transform=get_hist,
    )
    for info in tqdm(loader, total=len(loader)):
        hist = info['hist']
        hists.append(hist)
    hists = torch.cat(hists, dim=0)

    data_loader = create_tsv_dataset_loader(
        train_data, train_split,
        extra_transform=get_hist,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    #device = 'cuda'
    device = 'cpu'
    hists = hists.to(device)
    def gen_rows():
        for i, info in tqdm(enumerate(data_loader), total=len(data_loader)):
            hist = info['hist']
            hist = hist.squeeze(-1)

            hist = hist.to(device)

            sim = torch.matmul(hists, hist.t())
            idx = sim > threshold
            del sim

            idx_test, idx_train = idx.nonzero(as_tuple=True)
            idx_train = idx_train + i * batch_size
            for nz_test, nz_train  in zip(idx_test.to('cpu'), idx_train.to('cpu')):
                yield int(nz_test), int(nz_train)
    tsv_writer(gen_rows(), out_file)

    remove_duplicate_double_check_im(
        exclude['data'], exclude['split'], train_data, out_file, out_file2,
        1-threshold,
    )

def remove_duplicate_double_check_im(
        exclude_data, exclude_split, train_data, tsv_fn, out_tsv, threshold):
    row_processor = CheckDuplicate(exclude_data, exclude_split, train_data,
                                   threshold)
    parallel_tsv_process(row_processor, tsv_fn, out_tsv, 64)

class CheckDuplicate(object):
    def __init__(self, exclude_data, exclude_split, train_data, threshold):
        from .tsv_io import TSVSplitProperty
        self.tsv_exclude = TSVSplitProperty(exclude_data, exclude_split)
        self.tsv_train = TSVSplitProperty(train_data, 'train')
        self.threshold = threshold

    def __call__(self, row):
        idx_exclude, idx_train = map(int, row)
        im_exclude = self.tsv_exclude[idx_exclude][-1]
        im_train = self.tsv_train[idx_train][-1]
        im_exclude = img_from_base64(im_exclude)
        im_train = img_from_base64(im_train)
        im2 = im_exclude
        im1 = im_train
        if im2.shape != im1.shape:
            im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
        if np.absolute(im1 - im2).mean() < self.threshold * im1.mean():
            return row
        return None


def create_tsv_dataset_loader(
        data, split, extra_transform=None, batch_size=1, num_workers=1):
    from qd.data_layer.loader import create_data_loader
    from qd.data_layer.dataset import (
        CaptionIdxTSVDataset,
        ImageIdxTSVDataset,
        DatasetPlusTransform,
    )
    from qd.data_layer.transform import (
        LoadLabel,
        LoadHW,
        LoadFeature,
        LoadImage,
        LoadCaption,
        IdentifyTextAB,
        RandomPairNegative,
        CaptionTensorizer,
        TokenizeTransform,
        NoChange,
        PrepareLabel,
        RemoveUselessKeys,
        RenameKey,
        AppendDummyFeature,
    )
    from torchvision.transforms import transforms
    tsv = ImageIdxTSVDataset(data, split)
    loader = LoadImage(data, split)
    hw = LoadHW(data, split)
    useless = RemoveUselessKeys(['dataset'])
    ts = [
        loader,
        hw,
        useless,
    ]
    if extra_transform is not None:
        ts.append(extra_transform)
    transform = transforms.Compose(ts)
    return create_data_loader(tsv, transform, batch_size, num_workers=num_workers)

def parse_hw_column(s_hw):
    try:
        h, w = map(int, s_hw.split(' '))
        return h, w
    except:
        info = json.loads(s_hw)
        if isinstance(info, list):
            assert len(info) == 1
            info = info[0]
        return info['height'], info['width']

def limit_image_size_with_filter(in_rows, min_size, max_size, resave=True, quality=None):
    assert len(in_rows) == 2
    result = limit_image_size(in_rows[:1], min_size, max_size, resave=True, quality=None)
    if result is None:
        return None
    else:
        assert result[0][0] == in_rows[1][0]
        return result[0], in_rows[1]

def limit_image_size(in_rows, min_size, max_size, resave=True, quality=None):
    #debug = True
    debug = False
    if len(in_rows) == 2:
        in_img_row, in_hw_row = in_rows
        h, w = parse_hw_column(in_hw_row[1])
    else:
        in_img_row = in_rows[0]
        h, w = None, None
    img = img_from_base64(in_img_row[-1])
    if img is None:
        img = pilimg_from_base64(in_img_row[-1])
        if img is None:
            return None
        img = np.array(img)
        if img.size == 0:
            return None
        # Convert RGB to BGR
        img = img[:, :, ::-1].copy()
    if h is None:
        h, w = img.shape[:2]
    if w != img.shape[1] or h != img.shape[0]:
        logging.info('{}: w = {}; h = {}; img.shape = {}'.format(
            in_img_row[0], w, h, img.shape))
        w = img.shape[1]
        h = img.shape[0]
    image_min = min(h, w)
    image_max = max(h, w)
    if image_min < min_size and image_max < max_size:
        if resave or quality:
            im = img_from_base64(in_img_row[-1])
            in_img_row[-1] = encoded_from_img(im, quality=quality)
        return in_img_row,
    scale = min(min_size * 1. / image_min, max_size * 1. / image_max)
    w2 = int(round(scale * w))
    h2 = int(round(scale * h))
    w2 = max(1, w2)
    h2 = max(1, h2)
    #img = pilimg_from_base64(in_img_row[-1])
    #img = ImageOps.exif_transpose(img)
    if debug:
        img.show()
    img = cv2.resize(img, (w2, h2),
                     interpolation=cv2.INTER_AREA,
                     )
    #save_image(img, '/tmp/b.jpg')
    #img = img.resize((w2, h2))
    in_img_row[-1] = encoded_from_img(img, quality=quality)
    if debug:
        img.show()
    return in_img_row,
    # we need to do resizing

def parallel_limit_image_size(im_tsv, hw_tsv, min_size, max_size,
                              quality,
                              out_tsv):
    process_func = lambda x: limit_image_size(x, min_size, max_size,
                                              quality=quality)
    num_process = 64
    if hw_tsv is not None:
        parallel_tsv_process_NtoN(process_func, (im_tsv, hw_tsv), (out_tsv, ),
                                  num_process)
    else:
        parallel_tsv_process_NtoN(process_func, (im_tsv,), (out_tsv, ),
                                  num_process)

def parallel_limit_image_size_with_filter(im_tsv, other_tsv, min_size, max_size,
                              quality,
                              out_tsv, out_other_tsv):
    process_func = lambda x: limit_image_size_with_filter(
        x, min_size, max_size, quality=quality)
    num_process = 64
    parallel_tsv_process_NtoN(process_func, (im_tsv, other_tsv), (out_tsv, out_other_tsv), num_process)

def filter_dataset_by_size(data, out_data):
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)
    iter_hw = dataset.iter_data('train', 'hw')

    if op.isfile(dataset.get_shuffle_file('train')):
        iter_shuffle = tsv_reader(dataset.get_shuffle_file('train'))
        info = defaultdict(int)
        def gen_rows():
            for (key, str_hw), s in tqdm(zip(iter_hw, iter_shuffle)):
                try:
                    h, w = map(int, str_hw.split(' '))
                except:
                    x = json.loads(str_hw)
                    if isinstance(x, list):
                        x = x[0]
                    h, w = x['height'], x['width']
                r = 1. * h / w
                if r > 4 or r < 1. / 4:
                    info['ratio_too_large'] += 1
                    continue
                if min(h, w) < 64:
                    info['size_too_small'] += 1
                    continue
                yield s
        tsv_writer(gen_rows(), out_dataset.get_shuffle_file('train'))
        for t in [None, 'hw', 'caption']:
            ensure_copy_file(
                dataset.get_data('trainX', t),
                out_dataset.get_data('trainX', t),
            )
    else:
        info = defaultdict(int)
        def gen_rows():
            for i, (key, str_hw) in tqdm(enumerate(iter_hw)):
                try:
                    h, w = map(int, str_hw.split(' '))
                except:
                    x = json.loads(str_hw)
                    if isinstance(x, list):
                        x = x[0]
                    h, w = x['height'], x['width']
                r = 1. * h / w
                if r > 4 or r < 1. / 4:
                    info['ratio_too_large'] += 1
                    continue
                if min(h, w) < 64:
                    info['size_too_small'] += 1
                    continue
                yield (0, i)
        tsv_writer(gen_rows(), out_dataset.get_shuffle_file('train'))
        for t in [None, 'hw', 'caption']:
            write_to_file(
                dataset.get_data('train', t),
                out_dataset.get_data('trainX', t),
            )

def limit_dataset_image_size_to_single(
        data, split, image_t, min_size, max_size,
        quality, out_data):
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)

    process_func = lambda x: limit_image_size(
        x, min_size, max_size, quality=quality)

    i = TSVSplitProperty(data, split)
    h = TSVSplitProperty(data, split, 'hw')
    o = out_dataset.get_data(split, image_t)
    if not op.isfile(o):
        parallel_tsv_process_NtoN(process_func, (i, h), (o, ), 64)

    for t in ['label', 'caption']:
        if dataset.has(split, t):
            tsv_writer(
                TSVSplitProperty(data, split, t),
                out_dataset.get_data(split, t)
            )
    populate_dataset_hw(out_data, splits=[split], img_t=image_t)
    if File.isfile(dataset.get_labelmap_file()):
        ensure_copy_file(
            dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file()
        )

def limit_dataset_image_size(
        data, split, image_t, min_size, max_size,
        quality, out_data):
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)

    if op.isfile(dataset.get_data(split, image_t)):
        in_img_tsv = [dataset.get_data(split, image_t)]
        caption_tsvs = [dataset.get_data(split, 'caption')]
        in_hw_tsv = [dataset.get_data(split, 'hw')]
        out_tsv = [out_dataset.get_data(split, image_t)]
        single_tsv = True
    else:
        in_img_tsv = load_list_file(dataset.get_data(split + 'X', image_t))
        hw_file = dataset.get_data(split + 'X', 'hw')
        if op.isfile(hw_file):
            in_hw_tsv = load_list_file(dataset.get_data(split + 'X', 'hw'))
        else:
            hw_file = dataset.get_data(split, 'hw')
            assert op.isfile(hw_file)
            in_hw_tsv = [hw_file]
            iter_shuffle = tsv_reader(dataset.get_shuffle_file(split))
            for i, (idx_source, idx_row) in enumerate(iter_shuffle):
                assert i == int(idx_row)
                assert int(idx_source) == 0
            assert len(in_img_tsv) == 1
            assert len(TSVFile(hw_file)) == len(TSVFile(in_img_tsv[0]))
        num_split = len(in_img_tsv)
        out_tsv = [out_dataset.get_data(split + '_{}_{}'.format(i, num_split))
                   for i in range(num_split)]
        caption_tsvs = load_list_file(
            dataset.get_data(split + 'X', 'caption')
        )
        label_tsvs = load_list_file(
            dataset.get_data(split + 'X', 'label')
        )
        single_tsv = False

    process_func = lambda x: limit_image_size(
        x, min_size, max_size, quality=quality)

    for i, h, o in zip(in_img_tsv, in_hw_tsv, out_tsv):
        from qd.process_tsv import parallel_tsv_process_NtoN
        if not op.isfile(o):
            parallel_tsv_process_NtoN(process_func, (i, h), (o, ), 64)

    if not single_tsv:
        write_to_file(
            '\n'.join(out_tsv),
            out_dataset.get_data(split + 'X', image_t),
        )
        write_to_file(
            '\n'.join(caption_tsvs),
            out_dataset.get_data(split + 'X', 'caption')
        )
        write_to_file(
            '\n'.join(label_tsvs),
            out_dataset.get_data(split + 'X', 'label')
        )
        tsv_copy(
            dataset.get_shuffle_file(split),
            out_dataset.get_shuffle_file(split),
        )
        splits = ['train_{}_{}'.format(i, num_split) for i in range(num_split)]
        populate_dataset_hw(out_data, splits=splits)
        write_to_file(
            '\n'.join((out_dataset.get_data(
                split + '_{}_{}'.format(i, num_split), 'hw',
            ) for i in range(num_split))),
            out_dataset.get_data(split + 'X', 'hw'),
        )
    else:
        for t in ['label', 'caption']:
            from_f = dataset.get_data(split, t)
            if File.isfile(from_f):
                tsv_copy(
                    dataset.get_data(split, t),
                    out_dataset.get_data(split, t)
                )
        populate_dataset_hw(out_data, splits=[split], img_t=image_t)
    if File.isfile(dataset.get_labelmap_file()):
        ensure_copy_file(
            dataset.get_labelmap_file(),
            out_dataset.get_labelmap_file()
        )

def video_convert_tsv_one(old_tsv, new_tsv):
    def gen_rows():
        for row in tsv_reader(old_tsv):
            #assert len(row) == 34
            idx = (len(row) - 2) // 2 + 2
            yield row[0], row[idx]
    if not op.isfile(new_tsv):
        tsv_writer(gen_rows(), new_tsv)

def get_max_iter_for_vl_plus_l_data(
    data, vl_data_rate, num_epoch_vl, effective_batch_size,
    effective_batch_size_text_only,
):

    iter_cache = TSVDataset(data).iter_data(
        'train',
        'key_idximage_idxcaption_num_valid_image',
    )
    info = {k: int(v) for k, v in iter_cache}
    num_vl_pair = info['num_valid_vl_pair']
    num_l_pair = info['num_l_only_pair']
    max_iter = int(num_vl_pair * num_epoch_vl / effective_batch_size / vl_data_rate + 0.5)
    num_epoch_l = max_iter * (1 - vl_data_rate) * effective_batch_size_text_only / num_l_pair
    logging.info(f'num_epoch_l = {num_epoch_l}; max_iter={max_iter}')
    return max_iter

def generate_num_vl_data(data, split, out_type='key_idximage_idxcaption_num_valid_image'):
    # this is mainly used by VisionLanguagePlusLanguageBatchSampler
    dataset = TSVDataset(data)
    source_list = load_list_file(dataset.get_data(split + 'X'))
    is_valid = tuple([s != 'd' for s in source_list])
    source = TSVSplitProperty(data, split,
                               'key_idximage_idxcaption_idxsource')
    ranges = split_to_chunk_to_range(len(source), num_chunk=64)
    def process(x):
        start, end = x
        info = defaultdict(int)
        with_image = True
        for i in tqdm(range(start, end)):
            s = int(source[i][0])
            if is_valid[s]:
                assert with_image
                info['with_valid_image'] += 1
            elif with_image:
                with_image = False
            info['total'] += 1
        info['with_image'] = with_image
        return info
    infos = parallel_map(process, ranges)
    ret = defaultdict(int)
    for i in infos:
        for k, v in i.items():
            ret[k] += v
    assert ret['total'] == len(source)
    for i, info in enumerate(infos):
        if not info['with_image']:
            for j in range(i + 1, len(infos)):
                assert not info['with_image']

    out_file = dataset.get_data(split, out_type)
    tsv_writer((
        ('num_valid_vl_pair', ret['with_valid_image']),
        ('num_l_only_pair', ret['total'] - ret['with_valid_image']),
        ('num_total', ret['total']),
    ), out_file)

def generate_num_vl_data_old(data, split, out_type='key_idximage_idxcaption_num_valid_image'):
    # this is mainly used by VisionLanguagePlusLanguageBatchSampler
    dataset = TSVDataset(data)
    source_list = load_list_file(dataset.get_data(split + 'X'))
    num_valid_image = 0
    num_total = 0
    with_image = True
    for s, in tqdm(TSVSplitProperty(data, split,
                               'key_idximage_idxcaption_idxsource')):
        if source_list[int(s)] != 'd':
            assert with_image
            num_valid_image += 1
        elif with_image:
            with_image = False
        num_total += 1
    out_file = dataset.get_data(split, out_type)
    tsv_writer((
        ('num_valid_vl_pair', num_valid_image),
        ('num_l_only_pair', num_total - num_valid_image),
        ('num_total', num_total),
    ), out_file)

def generate_num_each_tsv(data, split,
                          idxsource_type='key_idximage_idxcaption_idxsource',
                          out_type='key_idximage_idxcaption_num_each_tsv',
                          ):
    result = []
    dataset = TSVDataset(data)
    #for s, in tqdm(TSVSplitProperty(data, split, 'key_idximage_idxcaption_idxsource')):
            # TSVSplitProperty here is twice slower
    total = dataset.num_rows(split, t=idxsource_type)
    ranges = split_to_chunk_to_range(total, num_chunk=64)

    def chunk_processor(x):
        start, end = x
        tsv = TSVSplitProperty(data, split, idxsource_type)
        info = defaultdict(int)
        prev_s = -1
        for i in tqdm(range(start, end)):
            s = int(tsv[i][0])
            info[s] += 1
            if prev_s == -1:
                prev_s = s
            elif prev_s < s:
                prev_s = s
            else:
                assert prev_s == s, (i, prev_s, s)
        return info

    infos = parallel_map(chunk_processor, ranges, num_worker=64)
    for i in range(len(infos) - 1):
        assert max(infos[i].keys()) <= min(infos[i + 1].keys())
    result = defaultdict(int)
    for i in infos:
        for k, v in i.items():
            result[k] += v

    # we assumed the last tsv is not empty. If it is, it will crash and we need
    # to append more zeros here.
    num_tsv = len(load_list_file(dataset.get_data(split + 'X')))
    assert min(result.keys()) >= 0
    assert max(result.keys()) < num_tsv

    dataset.write_data(((result.get(i, 0),) for i in range(num_tsv)), split, out_type,)

def generate_num_each_tsv_old(data, split, idxsource_type='key_idximage_idxcaption_idxsource'):
    result = []
    curr_num = 0
    curr_idx = -1
    dataset = TSVDataset(data)
    #for s, in tqdm(TSVSplitProperty(data, split, 'key_idximage_idxcaption_idxsource')):
            # TSVSplitProperty here is twice slower
    total = dataset.num_rows(split, t=idxsource_type)
    with File.open(dataset.get_data(split, idxsource_type), 'r') as fp:
        for line in tqdm(fp, total=total):
            s = int(line)
            if s != curr_idx:
                if curr_idx != -1:
                    # the order should not be random
                    assert len(result) == curr_idx
                    result.append(curr_num)
                while curr_idx + 1 < s:
                    result.append(0)
                    curr_idx += 1
                assert s == curr_idx + 1, (curr_idx, s)
                curr_idx = s
                curr_num = 0
            curr_num += 1

    assert len(result) == curr_idx, (len(result), curr_idx)
    result.append(curr_num)

    # we assumed the last tsv is not empty. If it is, it will crash and we need
    # to append more zeros here.
    num_tsv = len(load_list_file(dataset.get_data(split + 'X')))
    assert num_tsv == len(result), (num_tsv, len(result))
    dataset.write_data(((r,) for r in result),
                       split,
                       'key_idximage_idxcaption_num_each_tsv',
                       )

#@deprecated('use generate_idximage_idxcaption_idxsource_parallel')
#def generate_key_idximage_idxcaption_idxsource(data, split):
    #logging.info(f'{data}/{split}')
    #generate_key_idximage_idxcaption(data, split)
    #k_img_cap = TSVSplitProperty(data, split, 'key_idximage_idxcaption')
    #dataset = TSVDataset(data)
    #if dataset.has(split, 'key_idximage_idxcaption_idxsource'):
        #return
    #x_tsv = dataset.get_data(split + 'X', 'caption')
    #if File.isfile(x_tsv):
        #all_idx_source = [int(idx_source) for idx_source, _ in
                          #tqdm(TSVFile(dataset.get_shuffle_file(split)))]
        #logging.info('loading composite source idx')
        #result = ((all_idx_source[int(idx_img)],) for _, idx_img, _ in
                  #tqdm(k_img_cap))
        #dataset.write_data(result, split, 'key_idximage_idxcaption_idxsource')
    #else:
        #dataset.write_data([(0,)] * len(k_img_cap), split, 'key_idximage_idxcaption_idxsource')

def generate_idximage_idxcaption_idxsource(data, split):
    logging.info(f'{data}/{split}')
    generate_idximage_idxcaption_parallel(data, split)
    k_img_cap = TSVSplitProperty(data, split, 'idximage_idxcaption')
    dataset = TSVDataset(data)
    if dataset.has(split, 'idximage_idxcaption_idxsource'):
        return
    x_tsv = dataset.get_data(split + 'X', 'caption')
    if File.isfile(x_tsv):
        shuffle_tsv = TSVFile(dataset.get_shuffle_file(split))
        def row_processor(row):
            key, idximage, _ = row
            ret = int(shuffle_tsv[int(idximage)][0])
            return (ret, )
        parallel_tsv_process(
            row_processor,
            k_img_cap,
            dataset.get_data(split, 'idximage_idxcaption_idxsource'),
            64,
        )
    else:
        dataset.write_data([(0,)] * len(k_img_cap), split, 'idximage_idxcaption_idxsource')

@deprecated('use generate_idxsource_of_idximage_idxcaption')
def generate_idxsource_of_idximage_idxcaption_parallel(*args, **kwargs):
    generate_idxsource_of_idximage_idxcaption(*args, **kwargs)

def generate_idxsource_of_idximage_idxcaption(
    data, split, in_type='key_idximage_idxcaption', out_type='key_idximage_idxcaption_idxsource'):
    logging.info(f'{data}/{split}')
    if in_type == 'key_idximage_idxcaption':
        generate_key_idximage_idxcaption(data, split)
    elif in_type == 'idximage_idxcaption':
        generate_idximage_idxcaption(data, split)
    k_img_cap = TSVSplitProperty(data, split, in_type)
    dataset = TSVDataset(data)
    if dataset.has(split, out_type):
        return
    x_tsv = dataset.get_data(split + 'X', 'caption')
    if File.isfile(x_tsv):
        shuffle_tsv = TSVFile(dataset.get_shuffle_file(split))
        def row_processor(row):
            idximage = row[-2]
            ret = int(shuffle_tsv[int(idximage)][0])
            return (ret, )
        parallel_tsv_process(
            row_processor,
            k_img_cap,
            dataset.get_data(split, out_type),
            64,
        )
    else:
        dataset.write_data([(0,)] * len(k_img_cap), split, out_type)


@deprecated('use generate_idxsource_of_idximage_idxcaption_parallel')
def generate_key_idximage_idxcaption_idxsource_parallel(*args, **kwargs):
    generate_idxsource_of_idximage_idxcaption_parallel(*args, **kwargs)

@deprecated('use generate_idxsource_of_idximage_idxcaption_parallel')
def generate_key_idximage_idxcaption_idxsource(*args, **kwargs):
    generate_idxsource_of_idximage_idxcaption_parallel(*args, **kwargs)


def generate_cache_tsv(data):
    dataset = TSVDataset(data)
    all_file = []
    for t in [None, 'caption']:
        files = load_list_file(dataset.get_data('trainX', t=t))
        all_file.extend(files)
    all_file = [f for f in all_file if f != 'd']
    all_file.extend([get_tsv_associates(f)[0] for f in all_file])
    ret = ','.join(all_file)
    return ret

def get_precache_tsv_files(data):
    dataset = TSVDataset(data)
    all_fs = []
    for t in [None, 'caption']:
        if File.isfile(dataset.get_data('trainX', t=t)):
            fs = load_list_file(dataset.get_data('trainX', t=t))
            fs = [f for f in fs if f != 'd']
            fs.append(dataset.get_shuffle_file('train'))
        else:
            fs = [dataset.get_data('train', t=t)]
        fs.extend([x for f in fs for x in get_tsv_associates(f)])
        all_fs.extend(fs)
    return ','.join(set(all_fs))

def shuffle_dataset_by_file(in_data, out_data):
    in_dataset = TSVDataset(in_data)
    out_dataset = TSVDataset(out_data)
    ts = [None, 'caption']
    old2new = None
    random.seed(666)
    for t in ts:
        from_file = in_dataset.get_data('trainX', t=t)
        out_file = out_dataset.get_data('trainX', t=t)
        fnames = load_list_file(from_file)
        if old2new is None:
            old2new = list(range(len(fnames)))
            random.shuffle(old2new)
        new_out = [None] * len(fnames)
        for i, j in enumerate(old2new):
            new_out[j] = fnames[i]
        write_to_file('\n'.join(new_out), out_file)

    def mapper(ret, row, idx):
        i = int(row[0])
        if 'x' not in ret:
            ret['x'] = []
        ret = ret['x']
        if len(ret) == 0 or ret[-1]['img_tsv_idx'] != i:
            ret.append({
                'start': idx,
                'end': idx,
                'img_tsv_idx': i,
            })
        else:
            assert ret[-1]['img_tsv_idx'] == i
            ret[-1]['end'] = idx

    def reducer(rets):
        result = []
        for ret in rets:
            r = ret['x']
            for _i, curr_r in enumerate(r):
                if len(result) == 0:
                    result.append(curr_r)
                elif result[-1]['img_tsv_idx'] == curr_r['img_tsv_idx']:
                    assert result[-1]['end'] == curr_r['start'] - 1
                    result[-1]['end'] = curr_r['end']
                else:
                    result.append(curr_r)
        return result

    results = parallel_tsv_aggregator(
        mapper,
        reducer,
        in_tsv_file=in_dataset.get_shuffle_file('train'),
        with_row_idx=True,
        num_process=64,
    )

    for i in range(len(results) - 1):
        assert results[i]['img_tsv_idx'] < results[i + 1]['img_tsv_idx']
    random.shuffle(results)
    def process(r):
        folder = get_tmp_folder()
        new_idx_img = old2new[r['img_tsv_idx']]
        tmp_file = '{}/{}.tsv'.format(folder, new_idx_img)
        def gen_rows():
            in_dataset = TSVDataset(in_data)
            tsv = TSVFile(in_dataset.get_shuffle_file('train'))
            for i in range(r['start'], r['end'] + 1):
                old_idx_img, idx_caption = tsv[i]
                assert old_idx_img == str(r['img_tsv_idx'])
                yield new_idx_img, idx_caption
        tsv_writer(gen_rows(), tmp_file)

    parallel_map(process, results, num_worker=64)
    folder = get_tmp_folder()
    tmp_files = ['{}/{}.tsv'.format(folder, i) for i in range(len(results))]
    logging.info(len(tmp_files))
    tmp_files = [t for t in tmp_files if op.isfile(t)]
    logging.info(len(tmp_files))
    concat_tsv_files(tmp_files, out_dataset.get_shuffle_file('train'))
    delete_tsv_files(tmp_files)

def populate_vl_dataset(data, split='train'):
    generate_idximage_idxcaption(data, split)
    from qd.process_tsv import generate_idxsource_of_idximage_idxcaption
    generate_idxsource_of_idximage_idxcaption(
        data, split,
        in_type='idximage_idxcaption',
    )
    from qd.process_tsv import generate_num_each_tsv
    generate_num_each_tsv(data, split)
    from qd.process_tsv import generate_num_vl_data
    generate_num_vl_data(data, split)

def create_simple_tax(data):
    out_data = 'Tax' + data
    dataset = TSVDataset(data)
    out_dataset = TSVDataset(out_data)
    for split in ['train', 'val', 'test', 'trainval']:
        if not dataset.has(split):
            continue
        for t in [None, 'caption']:
            src_file = dataset.get_data(split, t)
            write_to_file(
                src_file,
                out_dataset.get_data(split + 'X', t)
            )
        num = dataset.num_rows(split)
        tsv_writer(((0, i) for i in range(num)),
                   out_dataset.get_shuffle_file(split))

def convert_cap_to_tag(predict_file, tag_predict):
    def gen_rows():
        if isinstance(predict_file, str):
            iter_pred = tsv_reader(predict_file)
        else:
            iter_pred = predict_file
        for key, s in iter_pred:
            s = json.loads(s)
            assert len(s) == 1
            tags = map(lambda x: x.strip(), s[0]['caption'].split(','))
            ret = [{'class': t, 'conf': 1.} for t in tags]
            yield key, json_dump(ret)
    tsv_writer(gen_rows(), tag_predict)

def create_question_annotation_json(data, split):
    gt = TSVSplitProperty(data, split, 'caption')
    questions = []
    annotations = []
    for image_id, s in gt:
        s = json.loads(s)
        for i in s:
            q = {
                'image_id': image_id,
                'question_id': i.get('question_id'),
                'question': i['question'],
            }
            questions.append(q)
            a = copy.deepcopy(q)
            answers = [{
                'answer': ans,
            } for ans in i['answers']]
            a['answers'] = answers
            a['question_type'] = 'null'
            a['answer_type'] = 'null'
            annotations.append(a)

    # list of dict (image_id, question, question_id), image_id and question_id
    # are integer
    question_info = {
        'info': None,
        'task_type': 'Open-Ended',
        'data_type': 'mscoco',
        'license': None,
        'data_subtype': None,
        'questions': questions,
    }
    annotation_info = {
        'info': None,
        'task_type': 'Open-Ended',
        'data_type': 'mscoco',
        'license': None,
        'data_subtype': None,
        'annotations': annotations,
    }
    out_f = f'data/{data}/{split}.caption.question.json'
    write_to_file(json_dump(question_info), out_f)
    out_f = f'data/{data}/{split}.caption.annotation.json'
    write_to_file(json_dump(annotation_info), out_f)

if __name__ == '__main__':
    from .common import parse_general_args
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
