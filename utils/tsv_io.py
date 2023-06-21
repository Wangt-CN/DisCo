import contextlib
import mmap
from .common import get_file_size
from .common import exclusive_open_to_read
import time
import multiprocessing as mp
from .common import limited_retry_agent
import logging
from pprint import pformat
import glob
import json
import random
from .common import ensure_directory
from .common import robust_open_to_write
from .common import copy_file
from .common import worth_create
from .common import hash_sha1
from .common import get_user_name
from .common import load_from_yaml_str
from .common import get_all_path, dict_get_path_value, dict_update_path_value
import six
import os
import os.path as op
from azfuse import File
import shutil
import re
try:
    from itertools import izip as zip
except ImportError:
    # python 3
    pass
#import progressbar
from .common import qd_tqdm as tqdm


def generate_lineidx(filein, idxout):
    with File.open(filein, 'r') as tsvin, File.open(idxout,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        fbar_last_pos = 0
        fbar = tqdm(total=fsize, unit_scale=True)
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n");
            tsvin.readline()
            fpos = tsvin.tell();
            fbar.update(fpos - fbar_last_pos)
            fbar_last_pos = fpos

def tsv_exists(o):
    return File.isfile(o) and all(File.isfile(f) for f in get_tsv_associates(o))

def get_default_splits():
    return ['train', 'trainval', 'test', 'val']

def get_tsv_lineidx(tsv_file):
    #assert tsv_file.endswith('.tsv') or tsv_file.endswith('.txt')
    return tsv_file[:-3] + 'lineidx'

def get_tsv_lineidx_8b(tsv_file):
    assert tsv_file.endswith('tsv') or tsv_file.endswith('txt')
    return tsv_file[:-3] + 'lineidx.8b'

def get_tsv_associates(tsv_file):
    return [
        #get_tsv_lineidx(tsv_file),
        get_tsv_lineidx_8b(tsv_file)
    ]

def rm_tsv(tsv_file):
    if op.isfile(tsv_file):
        os.remove(tsv_file)
        for line_idx in [get_tsv_lineidx(tsv_file),
                         get_tsv_lineidx_8b(tsv_file)]:
            if op.isfile(line_idx):
                os.remove(line_idx)

def tsv_rm(tsv_file):
    rm_tsv(tsv_file)

def tsv_copy(src_tsv, dst_tsv):
    copy_file(src_tsv, dst_tsv)
    for s, t in zip(get_tsv_associates(src_tsv), get_tsv_associates(dst_tsv)):
        if op.isfile(s):
            copy_file(s, t)

def tsv_mv(src_file, dst_file):
    shutil.move(src_file, dst_file)
    for s, t in zip(get_tsv_associates(src_file), get_tsv_associates(dst_file)):
        if op.isfile(s):
            shutil.move(s, t)

def reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file):
    tsv = TSVFile(in_tsv_file)
    logging.info('loading keys in input')
    keys = [tsv.seek_first_column(i) for i in tqdm(range(len(tsv)), mininterval=2)]
    key_to_idx = {key: i for i, key in enumerate(keys)}
    def gen_rows():
        logging.info('writing')
        for key in tqdm(ordered_keys, mininterval=2):
            idx = key_to_idx[key]
            yield tsv.seek(idx)
    tsv_writer(gen_rows(), out_tsv_file)

def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != b'' and s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return b''.join(result)

class EqualSplitTSVFile(object):
    def __init__(self, list_file):
        self.list_tsv = TSVFile(list_file)
        self.split_size = None
        self.split2tsv = None

    def ensure_initialized(self):
        if self.split_size is None:
            num_split = len(self.list_tsv)
            self.split2tsv = [None for _ in range(num_split)]
            self.split2tsv[0] = TSVFile(self.list_tsv[0][0])
            self.split_size = len(self.split2tsv[0])
            if num_split == 1:
                self.len = self.split_size
            else:
                self.split2tsv[-1] = TSVFile(self.list_tsv[num_split - 1][0])
                self.len = (num_split - 1) * self.split_size + len(self.split2tsv[-1])

    def __iter__(self):
        for i in range(len(self.split2tsv)):
            if self.split2tsv[i] is None:
                self.split2tsv[i] = TSVFile(self.list_tsv[i][0])
            tsv = self.split2tsv[i]
            for x in tsv:
                yield x

    def __getitem__(self, index):
        self.ensure_initialized()
        idx_split = index // self.split_size
        offset = index % self.split_size
        if self.split2tsv[idx_split] is None:
            self.split2tsv[idx_split] = TSVFile(self.list_tsv[idx_split][0])
        return self.split2tsv[idx_split][offset]

    def __len__(self):
        self.ensure_initialized()
        return self.len

class CompositeTSVFile(object):
    def __init__(self, list_file, seq_file, cache_policy=False,
                 hold_buffer=0, data_dir=None,
                 ):
        # list_file can be a loaded or constructed pair of index, rather than a
        # filename to load. In this case, seq_file will be a list of dataset,
        # which should implement len() and __getitem__() so that we can
        # reference it.
        self.seq_file = seq_file
        self.list_file = list_file
        self.cache_policy = cache_policy
        self.seq = None
        self.tsvs = []
        # please do ont call ensure_initialized here. we wil always do it
        # lazily. we may load a huge amount of seq, which could be super slow
        # when spawning multiple processes.

        # this means, how many tsv fp pointer we will hold. If it is 0 or less
        # than 0, we will hold all fp pointers we need. If it is larger than 0,
        # we only hold some, which are kept in self.hold_sources
        self.hold_buffer = hold_buffer
        self.hold_sources = []
        self.data_dir = data_dir

    def __repr__(self):
        return 'CompositeTSVFile(list_file={}, seq_file={})'.format(
            self.seq_file,
            self.list_file
        )

    def get_row_len(self, i):
        self.ensure_initialized()
        idx_source, idx_row = map(int, self.seq[i])
        result = self.tsvs[idx_source].get_row_len(idx_row)
        return result

    def get_key(self, index):
        # added by Linjie
        self.ensure_initialized()
        idx_source, idx_row = map(int, self.seq[index])
        k = self.tsvs[idx_source].get_key(idx_row)
        return '_'.join([self.list_file[idx_source], k])

    def __getitem__(self, index):
        self.ensure_initialized()
        idx_source, idx_row = map(int, self.seq[index])
        start = time.time()
        result = self.tsvs[idx_source].seek(idx_row)
        end = time.time()
        if end - start > 10:
            logging.warning('too long to load fname = {}, source={}, row={}, time={}'.format(
                self.tsvs[idx_source],
                idx_source,
                idx_row,
                end - start
            ))
        if self.hold_buffer > 0 and idx_source not in self.hold_sources:
            if len(self.hold_sources) >= self.hold_buffer:
                close_idx_source = self.hold_sources.pop(0)
                self.tsvs[close_idx_source].close_fp()
            self.hold_sources.append(idx_source)
        return result

    def __len__(self):
        self.ensure_initialized()
        return len(self.seq)

    def num_rows(self):
        # added by Linjie
        self.ensure_initialized()
        return len(self.seq)

    def __iter__(self):
        self.ensure_initialized()
        self.next_row = 0
        for idx_source, idx_row in self.seq:
            idx_source, idx_row = int(idx_source), int(idx_row)
            yield self.tsvs[idx_source][idx_row]

    def release(self):
        # this is to ensure we released all the resources
        self.seq = None
        for t in self.tsvs:
            t.close()

    def close(self):
        self.release()

    def seek_first_column(self, index):
        self.ensure_initialized()
        idx_source, idx_row = map(int, self.seq[index])
        return self.tsvs[idx_source].seek_first_column(idx_row)

    def get_composite_source_idx(self):
        return [int(i) for i, _ in self.seq]

    def is_from_valid_file(self, idx=None):
        self.ensure_initialized()
        if idx is None:
            return [self.tsvs[int(idx_source)] is not None for idx_source, _ in tqdm(self.seq)]
        else:
            return self.tsvs[int(self.seq[idx][0])] is not None

    def ensure_initialized(self):
        if self.seq is None:
            if isinstance(self.list_file, str) and \
                    isinstance(self.seq_file, str):
                self.seq = TSVFile(self.seq_file)
                if self.data_dir is not None:
                    self.tsvs = [TSVFile(f.replace('data', self.data_dir), self.cache_policy) if f != 'd' else None for f in load_list_file(self.list_file)]
                else:
                    self.tsvs = [TSVFile(f, self.cache_policy) if f != 'd' else None for f in load_list_file(self.list_file)]
            else:
                self.seq = self.list_file
                self.tsvs = self.seq_file

# wrapper of lineidx.8b
class LongFile(object):
    def __init__(self, fname):
        self.fname = fname
        self._len = None
        self.fp = None
        self.pid = None

    def __len__(self):
        if self._len is None:
            self._len = File.get_file_size(self.fname) // 8
        return self._len

    def __getitem__(self, index):
        self.ensure_opened()
        self.fp.seek(index * 8)
        return int.from_bytes(self.fp.read(8), 'little')

    def ensure_opened(self):
        if self.fp is None:
            self.fp = File.open(self.fname, 'rb')
            self.pid = os.getpid()
        if self.pid != os.getpid():
            self.fp.close()
            logging.info('re-open {} because the process id changed'.format(
                self.fname))
            self.fp= File.open(self.fname, 'rb')
            self.pid = os.getpid()

class TSVFile(object):
    def __init__(self, tsv_file, cache_policy=None):
        self.tsv_file = tsv_file
        #print(tsv_file)
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self.lineidx_8b = self.lineidx + '.8b'
        self._fp = None
        self._mfp = None
        self._lineidx = None
        self.fp8b = None
        self.cache_policy= cache_policy
        self.close_fp_after_read = False
        if os.environ.get('QD_TSV_CLOSE_FP_AFTER_READ'):
            self.close_fp_after_read = bool(os.environ['QD_TSV_CLOSE_FP_AFTER_READ'])
        self.use_mmap = False
        if os.environ.get('QD_TSV_MMAP'):
            self.use_mmap = int(os.environ['QD_TSV_MMAP'])
        self.use_fuse = False
        #if os.environ.get('QD_TSV_USE_FUSE'):
            #self.use_fuse = int(os.environ['QD_TSV_USE_FUSE'])
            #from qd.cloud_storage import create_cloud_fuse
            #self.fuser = create_cloud_fuse()
        self.has_lineidx_8b = int(os.environ.get('QD_USE_LINEIDX_8B', '0'))
        # the process always keeps the process which opens the
        # file. If the pid is not equal to the currrent pid, we will re-open
        # teh file.
        self.pid = None
        self.lineidx_8b_pid = None
        self.open_once = False

        self._len = None
        self._tsv_file_size = None
        print(f'{self.__repr__()}')

    @property
    def tsv_file_size(self):
        if self._tsv_file_size is None:
            self._tsv_file_size = File.get_file_size(self.tsv_file)
        return self._tsv_file_size

    def get_row_len(self, i):
        start = self.get_offset(i)
        if i < len(self) - 1:
            end = self.get_offset(i + 1)
        else:
            end = self.tsv_file_size
        return end - start

    def get_row_offsets(self, i):
        start = self.get_offset(i)
        if i < len(self) - 1:
            end = self.get_offset(i + 1)
        else:
            end = self.tsv_file_size
        return start, end

    def close_fp(self):
        if self._fp:
            self._fp.close()
            self._fp = None
        if self._mfp:
            self._mfp.close()
            self._mfp = None
        if self.has_lineidx_8b and self.fp8b:
            self.fp8b.close()
            self.fp8b = None

    def release(self):
        self.close_fp()
        self._lineidx = None

    def close(self):
        #@deprecated('use release to make it more clear not to release lineidx')
        self.close_fp()

    def __del__(self):
        self.release()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        self._ensure_tsv_opened()
        self.fp_seek(0)
        if not self.use_mmap:
            for line in self._fp:
                result = [s.strip() for s in line.decode().split('\t')]
                yield result
        else:
            while True:
                line = self._mfp.readline()
                if line == b'':
                    break
                result = [s.strip() for s in line.decode().split('\t')]
                yield result

    def num_rows(self):
        if self._len is None:
            if self.has_lineidx_8b:
                self._len = File.get_file_size(self.lineidx_8b) // 8
            else:
                self._ensure_lineidx_loaded()
                self._len = len(self._lineidx)
        return self._len

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def get_current_column(self):
        if self.use_mmap:
            result = [s.strip() for s in self._mfp.readline().decode().split('\t')]
        else:
            result = [s.strip() for s in self._fp.readline().split('\t')]
        return result

    def get_current_column2(self, size):
        if self.use_mmap:
            result = [s.strip() for s in self._mfp.read(size).decode().split('\t')]
        else:
            result = [s.strip() for s in self._fp.read(size).decode().split('\t')]
        return result

    def fp_seek(self, pos):
        if self.use_mmap:
            self._mfp.seek(pos)
        else:
            self._fp.seek(pos)

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos, end = self.get_row_offsets(idx)
        self.fp_seek(pos)
        result = self.get_current_column2(end - pos)
        if self.close_fp_after_read:
            self.close_fp()
        return result

    def seek3(self, idx):
        self._ensure_tsv_opened()
        pos = self.get_offset(idx)
        self.fp_seek(pos)
        result = self.get_current_column()
        if self.close_fp_after_read:
            self.close_fp()
        return result

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        pos = self.get_offset(idx)
        self._fp.seek(pos)
        return read_to_character(self._fp, b'\t').decode()

    def seek_first_columns(self):
        # assert self.has_lineidx_8b
        self._ensure_tsv_opened()
        result = []
        if self.has_lineidx_8b:
            self.ensure_lineidx_8b_opened()
            for idx in range(len(self)):
                self.fp8b.seek(idx * 8)
                pos = int.from_bytes(self.fp8b.read(8), 'little')
                self._fp.seek(pos)
                result.append(read_to_character(self._fp, b'\t').decode())
        else:
            self._ensure_lineidx_loaded()
            for idx in range(len(self)):
                pos = self._lineidx[idx]
                self._fp.seek(pos)
                result.append(read_to_character(self._fp, b'\t').decode())
        return result

    def open(self, fname, mode):
        return File.open(fname, mode)
        #if self.use_fuse:
            #return self.fuser.open(fname, mode)
        #else:
            #return exclusive_open_to_read(fname, mode)

    def ensure_lineidx_8b_opened(self):
        if not self.has_lineidx_8b:
            self.fp8b = None
            self.lineidx_8b_pid = None
            return
        if self.fp8b is None:
            self.fp8b = self.open(self.lineidx_8b, 'rb')
            self.lineidx_8b_pid = os.getpid()
        if self.lineidx_8b_pid != os.getpid():
            self.fp8b.close()
            logging.info('re-open {} because the process id changed'.format(
                self.lineidx_8b))
            self.fp8b= self.open(self.lineidx_8b, 'rb')
            self.lineidx_8b_pid = os.getpid()

    def get_offset(self, idx):
        # do not use op.isfile() to check whether lineidx_8b exists as it may
        # incur API call for blobfuse, which will be super slow if we enumerate
        # a bunch of data
        if self.has_lineidx_8b:
            self.ensure_lineidx_8b_opened()
            self.fp8b.seek(idx * 8)
            ret = int.from_bytes(self.fp8b.read(8), 'little')
            return ret
        else:
            self._ensure_lineidx_loaded()
            pos = self._lineidx[idx]
            return pos

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            with File.open(self.lineidx, 'r') as fp:
                self._lineidx = tuple([int(i.strip()) for i in fp.readlines()])
            logging.info('loaded {} from {}'.format(
                len(self._lineidx),
                self.lineidx
            ))

    def get_tsv_fp(self):
        start = time.time()
        fp = File.open(self.tsv_file, 'rb')
        #if self.use_fuse:
            #fp = self.fuser.open(self.tsv_file, 'rb')
        #else:
            #if not self.open_once:
                #fp = exclusive_open_to_read(self.tsv_file, 'rb')
                #self.open_once = True
            #else:
                #fp = open(self.tsv_file, 'rb')
        if self.use_mmap:
            mfp = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            mfp = fp
        end = time.time()
        if (end - start) > 10:
            logging.info('too long ({}) to open {}'.format(
                end - start,
                self.tsv_file))
        return mfp, fp

    def _ensure_tsv_opened(self):
        if self.cache_policy == 'memory':
            assert self._fp is not None
            return

        if self._fp is None:
            self._mfp, self._fp = self.get_tsv_fp()
            self.pid = os.getpid()

        if self.pid != os.getpid():
            self._mfp.close()
            self._fp.close()
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            from .common import print_opened_files
            print_opened_files()
            self._mfp, self._fp = self.get_tsv_fp()
            self.pid = os.getpid()

def get_all_associate_files(all_info):
    result = []
    for info in all_info:
        result.extend(get_associate_files(info))
    return result

def get_associate_files(info):
    data, split, t, version = info
    dataset = TSVDataset(data)
    fname = dataset.get_data(split, t, version)
    result = []
    if op.isfile(fname):
        result.append(fname)
    else:
        result.extend(load_list_file(dataset.get_data(split + 'X', t, version)))
        result.append(dataset.get_shuffle_file(split))
    extra = [get_tsv_lineidx_8b(r) for r in result]
    result.extend(extra)
    return result


class TSVDataset(object):
    def __init__(self, name, data_root=None):
        self.name = name
        if data_root is None:
            if os.environ.get('QD_DATA_ROOT') is not None:
                data_root = os.environ['QD_DATA_ROOT']
            else:
                proj_root = op.dirname(op.dirname(op.dirname(op.realpath(__file__))))
                data_root = op.join(proj_root, 'data')
        data_root = op.join(data_root, name)
        #print(f'data_root {data_root}')
        self._data_root = op.relpath(data_root)
        self._fname_to_tsv = {}

        self._split_to_key_to_idx = {}

    def __repr__(self):
        return 'TSVDataset({})'.format(self.name)

    def __str__(self):
        return 'TSVDataset({})'.format(self.name)

    def seek_by_key(self, key, split, t=None, version=None):
        idx = self.get_idx_by_key(key, split)
        return next(self.iter_data(split, t, version, filter_idx=[idx]))

    def seek_by_idx(self, idx, split, t=None, version=None):
        return next(self.iter_data(split, t, version, filter_idx=[idx]))

    def load_labelmap(self):
        return load_list_file(self.get_labelmap_file())

    def load_pos_labelmap(self):
        return load_list_file(self.get_pos_labelmap_file())

    def get_tree_file(self):
        return op.join(self._data_root, 'tree.txt')

    def get_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.txt')

    def load_txt(self, t='labelmap'):
        return load_list_file(self.get_txt(t))

    # labelmap or attribute map
    def get_txt(self, t='labelmap'):
        return op.join(self._data_root, '{}.txt'.format(t))

    def get_pos_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.pos.txt')

    def get_train_shuffle_file(self):
        return self.get_shuffle_file('train')

    def get_shuffle_file(self, split_name):
        return op.join(self._data_root, '{}.shuffle.txt'.format(split_name))

    def get_labelmap_of_noffset_file(self):
        return op.join(self._data_root, 'noffsets.label.txt')

    def get_idx_by_key(self, key, split):
        if split in self._split_to_key_to_idx:
            key_to_idx = self._split_to_key_to_idx[split]
        else:
            key_to_idx = {k: i for i, k in enumerate(self.load_keys(split))}
            self._split_to_key_to_idx[split] = key_to_idx
        idx = key_to_idx[key]
        return idx

    def load_key_to_idx(self, split):
        result = {}
        for i, row in enumerate(self.iter_data(split, 'label')):
            key = row[0]
            assert key not in result
            result[key] = i
        return result

    def load_keys(self, split, t='label'):
        assert self.has(split, t)
        result = []
        for row in tqdm(self.iter_data(split, t), mininterval=2):
            result.append(row[0])
        return result

    def dynamic_update(self, dataset_ops):
        '''
        sometimes, we update the dataset, and here, we should update the file
        path
        '''
        if len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'sample':
            self._data_root = op.join('./output/data/',
                    '{}_{}_{}'.format(self.name,
                        dataset_ops[0]['sample_label'],
                        dataset_ops[0]['sample_image']))
        elif len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'mask_background':
            target_folder = op.join('./output/data',
                    '{}_{}_{}'.format(self.name,
                        '.'.join(map(str, dataset_ops[0]['old_label_idx'])),
                        dataset_ops[0]['new_label_idx']))
            self._data_root = target_folder

    def get_test_tsv_file(self, t=None):
        return self.get_data('test', t)

    def get_test_tsv_lineidx_file(self):
        return op.join(self._data_root, 'test.lineidx')

    def get_train_tsvs(self, t=None):
        if op.isfile(self.get_data('train', t)):
            return [self.get_data('train', t)]
        trainx_file = op.join(self._data_root, 'trainX.tsv')
        if not op.isfile(trainx_file):
            return []
        train_x = load_list_file(trainx_file)
        if t is None:
            return train_x
        elif t =='label':
            if op.isfile(self.get_data('trainX', 'label')):
                return load_list_file(self.get_data('trainX', 'label'))
            else:
                files = [op.splitext(f)[0] + '.label.tsv' for f in train_x]
                return files

    def get_train_tsv(self, t=None):
        return self.get_data('train', t)

    def get_lineidx(self, split_name):
        return op.join(self._data_root, '{}.lineidx'.format(split_name))

    def get_latest_version(self, split, t=None):
        assert t is not None, 'if it is none, it is always 0'
        v = 0
        if t is None:
            pattern = op.join(self._data_root, '{}.v*.tsv'.format(split))
            re_pattern = '{}\.v([0-9]*)\.tsv'.format(split)
        else:
            pattern = op.join(self._data_root, '{}.{}.v*.tsv'.format(
                split, t))
            re_pattern = '{}\.{}\.v([0-9]*)\.tsv'.format(split, t)
        all_file = glob.glob(pattern)
        import re
        re_results = [re.match(re_pattern, op.basename(f)) for f in all_file]
        candidates = ([int(re_result.groups()[0]) for re_result, f in
            zip(re_results, all_file) if re_result])
        if len(candidates) > 0:
            v = max(candidates)
        assert v >= 0
        return v

    def get_gen_info_data(self, split, t=None, version=None):
        return self.get_data(split, '{}.generate.info'.format(t), version=version)

    def get_file(self, fname):
        return op.join(self._data_root, fname)

    def get_data(self, split_name, t=None, version=None):
        '''
        e.g. split_name = train, t = label
        if version = None or 0,  return train.label.tsv
        we don't have train.label.v0.tsv
        if version = 3 > 0, return train.label.v3.tsv
        if version = -1, return the highest version
        '''
        if t is None:
            # in this case, it is an image split, which has no version
            version = None
        if version is None or version in [0,'None','0']:
            if t is None:
                return op.join(self._data_root, '{}.tsv'.format(split_name))
            else:
                return op.join(self._data_root, '{}.{}.tsv'.format(split_name,
                    t))
        elif version == -1:
            if not op.isfile(self.get_data(split_name, t)):
                return self.get_data(split_name, t)
            v = self.get_latest_version(split_name, t)
            return self.get_data(split_name, t, v)
        else:
            return op.join(self._data_root, '{}.{}.v{}.tsv'.format(split_name,
                t, version))

    def get_num_train_image(self):
        if op.isfile(self.get_data('trainX')):
            if op.isfile(self.get_shuffle_file('train')):
                return len(load_list_file(self.get_shuffle_file('train')))
            else:
                return 0
        else:
            return len(load_list_file(op.join(self._data_root, 'train.lineidx')))

    def get_trainval_tsv(self, t=None):
        return self.get_data('trainval', t)

    def get_noffsets_file(self):
        return op.join(self._data_root, 'noffsets.txt')

    def load_noffsets(self):
        logging.info('deprecated: pls generate it on the fly')
        return load_list_file(self.get_noffsets_file())

    def load_inverted_label(self, split, version=None, label=None):
        fname = self.get_data(split, 'inverted.label', version)
        if not op.isfile(fname):
            return {}
        elif label is None:
            tsv = TSVFile(fname)
            num_rows = len(tsv)
            result = {}
            for row in tqdm(tsv, total=num_rows, mininterval=2):
                assert row[0] not in result
                assert len(row) == 2
                ss = row[1].split(' ')
                if len(ss) == 1 and ss[0] == '':
                    result[row[0]] = []
                else:
                    result[row[0]] = list(map(int, ss))
            return result
        else:
            all_label = load_list_file(self.get_data(split, 'labelmap', version))
            if label not in all_label:
                return {}
            result = {}
            idx = all_label.index(label)
            tsv = self._retrieve_tsv(fname)
            row = tsv.seek(idx)
            assert row[0] == label
            ss = row[1].split(' ')
            if len(ss) == 1 and ss[0] == '':
                result[row[0]] = []
            else:
                result[row[0]] = list(map(int, ss))
            return result

    def load_inverted_label_as_list(self, split, version=None, label=None):
        fname = self.get_data(split, 'inverted.label', version)
        if not op.isfile(fname):
            return []
        elif label is None:
            rows = tsv_reader(fname)
            result = []
            for row in rows:
                assert len(row) == 2
                ss = row[1].split(' ')
                if len(ss) == 1 and ss[0] == '':
                    result.append((row[0], []))
                else:
                    result.append((row[0], list(map(int, ss))))
            return result
        else:
            all_label = self.load_labelmap()
            result = []
            idx = all_label.index(label)
            tsv = self._retrieve_tsv(fname)
            row = tsv.seek(idx)
            assert row[0] == label
            ss = row[1].split(' ')
            if len(ss) == 1 and ss[0] == '':
                result.append((row[0], []))
            else:
                result.append((row[0], list(map(int, ss))))
            return result

    def has(self, split, t=None, version=None):
        return tsv_exists(self.get_data(split, t, version)) or (
                File.isfile(self.get_data('{}X'.format(split), t, version)) and
                tsv_exists(self.get_shuffle_file(split)))

    def last_update_time(self, split, t=None, version=None):
        tsv_file = self.get_data(split, t, version)
        if op.isfile(tsv_file):
            return os.path.getmtime(tsv_file)
        assert version is None or version == 0, 'composite dataset always v=0'
        tsv_file = self.get_data('{}X'.format(split), t, version)
        assert op.isfile(tsv_file)
        return os.path.getmtime(tsv_file)

    def iter_composite(self, split, t, version, filter_idx=None):
        splitX = split + 'X'
        file_list = load_list_file(self.get_data(splitX, t, version))
        tsvs = [self._retrieve_tsv(f) for f in file_list]
        shuffle_file = self.get_shuffle_file(split)
        if filter_idx is None:
            shuffle_tsv_rows = tsv_reader(shuffle_file)
            for idx_source, idx_row in shuffle_tsv_rows:
                idx_source, idx_row = int(idx_source), int(idx_row)
                row = tsvs[idx_source].seek(idx_row)
                if len(row) == 3:
                    row[1] == 'dont use'
                yield row
        else:
            shuffle_tsv = self._retrieve_tsv(shuffle_file)
            for i in filter_idx:
                idx_source, idx_row = shuffle_tsv.seek(i)
                idx_source, idx_row = int(idx_source), int(idx_row)
                row = tsvs[idx_source].seek(idx_row)
                if len(row) == 3:
                    row[1] == 'dont use'
                yield row

    def num_rows(self, split, t=None, version=None):
        f = self.get_data(split, t, version)
        if op.isfile(f) or op.islink(f):
            return TSVFile(f).num_rows()
        else:
            #f = self.get_data(split + 'X', t, version=version)
            #assert op.isfile(f), f
            #return len(load_list_file(self.get_shuffle_file(split)))
            return len(TSVFile(self.get_shuffle_file(split)))

    def iter_data(self, split, t=None, version=None,
            unique=False, filter_idx=None, progress=False):
        if progress:
            if filter_idx is None:
                num_rows = self.num_rows(split)
            else:
                num_rows = len(filter_idx)
            pbar = progressbar.ProgressBar(maxval=num_rows).start()
        splitX = split + 'X'
        if not op.isfile(self.get_data(split, t, version)) and \
                op.isfile(self.get_data(splitX, t, version)):
            if t is not None:
                if unique:
                    returned = set()
                for i, row in enumerate(self.iter_composite(split, t, version,
                        filter_idx=filter_idx)):
                    if unique and row[0] in returned:
                        continue
                    else:
                        yield row
                        if unique:
                            returned.add(row[0])
                    if progress:
                        pbar.update(i)
            else:
                rows_data = self.iter_composite(split, None, version=version,
                        filter_idx=filter_idx)
                #logging.info('breaking change: label is ignore for t=None')
                #rows_label = self.iter_data(split, 'label', version=version,
                        #filter_idx=filter_idx)
                if unique:
                    returned = set()
                for i, r in enumerate(rows_data):
                    if unique and r[0] in returned:
                        continue
                    else:
                        yield r
                        if unique:
                            returned.add(r[0])
                    if progress:
                        pbar.update(i)
        else:
            fname = self.get_data(split, t, version)
            if not op.isfile(fname):
                logging.info('no {}'.format(fname))
                return
            if filter_idx is None:
                for i, row in enumerate(tsv_reader(self.get_data(
                    split, t, version))):
                    yield row
                    if progress:
                        pbar.update(i)
            else:
                fname = self.get_data(split, t, version)
                tsv = self._retrieve_tsv(fname)
                if progress:
                    for i in tqdm(filter_idx):
                        yield tsv.seek(i)
                else:
                    for i in filter_idx:
                        yield tsv.seek(i)


    def _retrieve_tsv(self, fname):
        if fname in self._fname_to_tsv:
            tsv = self._fname_to_tsv[fname]
        else:
            tsv = TSVFile(fname)
            self._fname_to_tsv[fname] = tsv
        return tsv

    def safe_write_data(self, rows, split, t=None, version=None,
                        generate_info=None, force=False):
        assert force or not self.has(split, t, version)
        if generate_info is None:
            from .common import get_frame_info
            info = get_frame_info(last=1)
            def gen_info():
                for k, v in info.items():
                    if isinstance(v, str):
                        yield k, v
            generate_info = gen_info()
        self.write_data(rows, split, t, version,
                        generate_info=generate_info)

    def write_data(self, rows, split, t=None, version=None, generate_info=None):
        out_tsv = self.get_data(split, t, version)
        tsv_writer(rows, out_tsv)
        if generate_info is not None:
            out_tsv = self.get_data(split, '{}.generate.info'.format(t), version=version)
            tsv_writer(generate_info, out_tsv)

    def update_data(self, rows, split, t, generate_info=None):
        '''
        if the data are the same, we will not do anything.
        '''
        assert t is not None
        v = self.get_latest_version(split, t)
        if self.has(split, t, v):
            is_equal = True
            # we first save it to a tmp tsv file
            self.write_data(rows, split, t + '.tmp', v + 1)
            for origin_row, new_row in zip(self.iter_data(split, t, v),
                    self.iter_data(split, t + '.tmp', v + 1)):
                if len(origin_row) != len(new_row):
                    is_equal = False
                    break
                for o, n in zip(origin_row, new_row):
                    if o != n:
                        is_equal = False
                        break
                if not is_equal:
                    break
            if not is_equal:
                logging.info('creating {} for {}'.format(v + 1, self.name))
                if generate_info:
                    self.write_data(generate_info, split, '{}.generate.info'.format(t), v + 1)
                tsv_mv(self.get_data(split, t + '.tmp', v + 1),
                        self.get_data(split, t, v + 1))
                return v + 1
            else:
                logging.info('ignore to create since the label matches the latest')
        else:
            assert v == 0
            v = -1
            logging.info('creating {} for {}'.format(v + 1, self.name))
            if generate_info:
                self.write_data(generate_info, split, '{}.generate.info'.format(t), v + 1)
            self.write_data(rows, split, t, version=v + 1)
            return v + 1

    def load_composite_source_data_split(self, split):
        splitX = split + 'X'
        pattern = 'data/(.*)/(.*)\.tsv'
        tsv_sources = [l for l, in tsv_reader(self.get_data(splitX))]
        matched_result = [re.match(pattern, l).groups()
                for l in tsv_sources]

        return [(d, s) for d, s in matched_result]

    def load_composite_source_data_split_versions(self, split):
        # this function is only valid if we generated the composite dataset
        # from tsv, not from db. if it is from db, there is no file of
        # origin.label. use load_composite_source_data_split, instead.
        splitX = split + 'X'
        pattern = 'data/(.*)/(train|trainval|test)\.label\.v(.*)\.tsv'
        tsv_sources = [l for l, in tsv_reader(self.get_data(splitX,
            'origin.label'))]
        matched_result = [re.match(pattern, l).groups()
                for l in tsv_sources]

        return [(d, s, int(v)) for d, s, v in matched_result]

def csv_writer(values, file_name):
    tsv_writer(values, file_name, sep=',')
    return

class TSVSplitProperty(object):
    '''
    one instance of this class mean one tsv file or one composite tsv, it could
    be label tsv, or hw tsv, or image tsv
    currently, it depends on TSVDataset, and we should remove such dependency
    in future and deprecate TSVDataset
    '''
    def __init__(self, data, split, t=None, version=0, cache_policy=None,
                 hold_buffer=0, data_dir=None):
        self.data = data
        self.split = split
        self.t = t
        self.version = version
        dataset = TSVDataset(data, data_root=data_dir)

        split_tsv = dataset.get_data(split + 'S', t, version)
        single_tsv = dataset.get_data(split, t, version)
        
        if int(os.environ.get('QD_ENABLE_EQUAL_SPLIT', '0')) and File.isfile(split_tsv):
            self.tsv = EqualSplitTSVFile(split_tsv)
        elif File.isfile(single_tsv):
            self.tsv = TSVFile(dataset.get_data(split, t, version),
                    cache_policy)
        else:
            splitX = split + 'X'
            list_file = dataset.get_data(splitX, t, version=version)
            seq_file = dataset.get_shuffle_file(split)
            #print(list_file)
            #print(seq_file)
            assert File.isfile(list_file) and File.isfile(seq_file), (
                '{}, {}/{} not available'.format(single_tsv, list_file, seq_file)
            )
            self.tsv = CompositeTSVFile(list_file, seq_file, cache_policy,
                                        hold_buffer=hold_buffer, data_dir=data_dir)

    def is_from_valid_file(self, idx=None):
        return self.tsv.is_from_valid_file(idx)

    def get_row_len(self, i):
        return self.tsv.get_row_len(i)

    def __repr__(self):
        return 'TSVSplitProperty(tsv={})'.format(
            self.tsv
        )

    def __getitem__(self, index):
        row = self.tsv[index]
        return row

    def __len__(self):
        return len(self.tsv)

    def num_rows(self):
        return len(self)

    def close(self):
        self.tsv.close()

    def __iter__(self):
        return iter(self.tsv)

    def get_key(self, i):
        return self.tsv.seek_first_column(i)

    def seek_first_column(self, idx):
        return self.tsv.seek_first_column(idx)

    def get_composite_source_idx(self):
        return self.tsv.get_composite_source_idx()

def tsv_writers(all_values, tsv_file_names, sep='\t'):
    # values: a list of [row1, row2]. each row goes to each tsv_file_name
    for tsv_file_name in tsv_file_names:
        ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_files = [os.path.splitext(tsv_file_name)[0] + '.lineidx'
        for tsv_file_name in tsv_file_names]
    tsv_lineidx_8b_files = [x + '.8b' for x in tsv_lineidx_files]
    sep = sep.encode()
    assert all_values is not None
    from contextlib import ExitStack
    with ExitStack() as stack:
        fps = [stack.enter_context(File.open(f, 'wb')) for f in
                tsv_file_names]
        fpidxs = [stack.enter_context(File.open(f, 'w')) for f in
                tsv_lineidx_files]
        fpidx8bs = [stack.enter_context(File.open(f, 'wb')) for f in tsv_lineidx_8b_files]
        idxs = [0 for _ in fps]
        for values in all_values:
            assert values is not None
            for i, (value, fp, fpidx, fpidx8b) in enumerate(zip(values, fps, fpidxs, fpidx8bs)):
                value = map(lambda v: v if type(v) == bytes else str(v).encode(),
                        value)
                v = sep.join(value) + b'\n'
                fp.write(v)
                fpidx.write(str(idxs[i]) + '\n')
                fpidx8b.write(idxs[i].to_bytes(8, 'little'))
                idxs[i] = idxs[i]+ len(v)


def tsv_writers_backup(all_values, tsv_file_names, sep='\t'):
    # values: a list of [row1, row2]. each row goes to each tsv_file_name
    for tsv_file_name in tsv_file_names:
        ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_files = [os.path.splitext(tsv_file_name)[0] + '.lineidx'
        for tsv_file_name in tsv_file_names]
    tsv_lineidx_8b_files = [x + '.8b' for x in tsv_lineidx_files]
    tsv_lineidx_8b_file_tmps = [x + '.tmp' for x in tsv_lineidx_8b_files]
    tsv_file_name_tmps = [tsv_file_name + '.tmp' for tsv_file_name in
            tsv_file_names]
    tsv_lineidx_file_tmps = [tsv_lineidx_file + '.tmp' for tsv_lineidx_file in
            tsv_lineidx_files]
    sep = sep.encode()
    assert all_values is not None
    fps = [open(tsv_file_name_tmp, 'wb') for tsv_file_name_tmp in
            tsv_file_name_tmps]
    fpidxs = [open(tsv_lineidx_file_tmp, 'w') for tsv_lineidx_file_tmp in
            tsv_lineidx_file_tmps]
    fpidx8bs = [open(x, 'wb') for x in tsv_lineidx_8b_file_tmps]
    idxs = [0 for _ in fps]
    for values in all_values:
        assert values is not None
        for i, (value, fp, fpidx, fpidx8b) in enumerate(zip(values, fps, fpidxs, fpidx8bs)):
            value = map(lambda v: v if type(v) == bytes else str(v).encode(),
                    value)
            v = sep.join(value) + b'\n'
            fp.write(v)
            fpidx.write(str(idxs[i]) + '\n')
            fpidx8b.write(idxs[i].to_bytes(8, 'little'))
            idxs[i] = idxs[i]+ len(v)
    for f in fps:
        f.close()
    for f in fpidxs:
        f.close()
    for f in fpidx8bs:
        f.close()
    # the following might crash if there are two processes which are writing at
    # the same time. One process finishes the renaming first and the second one
    # will crash. In this case, we know there must be some errors when you run
    # the code, and it should be a bug to fix rather than to use try-catch to
    # protect it here.
    for tsv_file_name_tmp, tsv_file_name in zip(tsv_file_name_tmps,
            tsv_file_names):
        os.rename(tsv_file_name_tmp, tsv_file_name)
    for tsv_lineidx_file_tmp, tsv_lineidx_file in zip(tsv_lineidx_file_tmps,
            tsv_lineidx_files):
        os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)
    for x, y in zip(tsv_lineidx_8b_file_tmps, tsv_lineidx_8b_files,):
        os.rename(x, y)

#def tsv_writer(values, tsv_file_name, sep='\t'):
    #ensure_directory(os.path.dirname(tsv_file_name))
    #tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    #tsv_8b_file = tsv_lineidx_file + '.8b'
    #idx = 0
    #tsv_file_name_tmp = tsv_file_name + '.tmp'
    #tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    #tsv_8b_file_tmp = tsv_8b_file + '.tmp'
    #import sys
    #is_py2 = sys.version_info.major == 2
    #if not is_py2:
        #sep = sep.encode()
    #with open(tsv_file_name_tmp, 'wb') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx, open(tsv_8b_file_tmp, 'wb') as fp8b:
        #assert values is not None
        #for value in values:
            #assert value is not None
            #if is_py2:
                #v = sep.join(map(lambda v: str(v) if not isinstance(v, six.string_types) else v, value)) + '\n'
                #if type(v) is unicode:
                    #v = v.encode('utf-8')
            #else:
                #value = map(lambda v: v if type(v) == bytes else str(v).encode(),
                        #value)
                #v = sep.join(value) + b'\n'
            #fp.write(v)
            #fpidx.write(str(idx) + '\n')
            ## although we can use sys.byteorder to retrieve the system-default
            ## byte order, let's use little always to make it consistent and
            ## simple
            #fp8b.write(idx.to_bytes(8, 'little'))
            #idx = idx + len(v)
    ## the following might crash if there are two processes which are writing at
    ## the same time. One process finishes the renaming first and the second one
    ## will crash. In this case, we know there must be some errors when you run
    ## the code, and it should be a bug to fix rather than to use try-catch to
    ## protect it here.
    #os.rename(tsv_file_name_tmp, tsv_file_name)
    #os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)

    #os.rename(tsv_8b_file_tmp, tsv_8b_file)

def load_from_yaml_file(file_name):
    # do not use File.open as File.open depends on this function
    with File.open(file_name, 'r') as fp:
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

def write_to_file(contxt, file_name, append=False):
    if type(contxt) is str:
        contxt = contxt.encode()
    flag = 'wb'
    if append:
        flag = 'ab'
    with File.open(file_name, flag) as fp:
        fp.write(contxt)

def load_list_file(fname):
    with File.open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

def tsv_writer(values, tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    tsv_8b_file = tsv_lineidx_file + '.8b'
    idx = 0
    sep = sep.encode()
    with File.open(tsv_file_name, 'wb') as fp, File.open(tsv_lineidx_file, 'w') as fpidx, File.open(tsv_8b_file, 'wb') as fp8b:
        assert values is not None
        for value in tqdm(values):
            assert value is not None
            value = map(lambda v: v if type(v) == bytes else str(v).encode(),
                    value)
            v = sep.join(value) + b'\n'
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            # although we can use sys.byteorder to retrieve the system-default
            # byte order, let's use little always to make it consistent and
            # simple
            fp8b.write(idx.to_bytes(8, 'little'))
            idx = idx + len(v)

def tsv_reader(tsv_file_name, sep='\t'):
    #with open(tsv_file_name, 'r') as fp:
    with File.open(tsv_file_name, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def csv_reader(tsv_file_name):
    return tsv_reader(tsv_file_name, ',')

def get_meta_file(tsv_file):
    return op.splitext(tsv_file)[0] + '.meta.yaml'

def extract_label(full_tsv, label_tsv):
    if op.isfile(label_tsv):
        logging.info('label file exists and will skip to generate: {}'.format(
            label_tsv))
        return
    if not op.isfile(full_tsv):
        logging.info('the file of {} does not exist'.format(full_tsv))
        return
    rows = tsv_reader(full_tsv)
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info('extract_label: {}-{}'.format(full_tsv, i))
            del row[2]
            assert len(row) == 2
            assert type(row[0]) == str
            assert type(row[1]) == str
            yield row
    tsv_writer(gen_rows(), label_tsv)

def create_inverted_tsv(rows, inverted_label_file, label_map):
    '''
    deprecated, use create_inverted_list
    save the results based on the label_map in label_map_file. The benefit is
    to seek the row given a label
    '''
    inverted = {}
    for i, row in enumerate(rows):
        labels = json.loads(row[1])
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
        else:
            assert type(labels) is int
            curr_unique_labels = [label_map[labels]]
        for l in curr_unique_labels:
            assert type(l) == str or type(l) == unicode
            if l not in inverted:
                inverted[l] = [i]
            else:
                inverted[l].append(i)
    def gen_rows():
        for label in inverted:
            assert label in label_map
        for label in label_map:
            i = inverted[label] if label in inverted else []
            yield label, ' '.join(map(str, i))
    tsv_writer(gen_rows(), inverted_label_file)

def create_inverted_list2(rows, th=None):
    inverted = {}
    keys = []
    for i, row in enumerate(rows):
        keys.append(row[0])
        labels = json.loads(row[1])
        if th is not None:
            labels = [r for r in labels if 'conf' in r and r['conf'] > th or
                            'conf' not in r]
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
        else:
            assert type(labels) is int
            curr_unique_labels = [str(labels)]
        for l in curr_unique_labels:
            assert type(l) == str or type(l) == unicode
            if l not in inverted:
                inverted[l] = [i]
            else:
                inverted[l].append(i)
    return inverted, keys

def is_verified_rect(rect):
    #allowed_keys = set(['class', 'rect', 'uhrs_confirm', 'uhrs_uncertain',
            #'conf', 'merge_from', 'class_from', 'change_from', 'from', 'diff',
            #'IsInside', 'IsGroupOf', 'IsDepiction', 'IsOccluded',
            #'IsTruncated', 'workerId', 'class_propagate_from', 'obj', 'uhrs'])
    #unknown_keys = [k for k in rect if k not in allowed_keys]
    #if len(unknown_keys) > 0:
        #logging.info('unknown keys = {}\n'.format(pformat(unknown_keys)))
        #pass

    if 'uhrs' in rect:
        judge_result = rect['uhrs']
        assert judge_result.get('1', 0) >= judge_result.get('2', 0)
        return True

    if 'class' not in rect or 'rect' not in rect:
        return False

    if 'uhrs_confirm' in rect:
        assert rect['uhrs_confirm'] > 0
        return True

    if 'conf' in rect and rect['conf'] < 1:
        return False

    if 'merge_from' in rect:
        return all(is_verified_rect(r) for r in rect['merge_from'])

    return True

def create_inverted_list(rows):
    inverted = {}
    inverted_with_bb = {}
    inverted_no_bb = {}
    inverted_with_bb_verified = {}
    inverted_with_bb_noverified = {}
    logging.info('creating inverted')
    for i, row in tqdm(enumerate(rows), mininterval=2):
        labels = json.loads(row[1])
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
            curr_unique_with_bb_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect'])])
            curr_unique_no_bb_labels = set([l['class'] for l in labels
                if 'rect' not in l or all(x == 0 for x in l['rect'])])
            curr_unique_with_bb_verified_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect']) and is_verified_rect(l)])
            curr_unique_with_bb_noverified_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect']) and not is_verified_rect(l)])
        else:
            assert type(labels) is int
            curr_unique_labels = [str(labels)]
            curr_unique_with_bb_labels = []
            curr_unique_no_bb_labels = curr_unique_labels
            curr_unique_with_bb_verified_labels = set()
            curr_unique_with_bb_noverified_labels = set()
        def update(unique_labels, inv):
            for l in unique_labels:
                assert type(l) == str
                if l not in inv:
                    inv[l] = [i]
                else:
                    inv[l].append(i)
        update(curr_unique_labels, inverted)
        update(curr_unique_with_bb_labels, inverted_with_bb)
        update(curr_unique_no_bb_labels, inverted_no_bb)
        update(curr_unique_with_bb_verified_labels, inverted_with_bb_verified)
        update(curr_unique_with_bb_noverified_labels, inverted_with_bb_noverified)
    return {'inverted.label': inverted,
            'inverted.label.with_bb': inverted_with_bb,
            'inverted.label.no_bb': inverted_no_bb,
            'inverted.label.with_bb.verified': inverted_with_bb_verified,
            'inverted.label.with_bb.noverified': inverted_with_bb_noverified}

def tsv_shuffle_reader(tsv_file):
    logging.warn('deprecated: using TSVFile to randomly seek')
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    lineidx = load_list_file(lineidx_file)
    random.shuffle(lineidx)
    with open(tsv_file, 'r') as fp:
        for l in lineidx:
            fp.seek(int(float(l)))
            yield [x.strip() for x in fp.readline().split('\t')]

def load_labelmap(data):
    dataset = TSVDataset(data)
    return dataset.load_labelmap()

def get_caption_data_info(name):
    dataset = TSVDataset(name)
    splits = get_default_splits()
    from collections import defaultdict
    split_to_versions = defaultdict(list)
    for split in splits:
        v = 0
        while True:
            if not dataset.has(split, 'caption', v):
                break
            split_to_versions[split].append(v)
            v = v + 1
    return split_to_versions

def get_all_data_info2(name=None):
    if name is None:
        return sorted(os.listdir('./data'))
    else:
        dataset = TSVDataset(name)
        valid_split_versions = []
        splits = get_default_splits()

        for split in splits:
            v = 0
            while True:
                if not dataset.has(split, 'label', v):
                    break
                if dataset.has(split, 'inverted.label.count', v):
                    label_count_rows = dataset.iter_data(split, 'inverted.label.count', v)
                    label_count = [(r[0], int(r[1])) for r in label_count_rows]
                    label_count = sorted(label_count, key=lambda x: x[1])
                else:
                    label_count = []
                valid_split_versions.append((split, v, "", [(i, l, c) for i, (l, c) in
                    enumerate(label_count)]))
                v = v + 1
        name_splits_labels = [(name, valid_split_versions)]
        return name_splits_labels

def get_all_data_info():
    names = os.listdir('./data')
    name_splits_labels = []
    names.sort(key=lambda n: n.lower())
    for name in names:
        dataset = TSVDataset(name)
        if not op.isfile(dataset.get_labelmap_file()):
            continue
        labels = dataset.load_labelmap()
        valid_splits = []
        if len(dataset.get_train_tsvs()) > 0:
            valid_splits.append('train')
        for split in ['trainval', 'test']:
            if not op.isfile(dataset.get_data(split)):
                continue
            valid_splits.append(split)
        name_splits_labels.append((name, valid_splits, labels))
    return name_splits_labels

def load_labels(file_name):
    rows = tsv_reader(file_name)
    key_to_rects = {}
    key_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        rects = json.loads(row[1])
        #assert key not in labels, '{}-{}'.format(file_name, key)
        key_to_rects[key] = rects
        key_to_idx[key] = i
    return key_to_rects, key_to_idx

def azcopy_read(fname):
    # we ignore fname since it could be a mounted blobfuse folder in AML
    local_file_name = op.join('/tmp', '{}_{}'.format(get_user_name(),
        hash_sha1(op.realpath(op.abspath(fname))) +
        op.splitext(fname)[1]))
    if op.isfile(local_file_name):
        return open(local_file_name, 'r')
    config_file = os.environ['FILE_OPEN_AZCOPY_BLOB_ACCOUNT_PATH']
    from qd.cloud_storage import create_cloud_storage
    remote_path = op.join(os.environ['FILE_OPEN_AZCOPY_REMOTE_PATH'],
            op.relpath(fname, os.environ['FILE_OPEN_AZCOPY_LOCAL_PATH']))
    c = create_cloud_storage(config_file=config_file)
    logging.info('downloading from {} to {} for {}'.format(remote_path,
        local_file_name, fname))
    c.az_download(remote_path, local_file_name)
    return open(local_file_name, 'r')

def parallel_generate_lineidx8b_from_lineidx(x):
    lineidx, lineidx_8b = x
    generate_lineidx8b_from_lineidx(lineidx, lineidx_8b)

def generate_lineidx8b_from_lineidx(lineidx, lineidx_8b):
    logging.info(lineidx)
    with File.open(lineidx_8b, 'wb') as fp8b:
        for i, in tqdm(tsv_reader(lineidx)):
            fp8b.write(int(i).to_bytes(8, 'little'))

def convert_data_to_yaml(
    data, split, yaml,
    is_train=True,
    label=None,
    feature=None,
    qd_format=False,
    label_version=None,
    feature_version=None):
    # used for captioning-related scripts
    if qd_format:
        info = {
            'feature': feature if feature is not None else {
                'data': data,
                'split': split,
                't': 'feature',
                'version': feature_version,
            },
            'hw': {'data': data, 'split': split, 't': 'hw'},
            'img': {'data': data, 'split': split},
            'label': label if label is not None else {
                'data': data,
                'split': split,
                't': 'label',
                'version': label_version,
            },
            'caption': {'data': data, 'split': split, 't': 'hw'},
            'composite': False,
            'qd_format': True,
        }
    else:
        assert label is None and feature is None
        # will be deprecated
        from qd.tsv_io import TSVDataset
        yaml_folder = op.dirname(yaml)
        dataset = TSVDataset(data)
        if not op.isfile(dataset.get_data(split + 'X')):
            # we prefer to use the composite
            info = {
                'feature': op.relpath(dataset.get_data('train', 'feature', version=feature_version), yaml_folder),
                'label': op.relpath(dataset.get_data(split, 'label', version=label_version), yaml_folder),
                'hw': op.relpath(dataset.get_data(split, 'hw'), yaml_folder),
                'img': op.relpath(dataset.get_data(split), yaml_folder),
                'caption': op.relpath(dataset.get_data(split, 'caption'), yaml_folder),
                'composite': False,
            }
        else:
            def get_rel_path(p):
                return op.relpath(op.realpath(p), op.realpath(yaml_folder))
            splitX = split + 'X'
            from .common import load_list_file
            info = {
                'feature': list(map(get_rel_path, load_list_file(dataset.get_data(splitX, 'feature', version=feature_version)))),
                'label': list(map(get_rel_path, load_list_file(dataset.get_data( splitX, 'label', version=label_version)))),
                'hw': list(map(get_rel_path, load_list_file(dataset.get_data(splitX, 'hw')))),
                'img': list(map(get_rel_path, load_list_file(dataset.get_data(splitX)))),
                'caption': list(map(get_rel_path, load_list_file(dataset.get_data(splitX, 'caption')))),
                'composite': True,
            }
            if is_train:
                caption_linelist = dataset.get_data(split, 'caption_linelist')
                assert op.isfile(caption_linelist)
                info['caption_linelist'] = caption_linelist
            else:
                caption_linelist = dataset.get_data(split, 'caption_linelist_test')
                if not op.isfile(caption_linelist):
                    from qd.tsv_io import tsv_reader
                    tsv_writer(((a, b, 0) for a, b in
                                tsv_reader(dataset.get_shuffle_file(split))),
                               caption_linelist)
                info['caption_linelist'] = caption_linelist
    from .common import write_to_yaml_file
    write_to_yaml_file(info, yaml)

def save_to_yaml_file(context, file_name):
    with File.open(file_name, 'w') as fp:
        import yaml
        yaml.dump(context, fp, default_flow_style=False,
                encoding='utf-8', allow_unicode=True)

def concat_files(ins, out):
    File.prepare(ins)
    with File.open(out, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with File.open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)

def is_folder_content_same(folder1, folder2):
    files1 = [op.join(r, f) for r, _, fs in os.walk(folder1) for f in fs]
    files2 = [op.join(r, f) for r, _, fs in os.walk(folder2) for f in fs]
    if len(files1) != len(files2):
        return False
    sub_names1 = [f[len(folder1):] for f in files1]
    sub_names2 = [f[len(folder2):] for f in files2]
    sub_names1 = sorted(sub_names1)
    sub_names2 = sorted(sub_names2)
    from .common import float_tolorance_equal
    if not float_tolorance_equal(sub_names1, sub_names2):
        return False
    for f1 in files1:
        f2 = folder2 + f1[len(folder1):]
        if not is_file_content_same(f1, f2):
            return False
    return True


def is_file_content_same(f1, f2):
    s1 = File.get_file_size(f1)
    s2 = File.get_file_size(f2)
    if s1 != s2:
        return False
    with File.open(f1, 'rb') as fp1, File.open(f2, 'rb') as fp2:
        while True:
            s1 = fp1.read(1024*1024 * 10)
            s2 = fp2.read(1024*1024 * 10)
            if s1 == b'' and s2 == b'':
                break
            if s1 != s2:
                return False
    return True

def read_to_buffer(file_name):
    with File.open(file_name, 'rb') as fp:
        return fp.read()

'''class File(object):
    initialized = False
    use_fuser = False
    fuser = None

    @classmethod
    def ensure_initialized(cls):
        if not cls.initialized:
            cls.initialized = True
            cls.use_fuser = int(os.environ.get('QD_TSV_USE_FUSE', '0'))
            if cls.use_fuser:
                from qd.cloud_storage import create_cloud_fuse
                cls.fuser = create_cloud_fuse()
                gc  = int(os.environ.get('QD_USE_FUSE_ENABLE_GARBAGE_COLLECTION', '0'))
                from .common import get_mpi_local_rank
                if gc and get_mpi_local_rank() == 0:
                    cls.fuser.ensure_invoke_garbage_collect()
                fs = os.environ.get('QD_USE_FUSE_DOWNLOAD_FILE_LIST_AT_INIT')
                if fs:
                    cache = []
                    for f in fs.split(','):
                        with cls.fuser.open(f) as fp:
                            cache.extend([l.strip() for l in fp])
                    cls.fuser.ensure_cache(cache)

    @classmethod
    def isfile(cls, path):
        cls.ensure_initialized()
        if cls.use_fuser:
            return cls.fuser.isfile(path)
        else:
            return op.isfile(path)

    @classmethod
    def open(cls, fname, mode='r'):
        cls.ensure_initialized()
        if mode in ['r', 'rb']:
            if cls.use_fuser:
                return cls.fuser.open(fname, mode)
            else:
                return exclusive_open_to_read(fname, mode)
        elif mode in ['w', 'wb']:
            if cls.use_fuser:
                return cls.fuser.open(fname, mode)
            else:
                return robust_open_to_write(fname, mode)

    @classmethod
    def clear_cache(cls, folder):
        cls.ensure_initialized()
        if not cls.use_fuser:
            return
        return cls.fuser.clear_cache(folder)

    @classmethod
    def rm(cls, fname):
        cls.ensure_initialized()
        if cls.use_fuser:
            cls.fuser.rm(fname)
        else:
            os.remove(fname)

    @classmethod
    def async_upload(cls, enabled=False, shm_as_tmp=False):
        cls.ensure_initialized()
        if not cls.use_fuser:
            return contextlib.nullcontext()
        return cls.fuser.async_upload(enabled, shm_as_tmp)

    @classmethod
    @property
    def async_upload_queue(cls):
        cls.ensure_initialized()
        if not cls.use_fuser:
            return
        return cls.fuser.async_upload_queue

    @classmethod
    def get_file_size(cls, fname):
        cls.ensure_initialized()
        if cls.use_fuser:
            return cls.fuser.get_file_size(fname)
        else:
            return get_file_size(fname)

    @classmethod
    def list(cls, folder, recursive=False):
        cls.ensure_initialized()
        if cls.use_fuser:
            return cls.fuser.list(folder, recursive=recursive)
        else:
            return glob.glob(op.join(folder, '*'), recursive=recursive)

    @classmethod
    def prepare(cls, file_or_fnames, allow_aux_storage=False):
        cls.ensure_initialized()
        if cls.use_fuser:
            cls.fuser.ensure_cache(file_or_fnames,
                                   allow_aux_storage=allow_aux_storage)

QDFile = File'''

'''a=TSVSplitProperty('TaxCCSBUCocoVGCap384Q90SF', 'train', 'caption')
print(a.__getitem__(0))
print(a.__getitem__(0)[1])
['00000000', '[{"caption": "a very typical bus station", "tokens": ["a", "very", "typical", "bus", "station"], "deptree": [[4, 0], [2, 1], [4, 2], [4, 3], [-1, 4]]}]']

a=TSVSplitProperty('TaxCCSBUCocoVGCap384Q90SF', 'train')
print(a.__getitem__(0))
print(a.__getitem__(0)[1]) #'''
