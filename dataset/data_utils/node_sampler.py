from pprint import pformat
import random
import numpy as np
import time
import os
from utils.tsv_io import TSVDataset
from utils.tsv_io import load_list_file
from utils.tsv_io import get_tsv_lineidx, get_tsv_lineidx_8b
from utils.tsv_io import TSVFile
from utils.common import exclusive_open_to_read
import os.path as op
import logging
import torch.multiprocessing as mp
import math
from utils.common import list_to_dict
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from utils.common import get_mpi_rank, get_mpi_size, get_mpi_local_size, get_mpi_local_rank
from utils.common import print_frame_info
from azfuse import File as QDFile


def find_start_tsv_idx(num_each_tsv_file, node_offset):
    x = node_offset
    start_tsv_idx = None
    for tsv_idx, tsv_size in enumerate(num_each_tsv_file):
        x -= tsv_size
        if x < 0:
            start_tsv_idx = tsv_idx
            break
    assert start_tsv_idx is not None, (node_offset, sum(num_each_tsv_file))
    return start_tsv_idx

def construct_groups_by_chunk(num_each_tsv_file, offset, num, chunk_size):
    s = 0
    end_idx_each_tsv_file = []
    for x in num_each_tsv_file:
        end_idx_each_tsv_file.append(x + s)
        s += x
    curr_idx = offset
    num = min(num, end_idx_each_tsv_file[-1] - offset)
    tsv_groups = []
    while curr_idx < offset + num:
        curr_end_idx = curr_idx + chunk_size
        curr_end_idx = min(curr_end_idx, offset + num)
        g = {'start_idx': curr_idx, 'end_idx': curr_end_idx}
        tsv_idx1 = find_start_tsv_idx(num_each_tsv_file, curr_idx)
        tsv_idx2 = find_start_tsv_idx(num_each_tsv_file, curr_end_idx - 1)
        g['all_tsv_idx'] = list(range(tsv_idx1, tsv_idx2 + 1))
        tsv_groups.append(g)
        curr_idx = curr_end_idx
    return tsv_groups

def construct_tsv_groups(node_offset, num_on_node, start_tsv_idx,
                         num_each_tsv_file):
    tsv_groups = []
    start_idx = node_offset
    tsv_idx = start_tsv_idx
    end_idx_each_tsv_file = []
    s = 0
    for x in num_each_tsv_file:
        end_idx_each_tsv_file.append(x + s)
        s += x
    while True:
        g = {'start_idx': start_idx, 'tsv_idx': tsv_idx}
        #if tsv_idx == len(end_idx_each_tsv_file):
            #assert tsv_groups[-1]['end_idx'] == num_on_node + node_offset
            #break
        if end_idx_each_tsv_file[tsv_idx] < num_on_node + node_offset:
            g['end_idx'] = end_idx_each_tsv_file[tsv_idx]
            tsv_groups.append(g)
            tsv_idx += 1
            start_idx = g['end_idx']
        else:
            g['end_idx'] = num_on_node + node_offset
            tsv_groups.append(g)
            break
    return tsv_groups

class ScaleNodeSplitBySplitSampler(Sampler):
    def __init__(self, dataset, shuffle=False, random_seed=6, skip=0,
                 prepare_t_versions=[],
                 overwrite_dataset_len=None,
                 disable_prepare=False,
                 start_offset=0,
                 prepare_queue_len=0,
                 queue=None,
                 ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.world_size = get_mpi_size()
        self.local_size = get_mpi_local_size()
        self.node_size = self.world_size // self.local_size
        self.rank = get_mpi_rank()
        self.node_idx = self.rank // self.local_size
        self.local_rank = get_mpi_local_rank()
        #self.next_epoch_skip_shuffle = first_epoch_skip_shuffle
        self.overwrite_dataset_len = overwrite_dataset_len
        self.prepare_queue_len = prepare_queue_len

        # only be used when shuffle = True and first_epoch_skip_shuffle = True
        self.prepare = None
        self.prepare_t_versions = prepare_t_versions
        self.enable_prepare = not disable_prepare
        self.skip = skip
        self.start_offset = start_offset
        self.queue = queue
        assert not (start_offset > 0 and overwrite_dataset_len is not None)

    def __iter__(self):
        if self.skip > 0:
            logging.info('we will skip {}'.format(self.skip))
        if self.overwrite_dataset_len:
            # in vl-l sampler, we need to sample only the vl part, which is the
            # first part
            dataset_len = self.overwrite_dataset_len
        else:
            dataset_len = len(self.dataset)
        num = dataset_len - self.start_offset
        logging.info('num sample = {}'.format(num))
        num_on_node = (num + self.node_size - 1) // self.node_size
        node_offset = num_on_node * self.node_idx + self.start_offset
        num_on_node = min(num_on_node, dataset_len - node_offset)
        # each entry is the number of samples in one tsv, which means the
        # dataset should not be randomly and should be ordered in one tsv by
        # one tsv
        num_each_tsv_file = self.dataset.get_num_each_tsv()

        # find the tsv idx for the sample idx of node_offset
        start_tsv_idx = find_start_tsv_idx(num_each_tsv_file, node_offset)
        # construct tsv_groups
        tsv_groups = construct_tsv_groups(node_offset, num_on_node, start_tsv_idx,
                         num_each_tsv_file)

        if get_mpi_local_rank() == 0 and self.enable_prepare and self.prepare is None:
            self.prepare = PrepareData(self.dataset,
                                       prepare_t_versions=self.prepare_t_versions,
                                       #prepare_queue_len=self.prepare_queue_len,
                                       queue=self.queue,
                                       )

        local_random = random.Random()
        local_random.seed(self.random_seed)
        # prepare 3 tsv_split
        num_tsv_cache = 8
        local_rank_offset = self.local_rank
        while True:
            for idx_group, group in enumerate(tsv_groups):
                start_idx = group['start_idx']
                end_idx = group['end_idx']
                curr_idx = range(start_idx, end_idx)
                if self.shuffle:
                    curr_idx = list(curr_idx)
                    local_random.shuffle(curr_idx)
                if self.prepare and self.skip <= 0:
                    for i in range(num_tsv_cache):
                        g = tsv_groups[(idx_group + 1 + i) % len(tsv_groups)]
                        self.prepare.prepare(g['tsv_idx'])
                while local_rank_offset < len(curr_idx):
                    if self.skip <= 0:
                        yield curr_idx[local_rank_offset]
                    else:
                        self.skip -= 1
                    local_rank_offset += self.local_size
                local_rank_offset -= len(curr_idx)

    #def __len__(self):
    #    raise ValueError('should not be called')

class ScaleNodeSplitSampler(Sampler):
    def __init__(self, dataset, shuffle=False, random_seed=6, skip=0,
                 prepare_t_versions=[],
                 overwrite_dataset_len=None,
                 start_offset=0,
                 prepare_queue_len=0,
                 queue=None,
                 ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.world_size = get_mpi_size()
        self.local_size = get_mpi_local_size()
        self.node_size = self.world_size // self.local_size
        self.rank = get_mpi_rank()
        self.node_idx = self.rank // self.local_size
        self.local_rank = get_mpi_local_rank()
        #self.next_epoch_skip_shuffle = first_epoch_skip_shuffle
        self.overwrite_dataset_len = overwrite_dataset_len

        # only be used when shuffle = True and first_epoch_skip_shuffle = True
        self.prepare = None
        self.prepare_t_versions = prepare_t_versions
        self.enable_prepare = True
        self.skip = skip
        self.queue = queue
        self.start_offset = start_offset
        self.prepare_queue_len = prepare_queue_len

    def get_shufle_idx(self, n):
        g = torch.Generator()
        g.manual_seed(self.random_seed)
        random_idx = torch.randperm(n, generator=g)
        self.random_seed += 99
        return random_idx

    def __iter__(self):
        if self.skip > 0:
            logging.info('we will skip {}'.format(self.skip))
        idx_end_at = 0
        if self.overwrite_dataset_len:
            # in vl-l sampler, we need to sample only the vl part, which is the
            # first part
            num = self.overwrite_dataset_len
            idx_end_at = num
            assert self.start_offset == 0
        else:
            idx_end_at = len(self.dataset)
            num = idx_end_at - self.start_offset
        logging.info('num sample = {}'.format(num))
        num_on_node = (num + self.node_size - 1) // self.node_size
        node_offset = num_on_node * self.node_idx + self.start_offset
        chunk_size = 1048576
        chunk_size = chunk_size // self.local_size * self.local_size
        num_each_tsv_file = self.dataset.get_num_each_tsv()
        tsv_groups = construct_groups_by_chunk(num_each_tsv_file, node_offset, num_on_node, chunk_size)
        for t in tsv_groups[:-1]:
            assert t['start_idx'] < idx_end_at
            assert t['end_idx'] < idx_end_at
        t = tsv_groups[-1]
        assert t['start_idx'] < idx_end_at
        t['end_idx'] = min(idx_end_at, t['end_idx'])
        if get_mpi_local_rank() == 0 and self.prepare is None:
            self.prepare = PrepareData(self.dataset,
                                       prepare_t_versions=self.prepare_t_versions,
                                       #prepare_queue_len=self.prepare_queue_len,
                                       queue=self.queue,
                                       )
        num_cache_group = 8
        warmed_up = False
        while True:
            for idx_chunk, tsv_group in enumerate(tsv_groups):
                chunk_start = tsv_group['start_idx']
                chunk_end = tsv_group['end_idx']

                chunk_idxs = list(range(chunk_start + self.local_rank, chunk_end, self.local_size))
                all_sub_idx = self.get_shufle_idx(len(chunk_idxs)) if self.shuffle else range(len(chunk_idxs))

                prepare_idx = int(random.random() * len(all_sub_idx))
                prepare_idx = min(len(all_sub_idx) - 1, prepare_idx)
                logging.info(f'will prepare at {prepare_idx}/{len(all_sub_idx)}')

                for _idx, j in enumerate(all_sub_idx):
                    if self.prepare and self.skip <= 0 and (not warmed_up or _idx == prepare_idx) and self.enable_prepare:
                        if not warmed_up:
                            for cache_idx in range(num_cache_group):
                                self.prepare.prepare(tsv_groups[(idx_chunk + cache_idx)%len(tsv_groups)]['all_tsv_idx'])
                            warmed_up = True
                        else:
                            cache_idx = num_cache_group - 1
                            self.prepare.prepare(tsv_groups[(idx_chunk + cache_idx)%len(tsv_groups)]['all_tsv_idx'])
                    i = chunk_idxs[j]
                    if self.skip <= 0:
                        yield i
                    else:
                        self.skip -= 1

    #def __len__(self):
    #    raise ValueError('should not be called')

class NodeSplitSampler(Sampler):
    def __init__(self, dataset, shuffle, random_seed, first_epoch_skip_shuffle=False,
                 prepare_t_versions=[],
                 skip=0,
                 ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.world_size = get_mpi_size()
        self.local_size = get_mpi_local_size()
        self.node_size = self.world_size // self.local_size
        self.rank = get_mpi_rank()
        self.node_idx = self.rank // self.local_size
        self.local_rank = get_mpi_local_rank()
        self.next_epoch_skip_shuffle = first_epoch_skip_shuffle

        # only be used when shuffle = True and first_epoch_skip_shuffle = True
        self.prepare = None
        self.prepare_t_versions = prepare_t_versions
        self.skip = skip

    def get_index_on_node(self):
        # there is no need to cache source_list as we only call this function
        # once in the whole training life-time
        idx_split = self.dataset.get_composite_source_idx()
        idx_split = list(enumerate(idx_split))
        idx_split = torch.tensor(idx_split)
        if self.shuffle:
            random_idx = self.get_shufle_idx(len(idx_split))
            idx_split = idx_split[random_idx]
            max_split = idx_split[:, 1].max() + 1
            priority = self.get_shufle_idx(max_split)
            sort_idx = torch.argsort(priority[idx_split[:, 1]])
            idx_split = idx_split[sort_idx]
        num_idx_on_node = (len(idx_split) + self.node_size - 1) // self.node_size
        offset = num_idx_on_node * self.node_idx
        offset_end = offset + num_idx_on_node
        offset_end = min(offset_end, len(idx_split))
        if self.shuffle and self.next_epoch_skip_shuffle:
            unique_split_index = ordered_unique(idx_split[offset:offset_end, 1].tolist())
            logging.info(unique_split_index)
            if get_mpi_local_rank() == 0:
                self.prepare = PrepareData(self.dataset,
                                           prepare_t_versions=self.prepare_t_versions,
                                           )
                self.prepare.prepare(list(unique_split_index))
        return idx_split[offset:offset_end, 0]

    def get_shufle_idx(self, n):
        g = torch.Generator()
        g.manual_seed(self.random_seed)
        random_idx = torch.randperm(n, generator=g)
        self.random_seed += 99
        return random_idx

    def get_index_on_rank(self, idx_on_node):
        if self.shuffle:
            if not self.next_epoch_skip_shuffle:
                curr_idx_on_node = idx_on_node[self.get_shufle_idx(len(idx_on_node))]
            else:
                curr_idx_on_node = idx_on_node
                self.next_epoch_skip_shuffle = False
        else:
            curr_idx_on_node = idx_on_node
        idx_rank_size = (len(curr_idx_on_node) + self.local_size - 1) // self.local_size
        offset = idx_rank_size * self.local_rank
        offset_end = offset + idx_rank_size
        offset_end = min(offset_end, len(curr_idx_on_node))
        curr_idx_on_node = curr_idx_on_node.tolist()
        for i in range(offset, offset_end):
            yield curr_idx_on_node[i]

    def __iter__(self):
        self.curr_idx = 0
        idx_on_node = self.get_index_on_node()
        if self.skip > 0:
            logging.info('we will skip {}'.format(self.skip))
        while True:
            for i in self.get_index_on_rank(idx_on_node):
                if self.skip <= 0:
                    yield i
                else:
                    self.skip -= 1

    def __len__(self):
        return 100000
    #def __len__(self):
    #    raise ValueError('should not be called')

class RankSplitSampler(Sampler):
    def __init__(self, dataset, shuffle, random_seed):
        self.dataset = dataset
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.world_size = get_mpi_size()
        self.rank = get_mpi_rank()

    def get_index(self):
        source_list = self.dataset.get_composite_source_idx()
        idx_split = list(enumerate(source_list))
        idx_split = torch.tensor(idx_split)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.random_seed)
            random_idx = torch.randperm(len(idx_split), generator=g)
            idx_split = idx_split[random_idx]
        sort_idx = torch.argsort(idx_split[:, 1])
        idx_split = idx_split[sort_idx]
        rank_size = (len(idx_split) + self.world_size - 1) // self.world_size
        offset = rank_size * self.rank
        offset_end = offset + rank_size
        offset_end = min(offset_end, len(idx_split))
        return idx_split[offset:offset_end, 0].tolist()

    def __iter__(self):
        self.curr_idx = 0
        all_idx = self.get_index()
        while True:
            if self.curr_idx >= len(all_idx):
                self.curr_idx -= len(all_idx)
            yield all_idx[self.curr_idx]
            self.curr_idx += 1

    #def __len__(self):
    #    raise ValueError('should not be called')


class PrepareDataFileQueue(object):
    def __init__(self, folder):
        self.folder = folder
        from utils.common import ensure_remove_dir, ensure_directory
        logging.info('cleaning the folder of {}'.format(folder))
        ensure_remove_dir(folder)
        ensure_directory(folder)
        self.iter = 0
        self.curr_num_fnames = []

    def put(self, ls):
        fname = op.join(self.folder, str(self.iter))
        self.iter += 1
        if isinstance(ls, str):
            ls = [ls]
        from utils.tsv_io import write_to_file
        write_to_file('\n'.join(ls), fname)

    def get(self):
        # get should not rely on the variable from put() as this function may be executed in anohter process
        import glob
        while True:
            if len(self.curr_num_fnames) == 0:
                fnames = glob.glob(op.join(self.folder, '*'))
                if len(fnames) == 0:
                    time.sleep(10)
                    continue
                basenames = [op.basename(f) for f in fnames]
                num_fnames = [(int(b), f) for b, f in zip(basenames, fnames) if 'tmp' not in b]
                #num_fnames = [(int(op.basename(f)), f) for f in fnames]
                num_fnames = sorted(num_fnames, key=lambda x: x[0])
                self.curr_num_fnames.extend(num_fnames)
            fn = self.curr_num_fnames[0][1]
            ls = load_list_file(fn)
            from utils.common import ensure_remove_file
            ensure_remove_file(fn)
            del self.curr_num_fnames[0]
            return ls

def create_prepare_tsv_file_process(max_len=0):
    use_thread = True
    if use_thread:
        import threading
        import queue
        prepare_queue = queue.Queue()
        p = threading.Thread(
            target=prepare_tsv_file_process, args=(prepare_queue, max_len),
            daemon=True,
        )
        p.start()
    else:
        prepare_queue = mp.Queue()
        p = mp.Process(
            target=prepare_tsv_file_process, args=(prepare_queue, max_len),
            daemon=True,
        )
        p.start()
    return p, prepare_queue

def prepare_tsv_file_process(queue, max_len=0):
    if os.environ.get('QD_TSV_USE_FUSE'):
        if int(os.environ['QD_TSV_USE_FUSE']):
            ftype = 'fuser'
        else:
            ftype = 'blobfuse'
    else:
        ftype = 'blobfuse'

    if ftype == 'fuser':
        from utils.cloud_storage import create_cloud_fuse
        fuser = create_cloud_fuse()
    logging.info('ftype = {}'.format(ftype))

    prepared = []

    while True:
        start = time.time()
        fnames = queue.get()
        end = time.time()
        if (end - start) > 5:
            logging.info('waiting {} to get a new tsv to prepare'.format(
                end - start))
        curr_fs = []
        for fname in fnames:
            curr_fs.append(fname)
            if fname.endswith('.tsv'):
                lineidx8b = get_tsv_lineidx_8b(fname)
                curr_fs.append(lineidx8b)

        def unprepare(info):
            logging.info('unprepare {}'.format(info['fnames']))
            if ftype == 'blobfuse':
                for f in info['fps']:
                    f.close()
                logging.info('unprepared {}'.format(info['fnames']))
            else:
                to_remove = []
                for f in info['fnames']:
                    if all(f not in fs['fnames'] for fs in prepared):
                        to_remove.append(f)
                        fuser.ensure_del_cache(f)
                logging.info('unprepared {}'.format(to_remove))

        sames = [i for i, p in enumerate(prepared)
                   if all(f in p['fnames'] for f in curr_fs)]
        if len(sames) > 0 and ftype == 'blobfuse':
            # if it is cloud-fuse, we will check if the file exists in disk
            i = sames[0]
            p = prepared[i]
            del prepared[i]
            prepared.append(p)
            logging.info('no need to prepare {} as it prepared'.format(
                curr_fs
            ))
            continue

        while max_len > 0 and len(prepared) >= max_len:
            unprepare(prepared.pop(0))

        logging.info('prepare {}'.format(curr_fs))
        start = time.time()
        if ftype == 'blobfuse':
            info = {
                'fnames': curr_fs,
                'fps': [exclusive_open_to_read(x) for x in curr_fs]
            }
            prepared.append(info)
        else:
            info = {
                'fnames': curr_fs
            }
            if len(info['fnames']) > 0:
                prepared.append(info)
            fuser.ensure_cache(curr_fs, touch_cache_if_exist=True)
        logging.info('use {}s, prepared {}; max len = {}; curr len = {}; all prepared = {}'.format(
            time.time() - start,
            curr_fs,
            max_len,
            len(prepared),
            ';'.join([','.join(p['fnames']) for p in prepared]),
        ))
        time.sleep(random.random() * 5)

def ordered_unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

class PrepareData(object):
    fuser = None
    def __init__(self, dataset, prepare_t_versions=[],
                 fixed_samples_in_node=False,
                 disable_prepare=None,
                 disable_garbage_collection=False,
                 prepare_queue_len=0,
                 queue=None,
                 ):
        self.prepare_files = None
        #self.prepare_process = None
        self.dataset = dataset
        self.prepare_t_versions = prepare_t_versions
        #self.fixed_samples_in_node = fixed_samples_in_node
        self.disable_prepare = disable_prepare or (queue is None)
        #self.prepare_queue_len = prepare_queue_len
        self.queue = queue
        #assert queue is not None or disable_prepare

    def get_composite_source_files(self):
        data = self.dataset.dataset.data
        split = self.dataset.dataset.split
        dataset = TSVDataset(data)
        result = []
        for t, version in self.prepare_t_versions:
            tsv = dataset.get_data(split, t, version)
            if QDFile.isfile(tsv):
                result.append([(tsv,0)])
            else:
                x_tsv = dataset.get_data(split + 'X', t, version)
                assert op.isfile(x_tsv)
                result.append(TSVFile(x_tsv))
        return result

    def prepare(self, split):
        if self.disable_prepare:
            return
        self.ensure_init_prepare()
        q = self.prepare_queue
        if isinstance(split, (list, tuple)):
            tsv_files = [ps[s][0] for s in split for ps in self.prepare_files]
        else:
            tsv_files = [ps[split][0] for ps in self.prepare_files]
        logging.info('send to prepare: {}'.format(','.join(tsv_files)))
        q.put(tsv_files)

    def prepare_all(self, tsv_idx_start=0):
        if self.disable_prepare:
            return
        self.ensure_init_prepare()
        q = self.prepare_queue
        tsv_files = []
        for ps in self.prepare_files:
            for i in range(tsv_idx_start, len(ps)):
                tsv_files.append(ps[i][0])
        assert len(tsv_files) > 0, 'maybe bug?'
        q.put(tsv_files)

    def ensure_init_prepare(self):
        if self.prepare_files is None:
            self.prepare_files = self.get_composite_source_files()
        assert self.queue is not None
        self.prepare_queue = self.queue

class PrepareData_old(object):
    fuser = None
    def __init__(self, dataset, prepare_t_versions=[],
                 fixed_samples_in_node=False,
                 disable_prepare=None,
                 disable_garbage_collection=False,
                 prepare_queue_len=0,
                 queue=None,
                 ):
        self.prepare_files = None
        #self.prepare_process = None
        self.dataset = dataset
        self.prepare_t_versions = prepare_t_versions
        #self.fixed_samples_in_node = fixed_samples_in_node
        self.disable_prepare = disable_prepare or (queue is None)
        #self.prepare_queue_len = prepare_queue_len
        self.queue = queue
        #assert queue is not None or disable_prepare

    def get_composite_source_files(self):
        data = self.dataset.dataset.data
        split = self.dataset.dataset.split
        dataset = TSVDataset(data)
        result = []
        for t, version in self.prepare_t_versions:
            tsv = dataset.get_data(split, t, version)
            if QDFile.isfile(tsv):
                result.append([tsv])
            else:
                x_tsv = dataset.get_data(split + 'X', t, version)
                assert op.isfile(x_tsv)
                result.append(load_list_file(x_tsv))
        return result

    def prepare(self, split):
        if self.disable_prepare:
            return
        self.ensure_init_prepare()
        q = self.prepare_queue
        #size = q.qsize()
        #if size > 100:
            #logging.info('prepare queue is too long {}'.format(size))
        if isinstance(split, (list, tuple)):
            tsv_files = [ps[s] for s in split for ps in self.prepare_files]
        else:
            tsv_files = [ps[split] for ps in self.prepare_files]
        logging.info('send to prepare: {}'.format(','.join(tsv_files)))
        q.put(tsv_files)

    def prepare_all(self, tsv_idx_start=0):
        if self.disable_prepare:
            return
        self.ensure_init_prepare()
        q = self.prepare_queue
        tsv_files = []
        for ps in self.prepare_files:
            tsv_files.extend(ps[tsv_idx_start:])
        assert len(tsv_files) > 0, 'maybe bug?'
        q.put(tsv_files)

    def ensure_init_prepare(self):
        if self.prepare_files is None:
            self.prepare_files = self.get_composite_source_files()
        assert self.queue is not None
        self.prepare_queue = self.queue
        #if self.prepare_process is None:
            #if self.fixed_samples_in_node:
                #max_len = 0
            #else:
                #max_len = self.prepare_queue_len
            #logging.info('creating prepare thread from PrepareData')
            #p, prepare_queue = create_prepare_tsv_file_process(
                #max_len=max_len,
            #)
            #self.prepare_process = p
            #self.prepare_queue = prepare_queue

def create_download_worker(queue, max_len=0):
    p = mp.Process(
        target=prepare_tsv_file_process, args=(queue, max_len),
        daemon=True,
    )
    p.start()
    return p

class SplitBySplitSampler(Sampler):
    def __init__(self, dataset, group_size=1, shuffle=True,
                 fixed_samples_in_node=False,
                 random_seed=9,
                 skip=0,
                 prepare_t_versions=[],
                 disable_prepare=None,
                 ):
        print_frame_info()
        self.dataset = dataset
        self.group_size = group_size
        self.random_seed = random_seed
        self.shuffle = shuffle

        self.rank = get_mpi_rank()
        self.local_rank = get_mpi_local_rank()
        self.world_size = get_mpi_size()
        self.local_size = get_mpi_local_size()

        self.node_size = self.world_size // self.local_size
        self.node_idx = self.rank // self.local_size

        self.shuffle_group_process = None

        self.prepare_process = None
        self.prepare_queue = None
        self.prepare_files = None
        # currently, we only support to prepare one kind of files, but it could
        # be extendeed to multiple files if we need
        self.prepare_t_versions = prepare_t_versions
        self.sub_process_create_shuffle = False
        self._idx_split = None
        self.iter_shuffle_group = None

        self.curr_group_buffers = None
        self.next_group_index = 0
        self.cache_group_index_on_node = None

        self.disable_prepare = disable_prepare
        self.get_group_process = None
        self.fixed_samples_in_node = fixed_samples_in_node
        self.skip = skip

    def get_composite_source_idx(self):
        return self.dataset.get_composite_source_idx()

    def get_composite_source_files(self):
        data = self.dataset.dataset.data
        split = self.dataset.dataset.split
        dataset = TSVDataset(data)
        result = []
        for t, version in self.prepare_t_versions:
            tsv = dataset.get_data(split, t, version)
            if op.isfile(tsv):
                result.append([tsv])
            else:
                x_tsv = dataset.get_data(split + 'X', t, version)
                assert op.isfile(x_tsv)
                result.append(load_list_file(x_tsv))
        return result

    def load_idx_split(self):
        logging.info('loading source list')
        source_list = self.get_composite_source_idx()
        logging.info('loaded source list')
        idx_split = list(enumerate(source_list))
        idx_split = torch.tensor(idx_split)
        return idx_split

    @property
    def idx_split(self):
        if self._idx_split is None:
            self._idx_split = self.load_idx_split()
            #self._idx_split.share_memory_()
        return self._idx_split

    def get_shufle_idx(self, n):
        g = torch.Generator()
        g.manual_seed(self.random_seed)
        random_idx = torch.randperm(n, generator=g)
        self.random_seed += 99
        return random_idx

    def get_group_index_on_node_random(self):
        idx_split = self.idx_split

        max_split = idx_split[:, 1].max() + 1
        priority = self.get_shufle_idx(max_split)

        random_idx = self.get_shufle_idx(len(idx_split))
        idx_split = idx_split[random_idx]

        idx_split = torch.cat([idx_split[idx_split[:, 1] == p] for p in priority])

        num_idx_on_node = (len(idx_split) + self.node_size - 1) // self.node_size
        offset = num_idx_on_node * self.node_idx
        offset_end = offset + num_idx_on_node
        offset_end = min(offset_end, len(idx_split))
        idx_split = idx_split[offset:offset_end]

        unique_split_index = ordered_unique(idx_split[:, 1].tolist())
        logging.info(unique_split_index)
        result = [
            {
                'idx_in_group': idx_split[idx_split[:, 1] == s][:, 0].tolist(),
                'split_in_group': s,
            }
            for s in unique_split_index
        ]
        return result

    def get_group_index_on_node(self):
        if self.shuffle and not self.fixed_samples_in_node:
            return self.get_group_index_on_node_random()
        elif self.shuffle and self.fixed_samples_in_node:
            if self.cache_group_index_on_node is None:
                self.cache_group_index_on_node = self.get_group_index_on_node_random()
            idx = self.get_shufle_idx(len(self.cache_group_index_on_node))
            group_in_node = [self.cache_group_index_on_node[i] for i in idx]
            for g in group_in_node:
                idx = self.get_shufle_idx(len(g['idx_in_group']))
                g['idx_in_group'] = [g['idx_in_group'][i] for i in idx]
            return group_in_node
        else:
            if self.cache_group_index_on_node is None:
                self.cache_group_index_on_node = self.get_group_index_on_node_random()
            return self.cache_group_index_on_node

    def get_next_group_index_on_node(self):
        if self.curr_group_buffers is None:
            self.curr_group_buffers = self.get_group_index_on_node()
            self.next_group_index = 0
        if self.next_group_index >= len(self.curr_group_buffers):
            self.curr_group_buffers = self.get_group_index_on_node()
            self.next_group_index = 0
        g = self.curr_group_buffers[self.next_group_index]
        self.next_group_index += 1
        return g

    def get_group_thread(self, q):
        while True:
            if q.qsize() < 8:
                g = self.get_next_group_index_on_node()
                q.put(g)
            else:
                time.sleep(1)

    def __iter__(self):
        use_thread_to_get_group = True
        if not use_thread_to_get_group:
            assert self.skip == 0, 'not supported'
            group_buffers = [self.get_next_group_index_on_node()
                             for _ in range(4)]
            if self.local_rank == 0:
                for g in group_buffers:
                    self.prepare(g['split_in_group'])
            assert len(group_buffers) > 0
            idx = self.local_rank
            while True:
                while idx >= len(group_buffers[0]['idx_in_group']):
                    idx -= len(group_buffers[0]['idx_in_group'])
                    group_buffers.pop(0)
                    new_g = self.get_next_group_index_on_node()
                    if self.local_rank == 0:
                        self.prepare(new_g['split_in_group'])
                    group_buffers.append(new_g)
                r = group_buffers[0]['idx_in_group'][idx]
                yield r
                idx += self.local_size
        else:
            self.ensure_init_get_group_thread()
            group_buffers = [self.get_group_queue.get()
                             for _ in range(4)]
            if self.local_rank == 0:
                for g in group_buffers:
                    if self.skip <= 0:
                        self.prepare(g['split_in_group'])
            assert len(group_buffers) > 0
            idx = self.local_rank
            while True:
                while idx >= len(group_buffers[0]['idx_in_group']):
                    idx -= len(group_buffers[0]['idx_in_group'])
                    group_buffers.pop(0)
                    start = time.time()
                    new_g = self.get_group_queue.get()
                    cost = time.time() - start
                    logging.info('time to get group index on node: {}'.format(cost))
                    if self.local_rank == 0 and self.skip <= 0:
                        self.prepare(new_g['split_in_group'])
                    group_buffers.append(new_g)
                r = group_buffers[0]['idx_in_group'][idx]
                if self.skip <= 0:
                    yield r
                else:
                    self.skip -= 1
                idx += self.local_size

    def ensure_init_get_group_thread(self):
        if self.get_group_process is None:
            import threading
            import queue
            q = queue.Queue()
            t = threading.Thread(
                target=self.get_group_thread, args=(q,),
                daemon=True,
            )
            t.start()
            self.get_group_process = t
            self.get_group_queue = q

    def ensure_init_prepare(self):
        if self.prepare_files is None:
            self.prepare_files = self.get_composite_source_files()
        if self.prepare_process is None:
            max_len = 8 if not self.fixed_samples_in_node else 0
            p, prepare_queue = create_prepare_tsv_file_process(
                max_len=max_len)
            self.prepare_process = p
            self.prepare_queue = prepare_queue

    def prepare(self, split):
        if self.disable_prepare:
            return
        self.ensure_init_prepare()
        q = self.prepare_queue
        size = q.qsize()
        if size > 100:
            logging.info('prepare queue is too long {}'.format(size))
        q.put([ps[split] for ps in self.prepare_files])

    #def __len__(self):
    #    raise ValueError('should not be called')

class AttachIterationNumberBatchSampler(object):
    def __init__(self, batch_sampler, start_iter, num_iters,
                 gradient_accumulate=1):
        self.batch_sampler = batch_sampler
        self.curr_iter = start_iter
        self.max_iter = num_iters
        self.gradient_accumulate = gradient_accumulate

    def __getattr__(self, att):
        return getattr(self.batch_sampler, att)

    def __iter__(self):
        #if hasattr(self.batch_sampler, 'skip') and self.curr_iter > 0:
            #logging.info('we will skip {} batches'.format(self.curr_iter))
            #self.batch_sampler.skip(self.curr_iter)
        for idx_batch, batch in enumerate(self.batch_sampler):
            batch = [{'iteration': self.curr_iter,
                      'idx': i,
                      'max_iter': self.max_iter} for i in batch]
            yield batch
            if (idx_batch + 1) % self.gradient_accumulate == 0:
                self.curr_iter += 1

    def __len__(self):
        return len(self.batch_sampler)

class OrderedSplitSampler(Sampler):
    def __init__(self, data_length):
        curr_rank = get_mpi_rank()
        world_size = get_mpi_size()
        rank_size = (data_length + world_size - 1) // world_size
        start = rank_size * curr_rank
        end = start + rank_size
        assert start >= 0 and start <= data_length
        if curr_rank < world_size - 1:
            assert end >= 0 and end <= data_length
        end = min(end, data_length)
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start

class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0,
                 ):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

        if hasattr(batch_sampler, 'batch_size'):
            self.batch_size = batch_sampler.batch_size

        if hasattr(batch_sampler, 'drop_last'):
            self.drop_last = batch_sampler.drop_last

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class DynamicBatchSampler(BatchSampler):
    def __init__(self, sampler, get_batch_size, start_iter=0):
        self.sampler = sampler
        self.get_batch_size = get_batch_size
        self.start_iter = start_iter

    def __iter__(self):
        batch = []
        batch_size = None
        curr_iter = self.start_iter
        for idx in self.sampler:
            batch.append(idx)
            if batch_size is None:
                batch_size = self.get_batch_size(curr_iter)
            if len(batch) == batch_size:
                yield batch
                batch_size = None
                curr_iter += 1
                batch = []

class InfinityDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 skip=0,
                 start_offset=0,
                 prepare_t_versions=[],
                 queue=None,
                 ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            from utils.common import get_mpi_size
            num_replicas = get_mpi_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            from utils.common import get_mpi_rank
            rank = get_mpi_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.start_offset = start_offset
        self.num_samples = int(math.ceil(self.get_effective_dataset_len() * 1.0 / self.num_replicas))
        length_divisible = num_replicas
        if length_divisible > 1:
            import logging
            logging.info('before making divisible = {}'.format(self.num_samples))
            self.num_samples = ((self.num_samples + length_divisible - 1) //
                    length_divisible) * length_divisible
            logging.info('adjust to = {}'.format(self.num_samples))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.skip = skip
        self.num_passed = 0
        # used in vl-l samplers
        self.queue = queue
        self.prepare = None
        self.prepare_t_versions = prepare_t_versions

    def get_effective_dataset_len(self):
        return len(self.dataset) - self.start_offset

    def state_dict(self):
        return {'num_passed': self.num_passed}

    def load_state_dict(self, info):
        if info is not None:
            self.skip = info['num_passed']

    def __iter__(self):
        self.num_passed = 0
        if get_mpi_local_rank() == 0 and self.prepare is None:
            self.prepare = PrepareData(self.dataset,
                                       prepare_t_versions=self.prepare_t_versions,
                                       queue=self.queue,
                                       )
            if self.start_offset == 0:
                self.prepare.prepare_all()
            else:
                num_each_tsv_file = self.dataset.get_num_each_tsv()
                start_tsv_idx = find_start_tsv_idx(num_each_tsv_file, self.start_offset)
                self.prepare.prepare_all(start_tsv_idx)
        while True:
            if self.shuffle:
                # deterministically shuffle based on epoch
                g = torch.Generator()
                g.manual_seed(self.epoch)
                self.epoch += 1
                indices = torch.randperm(self.get_effective_dataset_len(), generator=g).tolist()
            else:
                indices = torch.arange(self.get_effective_dataset_len()).tolist()

            while len(indices) < self.total_size:
                if 2 * len(indices) < self.total_size:
                    indices += indices
                else:
                    indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset : offset + self.num_samples]
            assert len(indices) == self.num_samples
            for i in indices:
                if self.skip > 0:
                    self.skip -= 1
                else:
                    yield i + self.start_offset
                self.num_passed += 1

    def __len__(self):
        raise ValueError('invalid')


class DistributedSampler(Sampler):
    # should only be used during testing
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            from utils.common import get_mpi_size
            num_replicas = get_mpi_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            from utils.common import get_mpi_rank
            rank = get_mpi_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        length_divisible = num_replicas
        if length_divisible > 1:
            logging.info('before making divisible = {}'.format(self.num_samples))
            self.num_samples = ((self.num_samples + length_divisible - 1) //
                    length_divisible) * length_divisible
            logging.info('adjust to = {}'.format(self.num_samples))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        logging.info('force shuffle = False')
        self.shuffle = False

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        while len(indices) < self.total_size:
            if 2 * len(indices) <= self.total_size:
                indices += indices
            else:
                indices += indices[: (self.total_size - len(indices))]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ScaleNodeSplitVLPlusScaleNodeSplitLBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 one_fwd_batch_size_per_gpu,
                 one_fwd_batch_size_per_gpu_text_only,
                 gradient_accumulate,
                 max_iter,
                 train_shuffle,
                 vl_data_rate,
                 prepare_t_versions=[],
                 vl_queue=None,
                 l_queue=None,
                 skip=0,
                 ):
        print_frame_info()
        self.dataset = dataset
        self.one_fwd_batch_size_per_gpu = [
            one_fwd_batch_size_per_gpu,
            one_fwd_batch_size_per_gpu_text_only,
        ]
        self.gradient_accumulate = gradient_accumulate
        self.max_iter = max_iter
        self.train_shuffle = train_shuffle
        self.rank = get_mpi_rank()
        self.world_size = get_mpi_size()
        self.batch_size_per_gpu = [
            one_fwd_batch_size_per_gpu * gradient_accumulate,
            one_fwd_batch_size_per_gpu_text_only * gradient_accumulate,
        ]
        self.prepare_t_versions = prepare_t_versions

        self.vl_data_rate = vl_data_rate
        self.vl_queue = vl_queue
        self.l_queue = l_queue

        self.skip_iter = skip
        self.passed_iter = skip

    def state_dict(self):
        return {'passed_iter': self.passed_iter}

    def load_state_dict(self, info):
        self.skip_iter = info['passed_iter']

    def __iter__(self):
        logging.info('we will skip {}'.format(self.skip_iter))
        vl_l_info = self.dataset.get_vl_l_info_from_cache()
        vl_sampler = ScaleNodeSplitSampler(
            self.dataset,
            shuffle=self.train_shuffle,
            prepare_t_versions=self.prepare_t_versions,
            overwrite_dataset_len=vl_l_info['num_valid_vl_pair'],
            queue=self.vl_queue,
        )
        l_sampler = ScaleNodeSplitSampler(
            self.dataset,
            shuffle=self.train_shuffle,
            start_offset=vl_l_info['num_valid_vl_pair'],
            prepare_t_versions=[('caption', None)],
            queue=self.l_queue,
        )
        samplers = [vl_sampler, l_sampler]
        samplers = [iter(s) for s in samplers]

        if self.vl_data_rate is None:
            self.vl_data_rate = vl_l_info['num_valid_vl_pair'] / vl_l_info['num_total']
            logging.info(f'{self.vl_data_rate}')

        if self.skip_iter > 0:
            vl_sampler.enable_prepare = False
            l_sampler.enable_prepare = False

        passed_data = [0, 0]
        for iteration in range(self.max_iter):
            random.seed(self.passed_iter)
            r = random.random()
            source_idx = 0 if r <= self.vl_data_rate else 1
            passed_data[source_idx] += 1
            if (iteration % 100) == 0:
                logging.info(f'vl iter = {passed_data[0]}; l iter={passed_data[1]}')

            batch_size_per_gpu = self.batch_size_per_gpu[source_idx]
            one_fwd_batch_size_per_gpu = self.one_fwd_batch_size_per_gpu[source_idx]
            # we should run next(sampler) before skipping logic
            ret = [next(samplers[source_idx]) for _ in range(batch_size_per_gpu)]

            if self.skip_iter > 0:
                self.skip_iter -= 1
            else:
                if not vl_sampler.enable_prepare:
                    vl_sampler.enable_prepare = True
                if not l_sampler.enable_prepare:
                    l_sampler.enable_prepare = True
                for i in range(self.gradient_accumulate):
                    yield [{
                        'idx': j,
                        'iteration': self.passed_iter,
                    } for j in ret[
                        i * one_fwd_batch_size_per_gpu:
                        (i + 1) * one_fwd_batch_size_per_gpu
                    ]]
            self.passed_iter += 1

    def __len__(self):
        return self.max_iter * self.gradient_accumulate

class ScaleNodeSplitVLPlusRandomLBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 one_fwd_batch_size_per_gpu,
                 one_fwd_batch_size_per_gpu_text_only,
                 gradient_accumulate,
                 max_iter,
                 train_shuffle,
                 vl_data_rate,
                 prepare_t_versions=[],
                 vl_queue=None,
                 l_queue=None,
                 skip=0,
                 ):
        print_frame_info()
        self.dataset = dataset
        self.one_fwd_batch_size_per_gpu = [
            one_fwd_batch_size_per_gpu,
            one_fwd_batch_size_per_gpu_text_only,
        ]
        self.gradient_accumulate = gradient_accumulate
        self.max_iter = max_iter
        self.train_shuffle = train_shuffle
        self.rank = get_mpi_rank()
        self.world_size = get_mpi_size()
        self.batch_size_per_gpu = [
            one_fwd_batch_size_per_gpu * gradient_accumulate,
            one_fwd_batch_size_per_gpu_text_only * gradient_accumulate,
        ]
        self.prepare_t_versions = prepare_t_versions

        self.vl_data_rate = vl_data_rate
        self.vl_queue = vl_queue
        self.l_queue = l_queue

        self.skip_iter = skip
        self.passed_iter = skip

    def state_dict(self):
        return {'passed_iter': self.passed_iter}

    def load_state_dict(self, info):
        self.skip_iter = info['passed_iter']

    def __iter__(self):
        logging.info('we will skip {}'.format(self.skip_iter))
        vl_l_info = self.dataset.get_vl_l_info_from_cache()
        vl_sampler = ScaleNodeSplitSampler(
            self.dataset,
            shuffle=self.train_shuffle,
            prepare_t_versions=self.prepare_t_versions,
            overwrite_dataset_len=vl_l_info['num_valid_vl_pair'],
            queue=self.vl_queue,
        )
        l_sampler = InfinityDistributedSampler(
            self.dataset,
            shuffle=self.train_shuffle,
            start_offset=vl_l_info['num_valid_vl_pair'],
            prepare_t_versions=[('caption', None)],
            queue=self.l_queue,
        )
        samplers = [vl_sampler, l_sampler]
        samplers = [iter(s) for s in samplers]

        if self.vl_data_rate is None:
            self.vl_data_rate = vl_l_info['num_valid_vl_pair'] / vl_l_info['num_total']
            logging.info(f'{self.vl_data_rate}')

        passed_data = [0, 0]
        if self.skip_iter > 0:
            vl_sampler.enable_prepare = False
        for iteration in range(self.max_iter):
            random.seed(self.passed_iter)
            r = random.random()
            source_idx = 0 if r <= self.vl_data_rate else 1
            passed_data[source_idx] += 1
            if (iteration % 100) == 0 and self.skip_iter <= 0:
                logging.info(f'vl iter = {passed_data[0]}; l iter={passed_data[1]}')

            batch_size_per_gpu = self.batch_size_per_gpu[source_idx]
            one_fwd_batch_size_per_gpu = self.one_fwd_batch_size_per_gpu[source_idx]
            # we should run next(sampler) before skipping logic
            ret = [next(samplers[source_idx]) for _ in range(batch_size_per_gpu)]

            if self.skip_iter > 0:
                self.skip_iter -= 1
            else:
                if not vl_sampler.enable_prepare:
                    vl_sampler.enable_prepare = True
                for i in range(self.gradient_accumulate):
                    yield [{
                        'idx': j,
                        'iteration': self.passed_iter,
                    } for j in ret[
                        i * one_fwd_batch_size_per_gpu:
                        (i + 1) * one_fwd_batch_size_per_gpu
                    ]]
            self.passed_iter += 1

    def __len__(self):
        return self.max_iter * self.gradient_accumulate

class S3VLPlusS3LBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 one_fwd_batch_size_per_gpu,
                 one_fwd_batch_size_per_gpu_text_only,
                 gradient_accumulate,
                 max_iter,
                 train_shuffle,
                 vl_data_rate,
                 prepare_t_versions=[],
                 disable_prepare=False,
                 vl_queue=None,
                 l_queue=None,
                 ):
        print_frame_info()
        self.dataset = dataset
        self.one_fwd_batch_size_per_gpu = [
            one_fwd_batch_size_per_gpu,
            one_fwd_batch_size_per_gpu_text_only,
        ]
        self.gradient_accumulate = gradient_accumulate
        self.max_iter = max_iter
        self.train_shuffle = train_shuffle
        self.rank = get_mpi_rank()
        self.world_size = get_mpi_size()
        self.batch_size_per_gpu = [
            one_fwd_batch_size_per_gpu * gradient_accumulate,
            one_fwd_batch_size_per_gpu_text_only * gradient_accumulate,
        ]
        self.prepare_t_versions = prepare_t_versions

        self.vl_data_rate = vl_data_rate
        self.disable_prepare = disable_prepare
        self.vl_queue = vl_queue
        self.l_queue = l_queue

        self.skip_iter = 0
        self.passed_iter = 0

    def state_dict(self):
        return {'passed_iter': self.passed_iter}

    def load_state_dict(self, info):
        self.skip_iter = info['passed_iter']

    def __iter__(self):
        logging.info('we will skip {}'.format(self.skip_iter))
        vl_l_info = self.dataset.get_vl_l_info_from_cache()
        vl_sampler = ScaleNodeSplitBySplitSampler(
            self.dataset,
            shuffle=self.train_shuffle,
            prepare_t_versions=self.prepare_t_versions,
            overwrite_dataset_len=vl_l_info['num_valid_vl_pair'],
            disable_prepare=self.disable_prepare,
            queue=self.vl_queue,
        )
        l_sampler = ScaleNodeSplitBySplitSampler(
            self.dataset,
            shuffle=self.train_shuffle,
            prepare_t_versions=[('caption', None)],
            start_offset=vl_l_info['num_valid_vl_pair'],
            prepare_queue_len=0,
            disable_prepare=self.disable_prepare,
            queue=self.l_queue,
        )
        samplers = [vl_sampler, l_sampler]
        samplers = [iter(s) for s in samplers]

        if self.vl_data_rate is None:
            self.vl_data_rate = vl_l_info['num_valid_vl_pair'] / vl_l_info['num_total']
            logging.info(f'{self.vl_data_rate}')

        passed_data = [0, 0]
        for iteration in range(self.max_iter):
            random.seed(self.passed_iter)
            r = random.random()
            source_idx = 0 if r <= self.vl_data_rate else 1
            passed_data[source_idx] += 1
            if (iteration % 10) == 0:
                logging.info(f'vl iter = {passed_data[0]}; l iter={passed_data[1]}')

            batch_size_per_gpu = self.batch_size_per_gpu[source_idx]
            one_fwd_batch_size_per_gpu = self.one_fwd_batch_size_per_gpu[source_idx]
            # we should run next(sampler) before skipping logic
            ret = [next(samplers[source_idx]) for _ in range(batch_size_per_gpu)]

            if self.skip_iter > 0:
                self.skip_iter -= 1
            else:
                for i in range(self.gradient_accumulate):
                    yield [{
                        'idx': j,
                        'iteration': self.passed_iter,
                    } for j in ret[
                        i * one_fwd_batch_size_per_gpu:
                        (i + 1) * one_fwd_batch_size_per_gpu
                    ]]
            self.passed_iter += 1

    def __len__(self):
        return self.max_iter * self.gradient_accumulate


