import time
from utils.tsv_io import TSVDataset
from utils.tsv_file import load_list_file
from utils.tsv_io import get_tsv_lineidx, get_tsv_lineidx_8b
from utils.common import exclusive_open_to_read
import os.path as op
import logging
import math
import random
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import os
import torch.multiprocessing as mp
from utils.dist import (
    get_local_rank, get_local_size, get_rank, get_world_size)


class RankSplitSampler(Sampler):
    def __init__(self, dataset, shuffle, random_seed):
        self.dataset = dataset
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.world_size = get_world_size()
        self.rank = get_rank()

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

    def __len__(self):
        raise ValueError('should not be called')


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
            # else:
            #     to_remove = []
            #     for f in info['fnames']:
            #         if all(f not in fs['fnames'] for fs in prepared):
            #             to_remove.append(f)
            #             fuser.ensure_del_cache(f)
            #     logging.info('unprepared {}'.format(to_remove))

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
    def __init__(self, dataset, prepare_t_versions=[],
                 fixed_samples_in_node=False,
                 disable_prepare=None,
                 ):
        self.prepare_files = None
        self.prepare_process = None
        self.dataset = dataset
        self.prepare_t_versions = prepare_t_versions
        self.fixed_samples_in_node = fixed_samples_in_node
        self.disable_prepare = disable_prepare

    def get_composite_source_files(self):
        root = self.dataset.root
        assert self.dataset.is_composite
        result = []
        for t in ['visual_tsv', 'label_tsv', 'cap_tsv']:
            tsv = getattr(self.dataset, t, None)
            # tsv.ensure_initialized()
            if tsv is not None:
                result.append(
                    [f if op.exists(f) else op.join(root, f)
                     for f in tsv.file_list])

        # data = self.dataset.dataset.data
        # split = self.dataset.dataset.split
        # dataset = TSVDataset(data)
        # result = []
        # for t, version in self.prepare_t_versions:
        #     tsv = dataset.get_data(split, t, version)
        #     if op.isfile(tsv):
        #         result.append([tsv])
        #     else:
        #         x_tsv = dataset.get_data(split + 'X', t, version)
        #         assert op.isfile(x_tsv)
        #         result.append(load_list_file(x_tsv))
        return result

    def prepare(self, split):
        if self.disable_prepare:
            return
        self.ensure_init_prepare()
        q = self.prepare_queue
        size = q.qsize()
        if size > 100:
            logging.info('prepare queue is too long {}'.format(size))
        q.put([ps[split] for ps in self.prepare_files])

    def ensure_init_prepare(self):
        if self.prepare_files is None:
            self.prepare_files = self.get_composite_source_files()
        if self.prepare_process is None:
            max_len = 8 if not self.fixed_samples_in_node else 0
            p, prepare_queue = create_prepare_tsv_file_process(
                max_len=max_len)
            self.prepare_process = p
            self.prepare_queue = prepare_queue

            if os.environ.get('QD_TSV_USE_FUSE') and \
                    int(os.environ['QD_TSV_USE_FUSE']) and \
                    not self.fixed_samples_in_node:
                from utils.cloud_storage import create_cloud_fuse
                fuser = create_cloud_fuse()
                fuser.ensure_invoke_garbage_collect()
                self.fuser = fuser


class SplitBySplitSampler(Sampler):
    # only used in training mode.
    # prefer to use PrepareData(), but this class has already been used for a
    # while and is working great. New approaches can leverage PrepareData, but
    # at this moment, it is ok not to re-factor it
    def __init__(self, dataset, group_size=1, shuffle=True,
                 fixed_samples_in_node=False,
                 random_seed=9,
                 prepare_t_versions=[],
                 disable_prepare=None,
                 ):
        from utils.common import print_frame_info
        print_frame_info()
        self.dataset = dataset
        self.group_size = group_size
        self.random_seed = random_seed
        self.shuffle = shuffle

        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.local_size = get_local_size()

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
            self._idx_split.share_memory_()
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

        idx_split = torch.cat(
            [idx_split[idx_split[:, 1] == p] for p in priority])

        num_idx_on_node = (
            len(idx_split) + self.node_size - 1) // self.node_size
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
                    logging.info(
                        'time to get group index on node: {}'.format(cost))
                    if self.local_rank == 0:
                        self.prepare(new_g['split_in_group'])
                    group_buffers.append(new_g)
                r = group_buffers[0]['idx_in_group'][idx]
                yield r
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

            # if os.environ.get('QD_TSV_USE_FUSE') and \
            #         int(os.environ['QD_TSV_USE_FUSE']) and \
            #         not self.fixed_samples_in_node:
            #     from src.utils.cloud_storage import create_cloud_fuse
            #     fuser = create_cloud_fuse()
            #     fuser.ensure_invoke_garbage_collect()
            #     self.fuser = fuser

    def prepare(self, split):
        if self.disable_prepare:
            return
        self.ensure_init_prepare()
        q = self.prepare_queue
        size = q.qsize()
        if size > 100:
            logging.info('prepare queue is too long {}'.format(size))
        q.put([ps[split] for ps in self.prepare_files])

    def __len__(self):
        raise ValueError('should not be called')


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
        # if hasattr(self.batch_sampler, 'skip') and self.curr_iter > 0:
        #     logging.info('we will skip {} batches'.format(self.curr_iter))
        #     self.batch_sampler.skip(self.curr_iter)
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
        curr_rank = get_rank()
        world_size = get_world_size()
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


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(
            self, dataset, num_replicas=None, rank=None, shuffle=True,
            length_divisible=1):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(
            len(self.dataset) * 1.0 / self.num_replicas))
        if length_divisible > 1:
            import logging
            logging.info(
                'before making divisible = {}'.format(self.num_samples))
            self.num_samples = (
                (self.num_samples + length_divisible - 1) //
                length_divisible) * length_divisible
            logging.info('adjust to = {}'.format(self.num_samples))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        assert (self.total_size - len(indices)) <= len(indices), 'not implemented'
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
