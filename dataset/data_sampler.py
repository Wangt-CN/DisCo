import math
import random
from datetime import datetime
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
from utils.dist import get_local_rank, get_local_size, get_rank, get_world_size
from .data_utils.sampler_utils import PrepareData
import logging
logger = logging.getLogger(__name__)


class DistributedSamplerLimited(Sampler):
    def __init__(self, dataset: Dataset, num_replicas: int = None,
                 rank: int = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, limited=-1) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        print(f'Dbg: distribeted sampler limited: rank={rank}, num_replicas={num_replicas}')
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.limited = limited
        if self.limited > -1:
            self.num_samples = min(self.limited, self.num_samples)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.limited > -1 and len(indices) > self.limited:
            print(f'Trim indices: {len(indices)} --> {self.limited}')
            indices = indices[:self.limited]
        assert len(indices) == self.num_samples
        # shuffle subsample
        if self.shuffle:  # and self.epoch > 0:
            # random.seed(self.seed + self.epoch)
            random.seed(datetime.now())
            random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = 0  # keep data unchanged
        # self.epoch = epoch


class NodeSplitSampler(Sampler):
    def __init__(self, dataset, shuffle, random_seed,
                 first_epoch_skip_shuffle=False,
                 prepare_data=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.world_size = get_world_size()
        self.local_size = get_local_size()
        self.node_size = self.world_size // self.local_size
        self.rank = get_rank()
        self.node_idx = self.rank // self.local_size
        self.local_rank = get_local_rank()
        self.next_epoch_skip_shuffle = first_epoch_skip_shuffle

        # only be used when shuffle = True and first_epoch_skip_shuffle = True
        self.prepare_data = prepare_data
        self.prepare = None
        self.skip = 0

    def get_index_on_node(self):
        # there is no need to cache source_list as we only call this function
        # once in the whole training life-time
        source_list = self.dataset.get_composite_source_idx()
        idx_split = list(enumerate(source_list))
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
        unique_split_index = list(
            set(idx_split[offset:offset_end, 1].tolist()))
        logger.info('calling get_index_on_node in NodeSplitSampler ....')
        logger.info(f'dataset is {self.dataset.yaml_file}')
        logger.info(unique_split_index)
        if self.shuffle and self.next_epoch_skip_shuffle and self.prepare_data:
            if get_local_rank() == 0:
                self.prepare = PrepareData(
                    self.dataset,
                    prepare_t_versions=[],
                    fixed_samples_in_node=True)
                for s in unique_split_index:
                    self.prepare.prepare(s)
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
                curr_idx_on_node = idx_on_node[
                    self.get_shufle_idx(len(idx_on_node))]
            else:
                curr_idx_on_node = idx_on_node
                self.next_epoch_skip_shuffle = False
        else:
            curr_idx_on_node = idx_on_node
        idx_rank_size = (
            len(curr_idx_on_node) + self.local_size - 1) // self.local_size
        offset = idx_rank_size * self.local_rank
        offset_end = offset + idx_rank_size
        offset_end = min(offset_end, len(curr_idx_on_node))
        curr_idx_on_node = curr_idx_on_node.tolist()
        for i in range(offset, offset_end):
            yield curr_idx_on_node[i]

    def skip(self, num):
        self.skip = num

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
        raise ValueError('should not be called')


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

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
