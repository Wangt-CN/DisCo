"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

from utils.lib import *
import torch
import torch.distributed as dist
import subprocess as sp
from .logger import LOGGER


def iter_tqdm(item):
    if is_main_process():
        return tqdm(item, ascii=True)
    else:
        return item


def dist_init(args, distributed=True):
    if distributed:
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            master_addr = os.environ.get("MASTER_ADDR", 'localhost')
            master_port = os.environ.get("MASTER_PORT", 12475)
            master_uri = f"tcp://{master_addr}:{master_port}" #if master_addr else 'localhost'
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            args.num_gpus = world_size
            args.word_size = args.num_gpus
            args.distributed = True
            args.local_rank = local_rank
            args.rank = world_rank
            
            args.num_nodes = world_size // args.local_size  
            args.node_id = int(args.rank / args.local_size)
            if args.distributed:
                LOGGER.info(
                    f"Init distributed training on "
                    f"local rank {args.local_rank}, "
                    f"global rank {world_rank}")
                torch.cuda.set_device(args.local_rank)
                dist.init_process_group(
                    backend='nccl',
                    init_method=master_uri,
                    world_size=world_size,
                    rank=world_rank,
                    timeout=timedelta(hours=5),  # 5 hrs
                )
                synchronize()
        elif 'WORLD_SIZE' in os.environ:
            args.num_gpus = int(
                os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
            args.word_size = args.num_gpus
            local_rank = int(os.environ['LOCAL_RANK'])
            rank = int(os.environ['RANK'])
            args.distributed = True  # args.num_gpus > 1
            args.local_rank = local_rank
            args.rank = rank
            args.num_nodes = world_size // args.local_size  # hardcoded
            args.node_id = int(args.rank / args.local_size)
            if args.distributed:
                LOGGER.info(
                    f"Init distributed training on "
                    f"local rank {args.local_rank}")
                torch.cuda.set_device(args.local_rank)
                dist.init_process_group(
                    backend='nccl', init_method='env://',
                    timeout=timedelta(hours=5),  # 5 hrs
                )
                synchronize()
        else:
            print("distributed training ... presumbly debug with 1 GPU")
            args.num_gpus = 1
            args.distributed = True
            args.num_nodes = 1
            args.word_size = args.num_gpus
            args.local_rank = 0
            args.rank = 0
            args.num_nodes = 1
            args.node_id = 0
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            master_addr = os.environ.get("MASTER_ADDR", 'localhost')
            master_port = os.environ.get("MASTER_PORT", 12475)
            master_uri = f"tcp://{master_addr}:{master_port}" #if master_addr else 'localhost'
            LOGGER.info(
                f"Init distributed training on "
                f"local rank {args.local_rank}")
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(
                backend='nccl', 
                init_method=master_uri,
                world_size=args.word_size,
                rank=args.rank,
                timeout=timedelta(hours=5),  # 5 hrs
            )
            # raise ValueError(
            #     "Unable to init torch.distributed. Did not find WORLD_SIZE or OMPI_COMM_WORLD_SIZE in os.environ")
    else:
        print("no distributed training ...")
        # no distributed training
        args.num_gpus = torch.cuda.device_count()
        args.word_size = args.num_gpus
        args.num_nodes = 1
        args.distributed = False
    # Setting seed
    set_seed(args)


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def validate_seed(seed):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            print(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                print(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
    return seed


def set_seed(args):
    seed, n_gpu = args.seed, args.num_gpus
    if args.do_train and args.fix_dist_seed:
        this_seed = validate_seed((args.rank+1)* seed)
    else:
        this_seed = validate_seed(seed)
    print(f"Using seed {this_seed} for rank {args.rank}")
    random.seed(this_seed)
    np.random.seed(this_seed)
    torch.random.manual_seed(this_seed)
    # torch.cuda.manual_seed(this_seed)
    if args.rank == 0:
        print(f"Using seed {this_seed} for torch.cuda")
        torch.cuda.manual_seed_all(this_seed)


def get_world_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))


def get_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))


def get_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))


def get_local_size():
    if 'LOCAL_SIZE' in os.environ:
        return int(os.environ['LOCAL_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))


def is_main_process():
    if get_rank()==0:
        try:
            if torch.distributed.get_rank()==0:
                return True
            else:
                return False
        except RuntimeError:
            return True
    else:
        return False

# def is_main_process():
#     return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    t = torch.randn((), device='cuda')
    dist.all_reduce(t)
    torch.cuda.synchronize()
    return
    # dist.barrier()


def gather_on_master(data):
    """Same as all_gather, but gathers data on master process only, using CPU.
    Thus, this does not work with NCCL backend unless they add CPU support.

    The memory consumption of this function is ~ 3x of data size. While in
    principal, it should be ~2x, it's not easy to force Python to release
    memory immediately and thus, peak memory usage could be up to 3x.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    # trying to optimize memory, but in fact,
    # it's not guaranteed to be released
    del data
    storage = torch.ByteStorage.from_buffer(buffer)
    del buffer
    tensor = torch.ByteTensor(storage)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()])
    size_list = [torch.LongTensor([0]) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,))
        tensor = torch.cat((tensor, padding), dim=0)
        del padding

    if is_main_process():
        tensor_list = []
        for _ in size_list:
            tensor_list.append(torch.ByteTensor(size=(max_size,)))
        dist.gather(tensor, gather_list=tensor_list, dst=0)
        del tensor
    else:
        dist.gather(tensor, gather_list=[], dst=0)
        del tensor
        return

    data_list = []
    for tensor in tensor_list:
        buffer = tensor.cpu().numpy().tobytes()
        del tensor
        data_list.append(pickle.loads(buffer))
        del buffer

    return data_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes
    so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


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
            ):
    if not silent:
        logging.info(
            'start to cmd run: {}'.format(' '.join(map(str, list_cmd))))
        if working_dir:
            logging.info(working_dir)
    # if we dont' set stdin as sp.PIPE, it will complain the stdin is not a tty
    # device. Maybe, the reson is it is inside another process.
    # if stdout=sp.PIPE, it will not print the result in the screen
    e = os.environ.copy()
    if 'SSH_AUTH_SOCK' in e:
        del e['SSH_AUTH_SOCK']
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
    if env:
        for k in env:
            e[k] = env[k]
    if dry_run:
        # we need the log result. Thus, we do not return at teh very beginning
        return
    if not return_output:
        # if env is None:
        #     p = sp.Popen(list_cmd, stdin=sp.PIPE, cwd=working_dir)
        # else:
        p = sp.Popen(' '.join(list_cmd) if shell else list_cmd,
                     stdin=stdin,
                     env=e,
                     shell=shell,
                     stdout=stdout,
                     cwd=working_dir)
        message = p.communicate(input=process_input)
        if p.returncode != 0:
            raise ValueError(message)
        return message
    else:
        if shell:
            message = sp.check_output(
                ' '.join(list_cmd),
                env=e,
                cwd=working_dir,
                shell=True)
        else:
            message = sp.check_output(list_cmd,
                                      env=e,
                                      cwd=working_dir,
                                      )
        if not silent:
            logging.info('finished the cmd run')
        return decode_to_str(message)


def broadcast(data, src=0):
    output = dist.broadcast(data, src=src)
    return output
