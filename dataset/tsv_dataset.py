from utils.lib import *
from .data_utils.video_transforms import (
    Normalize, Resize, CenterCrop,  ClipToTensor,
    RandomCrop, Compose)
from .data_utils.albef_randaug import RandomAugment
# from utils.tsv_file import TSVFile, CompositeTSVFile
from utils.tsv_io import tsv_reader
from utils.tsv_io import (
    TSVFile, load_list_file, load_from_yaml_file)
from utils.load_files import (
    find_file_path_in_yaml, load_box_linelist_file)
from utils.logger import LOGGER
from utils.dist import get_world_size, get_rank
from .data_sampler import (
    DistributedSamplerLimited, NodeSplitSampler, IterationBasedBatchSampler)
from .data_utils.node_sampler import ScaleNodeSplitSampler
try:
    from azfuse import File
except ImportError:
    print("azfuse is not installed")


class Dataset_Base(T.utils.data.Dataset):
    def __init__(self, args, split="train", size_frame=4, tokzr=None):
        super().__init__()
        self.args = args
        self.size_frame = size_frame
        self.split = split
        
        self.tokzr = tokzr
        # if tokzr is not None:
        #     self.tokzr = tokzr
        # else:
        #     self.tokzr = transformers.AutoTokenizer.from_pretrained(
        #         self.args.tokenizer)
        if self.tokzr is not None:
            (self.cls_token_id, self.sep_token_id,
                self.pad_token_id, self.mask_token_id,
                self.unk_token_id) = self.tokzr.convert_tokens_to_ids(
                [self.tokzr.cls_token,
                    self.tokzr.sep_token, self.tokzr.pad_token,
                    self.tokzr.mask_token,
                    self.tokzr.unk_token])
            self.true_token_id = self.tokzr.convert_tokens_to_ids(
                ["true"])[0]
            self.false_token_id = self.tokzr.convert_tokens_to_ids(
                ["false"])[0]

    def read_tsv(self, worker_id):
        assert hasattr(self, 'img_tsv_path')
        self.img = open(self.img_tsv_path, 'r')

    def seek_img_tsv(self, pos):
        self.img.seek(pos)
        return [s.strip() for s in self.img.readline().split('\t')]

    def get_partial_data(self):
        if self.split != 'train' or self.args.data_ratio == 1:
            return
        assert self.args.data_ratio > 0
        self.video2txt = defaultdict(list)
        for item in self.txt:
            self.video2txt[item["video"]].append(item)
        vids = list(self.video2txt.keys())
        random.shuffle(vids)
        if self.args.data_ratio < 1:
            n_partial_vids = math.ceil(len(vids)*self.args.data_ratio)
        else:
            n_partial_vids = min(
                int(self.args.data_ratio), len(vids))
        partial_vids = vids[:n_partial_vids]
        partial_txt = []
        for vid in partial_vids:
            partial_txt.extend(self.video2txt[vid])
        self.txt = partial_txt

    def concat_txt(self, txt_a, txt_b):
        self.sep_token = self.tokzr.sep_token
        return txt_a + f" {self.sep_token} " + txt_b

    def pad_resize(self, img):
        w, h = img.size
        img = TV.transforms.Compose([
            TV.transforms.Pad([0, (w-h)//2] if w > h else [(h-w)//2, 0]),
            TV.transforms.Resize(
                [self.args.size_img, self.args.size_img]),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ])(img)
        return img

    def img_center_crop(self, img):
        img = TV.transforms.Compose([
            TV.transforms.Resize(self.args.size_img),
            TV.transforms.CenterCrop(
                (self.args.size_img, self.args.size_img)),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])(img)
        return img

    def vid_center_crop(self, img):
        img = Compose([
                Resize(self.args.size_img),
                CenterCrop(
                    (self.args.size_img, self.args.size_img)),
                ClipToTensor(channel_nb=3),
                Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])(img)
        img = img.permute(1, 0, 2, 3)
        return img

    def vid_rand_crop(self, img):
        assert self.split == "train"
        img = Compose([
            Resize(self.args.size_img),
            RandomCrop(
                (self.args.size_img, self.args.size_img)),
            ClipToTensor(channel_nb=3),
            Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])(img)
        # adapt from torch_videovision:
        # https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W)
        #   in the range [0, 1.0]
        # (C x T x H x W) --> (T x C x H x W)
        img = img.permute(1, 0, 2, 3)
        return img

    def img_rand_crop(self, img):
        assert self.split == "train"
        img = TV.transforms.Compose([
            TV.transforms.Resize(self.args.size_img),
            TV.transforms.RandomCrop(
                (self.args.size_img, self.args.size_img)),
            RandomAugment(
                2, 5, isPIL=True, augs=[
                    'Identity', 'AutoContrast', 'Equalize',
                    'Brightness', 'Sharpness', 'ShearX',
                    'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])(img)
        return img

    def str2img(self, b):
        try:
            img = Image.fromarray(
                cv2.imdecode(
                    np.frombuffer(base64.b64decode(b), np.uint8),
                    cv2.IMREAD_COLOR)[:, :, ::-1]
                ).convert('RGB')
        except Exception:
            img = Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')
        return img

    def sampling(self, start, end, n):
        if n == 1:
            return [int(round((start+end)/2.))]
        if n < 1:
            raise Exception("behaviour not defined for n<2")
        step = (end-start)/float(n-1)
        return [int(round(start+x*step)) for x in range(n)]

    def temporal_sample(self, list_of_b, random_sample=False):
        max_size_frame = len(list_of_b)
        if max_size_frame == 1 or self.size_frame == max_size_frame:
            return list_of_b
        if max_size_frame < self.size_frame:
            print(f"Error in size_frame",
                  f"\tasked for {size_frame} from {max_size_frame} frames")

        size_frame = min(self.size_frame, max_size_frame)
        size_clips = int(math.ceil(max_size_frame / size_frame))
        if random_sample:
            sampled_start = random.choice(range(size_clips))
            sampled_end = min(
                sampled_start + (size_frame - 1) * size_clips,
                max_size_frame - 1)
        else:
            sampled_start = 0
            sampled_end = max_size_frame - 1
        sampled_index = self.sampling(sampled_start, sampled_end, size_frame)
        sampled_video = [list_of_b[i] for i in sampled_index]
        return sampled_video

    def get_img_or_video_w_transform(self, list_of_b):
        bufs = self.temporal_sample(
            list_of_b, random_sample=(self.split == 'train'))
        img = []
        for b in bufs:
            single_img = self.str2img(b)
            if self.split == "train":
                vis_transform = random.choice(self.args.img_transform)
                if vis_transform == "vid_rand_crop":
                    img.append(single_img)
                else:
                    if vis_transform == "pad_resize":
                        single_img = self.pad_resize(single_img)
                    elif vis_transform == "img_center_crop":
                        single_img = self.img_center_crop(single_img)
                    else:
                        single_img = self.img_rand_crop(single_img)
                    img.append(single_img.unsqueeze(0))
            else:
                if self.args.img_transform == ["vid_rand_crop"]:
                    vis_transform = "vid_center_crop"
                    img.append(single_img)
                else:
                    if self.args.img_transform == ["pad_resize"]:
                        vis_transform = "pad_resize"
                        single_img = self.pad_resize(single_img)
                    else:
                        vis_transform = "img_center_crop"
                        single_img = self.img_center_crop(single_img)
                    img.append(single_img.unsqueeze(0))

        if vis_transform == "vid_rand_crop":
            img = self.vid_rand_crop(img)
        elif vis_transform == "vid_center_crop":
            img = self.vid_center_crop(img)
        else:
            img = T.cat(img, dim=0)

        return img

    def str2txt(self, s):
        # if version.parse(transformers.__version__) >= version.parse("4.16.1"):
        #     txt = self.tokzr.encode(s)
        #     old_len = len(txt)
        #     txt = txt[:self.args.size_txt-1]
        #     new_len = len(txt)
        #     if new_len < old_len:
        #         txt = txt + [self.sep_token_id]
        #     padding_len = self.args.size_txt-len(txt)
        #     txt = txt + [self.pad_token_id]*(padding_len)
        # else:
        assert self.tokzr is not None
        txt = self.tokzr.encode(
            s, padding='max_length', max_length=self.args.size_txt,
            truncation=True)
        mask = [1 if w != self.pad_token_id else 0 for w in txt]
        mask = T.LongTensor(mask)
        txt = T.LongTensor(txt)
        assert len(txt[txt == self.sep_token_id]) == 1, f'{txt}'
        return txt, mask


def get_dl(ds, args, worker_init_fn=None, collate_fn=None):
    if args.distributed:
        sp = T.utils.data.distributed.DistributedSampler(
            ds, shuffle=(ds.split == 'train'))
    else:
        if ds.split == 'train':
            sp = T.utils.data.RandomSampler(ds)
        else:
            sp = T.utils.data.SequentialSampler(ds)
    # if ds.split=='train':
    #     sp.set_epoch(ep)
    dl = T.utils.data.DataLoader(
        ds, batch_size=args.size_batch, num_workers=args.n_workers,
        pin_memory=True, sampler=sp, worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)
    return dl


def get_tsv_dls(args, DataCls, tokzr=None):
    if tokzr is None:
        tokzr = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer)
    img_path = f'{args.data_dir}/img_{args.dataset}.tsv'
    LOGGER.info(f"rank {get_rank()}: loading video frames from {img_path}")
    lineidx_data = pickle.load(open(
        f'{args.data_dir}/img_{args.dataset}.id2lineidx.pkl', 'rb'))
    txt_path = f'{args.data_dir}/txt_{args.task}.json'
    LOGGER.info(f"rank {get_rank()}: loading text from {txt_path}")
    txt_data = json.load(open(txt_path, 'r'))
    splits = ['train', 'val']
    if 'test' in txt_data:
        splits.append('test')

    ds_all = {
        split: DataCls(
            args, img_path, txt_data, lineidx_data, split,
            tokzr=tokzr)
        for split in splits}
    log_data_len = f"data_ratio: {args.data_ratio}"
    for split in splits:
        log_data_len += f", {split}: {len(ds_all[split])}"
    LOGGER.info(log_data_len)

    dl_all = {
        split:
        get_dl(
            ds, args,
            worker_init_fn=ds.read_tsv if hasattr(ds, 'read_tsv') else None,
            collate_fn=ds.collate_batch if hasattr(ds, 'collate_batch') else None)
        for split, ds in ds_all.items()}
    dl_tr, dl_vl = [
        dl_all[split] for split in ["train", "val"]]
    dl_ts = dl_all["test"] if "test" in dl_all else None
    return dl_tr, dl_vl, dl_ts


def move_to_cuda(batch):
    if isinstance(batch, T.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


class CompositeTSVFile(object):
    def __init__(self, list_file, seq_file, cache_policy=False,
                 hold_buffer=0, data_dir=None,
                 ):
        # list_file can be a loaded or constructed pair of index, rather than a
        # filename to load. In this case, seq_file will be a list of dataset,
        # which should implement len() and __getitem__() so that we can
        # reference it.
        self.seq_file = seq_file
        # self.file_list = list_file
        if isinstance(list_file, str):
            self.file_list = load_list_file(list_file)
        else:
            assert isinstance(list_file, list)
            self.file_list = list_file
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
            self.file_list
        )

    def get_row_len(self, i):
        self.ensure_initialized()
        idx_source, idx_row, _ = map(int, self.seq[i])
        result = self.tsvs[idx_source].get_row_len(idx_row)
        return result

    def get_key(self, index):
        # added by Linjie
        self.ensure_initialized()
        idx_source, idx_row, _ = map(int, self.seq[index])
        k = self.tsvs[idx_source].get_key(idx_row)
        return '_'.join([self.file_list[idx_source], k])

    def __getitem__(self, index):
        self.ensure_initialized()
        idx_source, idx_row, _ = map(int, self.seq[index])
        start = time.time()
        result = self.tsvs[idx_source].seek(idx_row)
        end = time.time()
        if end - start > 10:
            import logging
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
        idx_source, idx_row, _ = map(int, self.seq[index])
        return self.tsvs[idx_source].seek_first_column(idx_row)

    def get_composite_source_idx(self):
        return [int(i) for i, _, _ in self.seq]

    def is_from_valid_file(self, idx=None):
        self.ensure_initialized()
        if idx is None:
            return [self.tsvs[int(idx_source)] is not None for idx_source, _, _ in tqdm(self.seq)]
        else:
            return self.tsvs[int(self.seq[idx][0])] is not None

    def ensure_initialized(self):
        if self.seq is None:
            print("CompositeTSVFile: initializing")
            # if isinstance(self.file_list, str) and \
            #         isinstance(self.seq_file, str):
            self.seq = TSVFile(self.seq_file)
            tsv_path = [f  if op.exists(f) else op.join(self.data_dir, f) for f in self.file_list]
            print(f"CompositeTSVFile: {tsv_path}")
            # if self.data_dir is not None:
            self.tsvs = [TSVFile(f, self.cache_policy) if f != 'd' else None for f in tsv_path]
            print("CompositeTSVFile: initialized")
            # else:
            #     self.tsvs = [TSVFile(f, self.cache_policy) if f != 'd' else None for f in self.file_list]
            # else:
            #     self.seq = self.file_list
            #     self.tsvs = self.seq_file


class TsvCompositeDataset(Dataset_Base):
    def __init__(self, args, yaml_file,
                 split="train", size_frame=4, tokzr=None):
        super().__init__(args, split, size_frame, tokzr)
        # yaml_file = op.join(args.data_dir, yaml_file)
        # LOGGER.info(f'yaml_file:{yaml_file}')
        if not op.isfile(yaml_file):
            yaml_file = op.join(args.data_dir, yaml_file)
            assert op.isfile(yaml_file), f"{yaml_file} does not exists"
        try:
            self.cfg = load_from_yaml_file(yaml_file)
        except Exception as e:
            print(f"{yaml_file} does not exists")

        self.yaml_file = yaml_file
        self.root = op.dirname(yaml_file)

        self.is_composite = self.cfg.get('composite', False)
        self.cap_linelist_file = find_file_path_in_yaml(
            self.cfg.get('caption_linelist', None), self.root)
        self.visual_file = self.cfg.get('img', None)
        self.visual_tsv = self.get_tsv_file(self.visual_file)

        self.label_file = self.cfg.get('label', None)
        self.label_tsv = self.get_tsv_file(self.label_file)

        self.cap_file = self.cfg.get('caption', None)
        self.cap_tsv = self.get_tsv_file(self.cap_file)
        if self.is_composite:
            assert op.isfile(self.cap_linelist_file)
            self.cap_line_list = [
                int(row[2]) for row in tsv_reader(self.cap_linelist_file)]
            self.img_line_list = [i for i in range(len(self.cap_line_list))]
        elif self.cap_linelist_file:
            line_list = load_box_linelist_file(self.cap_linelist_file)
            self.img_line_list = line_list[0]
            self.cap_line_list = line_list[1]
        else:
            # one caption per image/video
            self.img_line_list = [i for i in range(self.cap_tsv.num_rows())]
            self.cap_line_list = [0 for i in range(self.cap_tsv.num_rows())]
        self.is_train = split == "train"
        if self.is_train:
            assert self.cap_tsv is not None
            # assert self.tokzr is not None
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        # self.img_res = self.args.size_img
        # self.patch_size = getattr(args, 'patch_size', 32)

        self.use_asr = getattr(args, 'use_asr', False)
        # for MERLOT/HT100M only
        self.append_pred_mf_cap = getattr(args, 'append_pred_mf_cap', False)
        self.pred_mf_cap_only = getattr(args, 'pred_mf_cap_only', False)
        self.alternate_asr_pred_cap = getattr(
            args, 'alternate_asr_pred_cap', False)
        self.alternate_asr_pred_cap = (
            self.alternate_asr_pred_cap and self.use_asr
            and self.pred_mf_cap_only)
        LOGGER.info(f'Use_asr: {self.use_asr}')
        self.on_memory = getattr(args, 'on_memory', False)

    def get_partial_data(self):
        if self.split != 'train' or self.args.data_ratio == 1:
            return
        assert self.args.data_ratio > 0
        list_of_idx = list(range(len(self.img_line_list)))
        random.shuffle(list_of_idx)
        if self.args.data_ratio < 1:
            num_samples = math.ceil(len(list_of_idx) * self.args.data_ratio)
        else:
            num_samples = min(
                int(self.args.data_ratio), len(list_of_idx))
        sampled_idx = list_of_idx[:num_samples]
        img_line_list = [self.img_line_list[idx] for idx in sampled_idx]
        cap_line_list = [self.cap_line_list[idx] for idx in sampled_idx]
        self.img_line_list = img_line_list
        self.cap_line_list = cap_line_list
        return

    def __len__(self):
        return len(self.img_line_list)

    def __cap_len__(self):
        return len(self.cap_line_list)

    def get_composite_source_idx(self):
        if self.is_composite:
            # assert op.isfile(self.cap_linelist_file)
            # self.composite_source_idx = [
            #     int(row[0]) for row in tsv_reader(self.cap_linelist_file)]
            self.composite_source_idx = self.cap_tsv.get_composite_source_idx()
        else:
            # only a single tsv file is used as input
            self.composite_source_idx = [
                0 for _ in range(len(self.cap_line_list))]
        return self.composite_source_idx

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(
                    tsv_file, self.cap_linelist_file, data_dir=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def load_caption_to_memory(self):
        self.caption_on_memory = {}
        for img_idx in set(self.img_line_list):
            row = self.get_row_from_tsv(self.cap_tsv, img_idx)
            for cap_idx, data in enumerate(json.loads(row[1])):
                self.caption_on_memory[(img_idx, cap_idx)] = data['caption']

    def get_valid_tsv(self):
        if self.is_train:
            return self.cap_tsv
        # sorted by file size
        if self.cap_tsv:
            return self.cap_tsv
        if self.visual_tsv:
            return self.visual_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i): i for i in range(tsv.num_rows())}

    def get_image_cap_index(self, idx):
        return self.img_line_list[idx], self.cap_line_list[idx]

    def get_row_from_tsv(self, tsv, img_idx):
        row = tsv[img_idx]
        if self.is_composite:
            # try:
            assert self.image_keys[img_idx].endswith(row[0])
            # except:
            #     print(img_idx)
            #     print(self.image_keys[img_idx])
            #     print(row[0])
        else:
            assert row[0] == self.image_keys[img_idx]
        return row

    def get_caption(self, img_idx, cap_idx):
        if self.is_train:
            if self.on_memory:
                return self.caption_on_memory[(img_idx, cap_idx)]
            row = self.get_row_from_tsv(self.cap_tsv, img_idx)
            return json.loads(row[1])[cap_idx]['caption']
        return ""

    def get_merlot_caption_asr(self, data_sample):
        try:
            if self.pred_mf_cap_only:
                caption = data_sample['pred_cap_mf15'][0]
            else:
                caption = data_sample['captions'][0]
                if self.append_pred_mf_cap:
                    caption += ' [SEP] ' + data_sample['pred_cap_mf15'][0]
            if 'noise_asr' in data_sample:
                asr = data_sample['noise_asr'][0]
            else:
                asr = data_sample['captions'][0]
            if self.alternate_asr_pred_cap:
                p = random.random()
                if p > 0.5:
                    return asr, caption, '', ''
            return caption, asr, '', ''
        except Exception:
            # FIXME: quick hack for youtube-short-videos
            return data_sample['caption'], '', '', ''

    def get_caption_and_timeinfo(self, data, cap_idx):
        caption, tag, start, end = '', ' ', None, None
        data_sample = data[cap_idx]
        if self.is_train:
            caption = data_sample['caption']
            if 'start' in data_sample:
                start = data_sample['start']
            if 'end' in data_sample:
                end = data_sample['end']
            if 'label' in data_sample and self.use_action_label:
                tag += data_sample['label']
            if 'asr' in data_sample and self.use_asr:
                asr = data_sample['asr']  # .lower()
                tag = asr
        else:
            if 'start' in data_sample:
                start = data_sample['start']
            if 'end' in data_sample:
                end = data_sample['end']
            if 'label' in data_sample and self.use_action_label:
                tag += data_sample['label']
            if 'asr' in data_sample and self.use_asr:
                asr = data_sample['asr']  # .lower()
                tag = asr
            if 'caption' in data_sample:
                caption = data_sample['caption']
        return caption, tag, start, end

    def get_caption_and_timeinfo_wrapper(self, img_idx, cap_idx):
        row = self.get_row_from_tsv(self.cap_tsv, img_idx)
        data_sample = json.loads(row[1])
        is_merlot = False
        if type(data_sample) is dict:
            # for merlot tsv format
            is_merlot = True
            caption, asr_or_tag, start, end = self.get_merlot_caption_asr(
                data_sample)
        else:
            # for other datasets (VATEX, MSRVTT, TVC, YouCook,
            # COCO, GoogleCC+SBU+COCO)
            caption, asr_or_tag, start, end = self.get_caption_and_timeinfo(
                data_sample, cap_idx)
        return caption, asr_or_tag, start, end, is_merlot

    def get_caption_file_in_coco_format(self):
        # for evaluation
        cap_file_coco_format = find_file_path_in_yaml(
            self.cfg.get('caption_coco_format', None), self.root)
        if cap_file_coco_format:
            return cap_file_coco_format
        test_split = op.basename(self.yaml_file).split('.')[0]
        return op.join(self.root, test_split + '_caption_coco_format.json')

    def get_captions_by_key(self, key):
        # get a list of captions for image (by key)
        img_idx = self.key2index[key]
        cap_info = json.loads(self.cap_tsv[img_idx][1])
        return [c['caption'] for c in cap_info]

    def get_video_key(self, idx):
        return self.get_row_from_tsv(self.label_tsv, idx)[0]

    def get_visual_data(self, idx, is_MERLOT=False):
        row = self.get_row_from_tsv(self.visual_tsv, idx)
        if row[0] == row[-1]:
            # if the input is a video tsv, on the fly decoding
            # return self.decode_and_get_frames(row[-1], start, end), True
            raise NotImplementedError("On the fly decoding is not supported")
        elif is_MERLOT or len(row) >= self.size_frame + 2:
            # return self.get_frames_from_tsv(row[2:]), True
            return self.get_img_or_video(row[2:]), True
        else:  # if the input is a image tsv, return image numpy array
            return self.get_img_or_video([row[-1]]), False

    def get_img_txt_pair(self, idx):
        img_idx, cap_idx = self.get_image_cap_index(idx)
        img_key = self.image_keys[img_idx]
        (caption_sample, tag, start,
         end, is_MERLOT) = self.get_caption_and_timeinfo_wrapper(
            img_idx, cap_idx)
        # get image or video frames
        # frames: (T, C, H, W),  is_video: binary tag
        frames, is_video = self.get_visual_data(
            img_idx, is_MERLOT)

        if isinstance(caption_sample, dict):
            caption = caption_sample["caption"]
        else:
            caption = caption_sample
            caption_sample = None

        # preparing outputs
        meta_data = {}
        meta_data['caption'] = caption  # raw text data, not tokenized
        meta_data['img_key'] = img_key
        meta_data['is_video'] = (
            is_video and len(frames) > 1
            )  # True: video data, False: image data
        meta_data['tag'] = tag
        meta_data['img'] = frames
        return meta_data


def make_batch_data_sampler(
        sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = T.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(
        dataset, shuffle, distributed, random_seed, limited_samples=-1):
    is_train = dataset.split == 'train'
    if distributed:
        enable_node_split_sampler = dataset.is_composite and getattr(dataset.args, "node_split_sampler", False) 
        enable_node_split_sampler = enable_node_split_sampler and is_train
        if enable_node_split_sampler:
            # first_epoch_skip_shuffle not working yet
            print(
                "Enable NodeSplitSampler with first_epoch_skip_shuffle=True")
            return NodeSplitSampler(
                dataset, shuffle=shuffle, random_seed=random_seed,
                first_epoch_skip_shuffle=True)
        elif limited_samples < 1:
            return T.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle)
        else:  # use limited distributed sampler
            return DistributedSamplerLimited(
                dataset, shuffle=shuffle, limited=limited_samples)
    if shuffle:
        sampler = T.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = T.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(
        args, dataset, ep=0):
    is_train = dataset.split == 'train'
    collate_fn = dataset.collate_batch
    is_distributed = args.distributed
    num_gpus = args.num_gpus
    if is_train:
        shuffle = True
        images_per_gpu = min(
            args.size_batch * (args.size_frame // dataset.size_frame),
            128)
        images_per_batch = images_per_gpu * get_world_size()
        iter_per_ep = len(dataset) // images_per_batch
        num_iters = iter_per_ep * args.size_epoch
        # num_iters = iter_per_ep
        start_iter = 0
    else:
        shuffle = False
        images_per_gpu = args.size_batch * (
            args.size_frame // dataset.size_frame)
        images_per_batch = images_per_gpu * get_world_size()
        iter_per_ep = None
        num_iters = None
        start_iter = 0

    if hasattr(args, 'limited_samples'):
        limited_samples = args.limited_samples // num_gpus
    else:
        limited_samples = -1
    random_seed = args.seed
    sampler = make_data_sampler(
        dataset, shuffle, is_distributed, limited_samples=limited_samples,
        random_seed=random_seed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )

    # if is_train:
    #     batch_sampler.set_epoch(ep)
    data_loader = T.utils.data.DataLoader(
        dataset, num_workers=args.n_workers, batch_sampler=batch_sampler,
        pin_memory=True, collate_fn=collate_fn
    )
    meta_info = (images_per_batch, iter_per_ep, num_iters)
    return data_loader, meta_info


class MetaLoader(object):
    """ wraps multiple data loaders """
    def __init__(self, loaders, accum_steps=1, distributed=False):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, T.utils.data.DataLoader):
                r = 1
            else:
                raise ValueError()
            assert isinstance(r, int)
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n]*r)

        self.accum_steps = accum_steps
        self.distributed = distributed
        self.step = 0

    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
                if self.distributed:
                    # make sure all process is training same task
                    objects = [task]
                    DIST.broadcast_object_list(
                        objects, src=0)
                    task = objects[0]
                    # task = any_broadcast(task, 0)
            self.step += 1
            # print(f'calling iter for {task}')
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch
