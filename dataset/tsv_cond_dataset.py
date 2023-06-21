from .tsv_dataset import TsvCompositeDataset
import json
import random


class TsvCondImgCompositeDataset(TsvCompositeDataset):
    def __init__(self, args, yaml_file,
                 split="train", size_frame=1, tokzr=None):
        super().__init__(args, yaml_file, split, size_frame, tokzr)

        # by defualt we store a vid2line file
        self.vid2line_file = self.cfg.get('vid2line', None) # get_row_from_tsv
        self.vid2line_tsv = self.get_tsv_file(self.vid2line_file)
        for cond in self.args.conds:
            # poses
            setattr(self, f"{cond}_file", self.cfg.get(cond, None))
            setattr(self, f"{cond}_tsv", self.get_tsv_file(getattr(self,f"{cond}_file")))
    
    def get_current_video_start_end(self, img_idx):
        row = self.get_row_from_tsv(self.vid2line_tsv, img_idx)
        image_key = row[0]
        if self.is_composite:
            assert image_key in self.image_keys[img_idx] # keli: ugly fix
        else:
            assert image_key == self.image_keys[img_idx]
        start_end = json.loads(row[1])
        return start_end

    def get_reference_frame_idx(self, img_idx):
        start_end = self.get_current_video_start_end(img_idx)
        start_end = [item+img_idx for item in start_end]
        try:
            if self.args.ref_mode == "first" or self.split != 'train':
            # if self.args.ref_mode == "first":
                return start_end[0]
            elif self.args.ref_mode  == "random":
                return random.randint(start_end[0], start_end[1])
            elif self.args.ref_mode  == "random_sparse":
                return random.randrange(start_end[0], start_end[1], 30)
            elif self.args.ref_mode  == "random_sparse_part": # 20% random
                if random.random() < 0.2:
                    return random.randrange(start_end[0], start_end[1], 30)
                else:
                    return start_end[0]
            # elif self.args.ref_mode == "random_sparse_filter":
            #     resample_flag = True
            #     ref_img_idx = random.randrange(start_end[0], start_end[1], 30)
            #     while resample_flag:
            #         ref_skeleton = self.get_cond(img_idx, 'poses')
            #         if len(ref_skeleton) > 10:
            #             resample_flag = False
            #         else:
            #             ref_img_idx = random.randrange(start_end[0], start_end[1], 30)
            else:
                raise NotImplementedError(f"Unknown ref_mode {self.args.ref_mode}")
        except:
            return start_end[0]
    
    def get_cond(self, img_idx, cond):
        cond_tsv = getattr(self,f"{cond}_tsv")
        row = self.get_row_from_tsv(cond_tsv, img_idx)
        if len(row) == 3:
            image_key, buf, valid = row
            # assert image_key == self.image_keys[img_idx]
            if not valid:
                return None
            else:
                return self.str2img(buf)
        else:
            return self.str2img(row[1])
            