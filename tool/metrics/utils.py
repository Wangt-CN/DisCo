import re
import os
from pathlib import Path
from random import gauss
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter, ImageSequence
import cv2
import torch.nn.functional as F
import ffmpeg
import random
import imageio

from tool.metrics.resize import build_resizer, make_resizer

def check_Image(path):
    try:
        img_pil = Image.open(path).convert('RGB')
        return True
    except:
        print(f"{path} can not open, delete")
        return False
        
class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, size=(299, 299), fn_resize=None):
        self.files = files
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fn_resize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil)

        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


class ResizeDatasetBlur(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, size=(299, 299), fn_resize=None, is_blur=False, gaussian_kernel=5):
        self.files = files
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fn_resize
        self.is_blur = is_blur
        self.gaussian_kernel = gaussian_kernel

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):

        path = str(self.files[i])
        if ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert('RGB')
            # if self.is_blur:
            #     img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=self.gaussian_kernel))
            img_np = np.array(img_pil)

        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np).astype(np.float32)
        # if self.is_blur:
        #     img_resized = cv2.GaussianBlur(img_resized, (self.gaussian_kernel, self.gaussian_kernel), 0)
        
        if self.is_blur:
            # if self.gaussian_kernel % 2 == 0:
            img_resized = Image.fromarray(img_resized.astype(np.uint8))
            img_resized = img_resized.filter(ImageFilter.GaussianBlur(radius=self.gaussian_kernel))
            img_resized = np.array(img_resized).astype(np.float32)
            # else:
            #     img_resized = cv2.GaussianBlur(img_resized, (self.gaussian_kernel, self.gaussian_kernel), 0)

        img_t = self.transforms(img_resized)

        return img_t


class ResizeDatasetInceptionScoreBlur(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, size=(299, 299), fn_resize=None, is_blur=False, gaussian_kernel=5):
        self.files = files
   
        self.transform = torchvision.transforms.ToTensor()
        self.pixel_mean = torch.as_tensor(np.array([0.485, 0.456, 0.406])).unsqueeze(1).unsqueeze(1) * 255.0  # 3*1*1
        self.pixel_std = torch.as_tensor(np.array([0.229, 0.224, 0.225])).unsqueeze(1).unsqueeze(1)*255.0  # 3*1*1
    
        self.size = size
        self.fn_resize = fn_resize
        self.is_blur = is_blur
        self.gaussian_kernel = gaussian_kernel

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil)

        img_resized = self.fn_resize(img_np).astype(np.float32)
        if self.is_blur:
            # if self.gaussian_kernel % 2 == 0:
            img_resized = Image.fromarray(img_resized.astype(np.uint8))
            img_resized = img_resized.filter(ImageFilter.GaussianBlur(radius=self.gaussian_kernel))
            img_resized = np.array(img_resized).astype(np.float32)
            # else:
            #     img_resized = cv2.GaussianBlur(img_resized, (self.gaussian_kernel, self.gaussian_kernel), 0)
        img_t = self.transform(img_resized)  # 3*h*w because the float32 format, the value are still in 0-255
        img_t = (img_t - self.pixel_mean) / self.pixel_std
        img_t = img_t.float()

        return img_t


class ResizeDatasetFIDBlur(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, size=(299, 299), fn_resize=None, is_blur=False, gaussian_kernel=5):
        self.files = files
   
        self.transform = torchvision.transforms.ToTensor()
        self.pixel_mean = torch.as_tensor(np.array([[0.485, 0.456, 0.406]])).unsqueeze(1).unsqueeze(1) * 255.0  # 3*1*1
        self.pixel_std = torch.as_tensor(np.array([0.229, 0.224, 0.225])).unsqueeze(1).unsqueeze(1)*255.0  # 3*1*1
    
        self.size = size
        self.fn_resize = fn_resize
        self.is_blur = is_blur
        self.gaussian_kernel = gaussian_kernel

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil)

        img_resized = self.fn_resize(img_np).astype(np.float32)
        if self.is_blur:
            # if self.gaussian_kernel % 2 == 0:
            img_resized = Image.fromarray(img_resized.astype(np.uint8))
            img_resized = img_resized.filter(ImageFilter.GaussianBlur(radius=self.gaussian_kernel))
            img_resized = np.array(img_resized).astype(np.float32)
            # else:
            #     img_resized = cv2.GaussianBlur(img_resized, (self.gaussian_kernel, self.gaussian_kernel), 0)

        img_t = self.transform(img_resized)  # 3*h*w because the float32 format, the value are still in 0-255
        img_t = (img_t - self.pixel_mean) / self.pixel_std

        return img_t


def resize_numpyimg(x):
    x = Image.fromarray(x)
    x = x.resize((64, 64))
    x = np.asarray(x).astype(np.uint8)
    return x


class ResizeDatasetFIDVideoBlur(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, size=(299, 299), fn_resize=None, is_blur=False, gaussian_kernel=5, is_sample_frames=False, sample_frames=10):
        self.files = files
   
        self.transform = torchvision.transforms.ToTensor()
        self.pixel_mean = torch.as_tensor(np.array([[0.485, 0.456, 0.406]])).unsqueeze(1).unsqueeze(1) * 255.0  # 3*1*1
        self.pixel_std = torch.as_tensor(np.array([0.229, 0.224, 0.225])).unsqueeze(1).unsqueeze(1)*255.0  # 3*1*1
    
        self.size = size
        self.fn_resize = fn_resize
        self.is_blur = is_blur
        self.gaussian_kernel = gaussian_kernel
        self.sample_frames = sample_frames
        self.is_sample_frames = is_sample_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            path = str(self.files[i])
            probe = ffmpeg.probe(path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            out, _ = (ffmpeg.input(path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True))
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            num_frame = video.shape[0]
            frame_ids = list(range(num_frame))

            if self.is_sample_frames and self.sample_frames < num_frame:
                # frame_ids = random.choice(frame_ids, self.sample_frames)
                print(path, num_frame, self.sample_frames)
                frame_ids = frame_ids[:self.sample_frames]

            # TODO resize img
            # video_new = []
            # for fid in frame_ids:
            #     video_new.append(resize_numpyimg(video[fid]))
            # video = np.stack(video_new)


            img_tensor = []
            for fid in frame_ids:
                img_np = video[fid]
                img_resized = self.fn_resize(img_np).astype(np.float32)
                if self.is_blur:
                    if self.gaussian_kernel % 2 == 0:
                        img_resized = Image.fromarray(img_resized)
                        img_resized = img_resized.filter(ImageFilter.GaussianBlur(radius=self.gaussian_kernel))
                        img_resized = np.array(img_resized).astype(np.float32)
                    else:
                        img_resized = cv2.GaussianBlur(img_resized, (self.gaussian_kernel, self.gaussian_kernel), 0)

                img_t = self.transform(img_resized)  # 3*h*w because the float32 format, the value are still in 0-255
                img_tensor.append(img_t)

            img_tensor = torch.stack(img_tensor, dim=0)  # N*3*H*W
            return img_tensor
        except Exception as e:
            print(f'index {i} skipped because {e}')
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


class DatasetFVDVideo(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, sample_duration=16, mode='FVD-3DRN50', img_size=112):
        self.files = files

        self.pixel_mean = torch.as_tensor(np.array([114.7748, 107.7354, 99.4750]))
        self.img_size = img_size
        self.sample_duration = sample_duration
        self.mode = mode
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):

        path = str(self.files[i])
        out, _ = (ffmpeg.input(path).filter('scale', self.img_size, self.img_size).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True))
        video = np.frombuffer(out, np.uint8).reshape([-1, self.img_size, self.img_size, 3])  # t*h*w*3
        video = torch.as_tensor(video.copy()).float()
        num_v = video.shape[0]

        if num_v % self.sample_duration != 0:
            num_v_ag = self.sample_duration - num_v % self.sample_duration
            video_aug = video[[-1], :, :, :].repeat(num_v_ag, 1, 1, 1)
            video = torch.cat([video, video_aug], dim=0)
            num_seg = num_v // self.sample_duration + 1
        else:
            num_seg = num_v // self.sample_duration
        
        if self.mode == 'FVD-3DRN50':
            video = video - self.pixel_mean
        elif self.mode == "FVD-3DInception":
            video = video / 127.5 - 1
        
        video = video.view(num_seg, self.sample_duration, self.img_size, self.img_size, 3).contiguous().permute(0, 4, 1, 2, 3).float()  # num_seg, 3, sample_during h, w
    
        return video


def save_video_from_numpy_array(array, output_filename, fps=3):
    assert array.dtype == 'uint8', "The input array must be of dtype uint8"

    with imageio.get_writer(output_filename, fps=fps) as writer:
        for frame in array:
            writer.append_data(frame)


def gif_to_nparray(gif_path):
    gif = Image.open(gif_path)
    frames = [np.array(frame.copy().convert('RGB'), dtype=np.uint8) for frame in ImageSequence.Iterator(gif)]
    video = np.stack(frames)
    return video


class DatasetFVDVideoResize(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, sample_duration=16, mode='FVD-3DRN50', img_size=112):
        self.files = files

        self.pixel_mean = torch.as_tensor(np.array([114.7748, 107.7354, 99.4750]))
        self.img_size = img_size
        self.sample_duration = sample_duration
        self.mode = mode
        self.resize_func = make_resizer("PIL", False, "bicubic", (img_size, img_size))
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            path = str(self.files[i])

            if Path(path).suffix == '.gif':
                video = gif_to_nparray(path)
            else:
                probe = ffmpeg.probe(path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                out, _ = (ffmpeg.input(path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True))
                video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

            #if video.shape[0] != 10:
            #    print('not 10, but', video.shape[0])
            #    return self.__getitem__(0)

            # TODO resize img
            # frame_ids = video.shape[0]
            # video_new = []
            # for fid in range(frame_ids):
            #     video_new.append(resize_numpyimg(video[fid]))
            # video = np.stack(video_new)


            video_resize = []
            for vim in video:
                vim_resize = self.resize_func(vim)
                video_resize.append(vim_resize)

            video = np.stack(video_resize, axis=0)
            # import ipdb; ipdb.set_trace()
            video = torch.as_tensor(video.copy()).float()
            num_v = video.shape[0]

            if num_v % self.sample_duration != 0 and self.mode == "FVD-3DRN50":
                num_v_ag = self.sample_duration - num_v % self.sample_duration
                video_aug = video[[-1], :, :, :].repeat(num_v_ag, 1, 1, 1)
                video = torch.cat([video, video_aug], dim=0)
                num_seg = num_v // self.sample_duration + 1
            else:
                num_seg = num_v // self.sample_duration

            if self.mode == 'FVD-3DRN50':
                video = video - self.pixel_mean
                video = video.view(num_seg, self.sample_duration, self.img_size, self.img_size, 3).contiguous().permute(0, 4, 1, 2, 3).float()
            elif self.mode == "FVD-3DInception" or self.mode == 'MAE':
                video = video / 127.5 - 1
                video = video.unsqueeze(0).permute(0, 4, 1, 2, 3).float()  # num_seg, 3, sample_during h, w

            return video
        except Exception as e:
            print(f'{i} skipped beacase {e}')
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


class DatasetFVDVideoFromFramesResize(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all frame files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, sample_duration=16, mode='FVD-3DRN50', img_size=112):
        frame_format1 = r"^(TiktokDance_\d+_)(\d+)(\D*\.\w+)$"
        frame_format2 = r"^(TiktokDance_\d+_\d+_1x1_)(\d+)(\D*\.\w+)$"

        # self.files = files
        files = sorted(files)
        self.video_frames = {}
        for file in files:
            file_name = Path(file).name
            file_parent = Path(file).parent
            if re.match(frame_format1, file_name):
                folder_format_re = frame_format1
            elif re.match(frame_format2, file_name):
                folder_format_re = frame_format2
            else:
                print(f"Frame name '{file_name}' does not match any format")
                continue
        
            match = re.match(folder_format_re, file_name)
            frame_index = int(match.group(2))

            self.video_frames[file_name] = [file]
            for i in range(1, sample_duration):
                next_frame_file_name = match.group(1) + str(frame_index + i).zfill(len(match.group(2))) + match.group(3)
                next_frame_file_path = Path(file_parent, next_frame_file_name)
                if not next_frame_file_path.exists():
                    del self.video_frames[file_name]
                    break
                self.video_frames[file_name].append(next_frame_file_path.as_posix())

        self.pixel_mean = torch.as_tensor(np.array([114.7748, 107.7354, 99.4750]))
        self.img_size = img_size
        self.sample_duration = sample_duration
        self.mode = mode
        self.resize_func = make_resizer("PIL", False, "bicubic", (img_size, img_size))
 
    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, i):
        try:
            video_name = list(self.video_frames.keys())[i]
            frame_list = []
            for frame_path in self.video_frames[video_name]:
                frame = Image.open(frame_path).convert('RGB')
                frame_list.append(np.array(frame))
            video = np.stack(frame_list, axis=0)

            video_resize = []
            for vim in video:
                vim_resize = self.resize_func(vim)
                video_resize.append(vim_resize)

            video = np.stack(video_resize, axis=0)
            # import ipdb; ipdb.set_trace()
            video = torch.as_tensor(video.copy()).float()
            num_v = video.shape[0]

            if num_v % self.sample_duration != 0 and self.mode == "FVD-3DRN50":
                num_v_ag = self.sample_duration - num_v % self.sample_duration
                video_aug = video[[-1], :, :, :].repeat(num_v_ag, 1, 1, 1)
                video = torch.cat([video, video_aug], dim=0)
                num_seg = num_v // self.sample_duration + 1
            else:
                num_seg = num_v // self.sample_duration

            if self.mode == 'FVD-3DRN50':
                video = video - self.pixel_mean
                video = video.view(num_seg, self.sample_duration, self.img_size, self.img_size, 3).contiguous().permute(0, 4, 1, 2, 3).float()
            elif self.mode == "FVD-3DInception" or self.mode == 'MAE':
                video = video / 127.5 - 1
                video = video.unsqueeze(0).permute(0, 4, 1, 2, 3).float()  # num_seg, 3, sample_during h, w

            return video
        except Exception as e:
            print(f'{i} skipped beacase {e}')
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy'}
