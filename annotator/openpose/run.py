import argparse
import glob
import json
import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)


import torch
import code
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from tqdm import tqdm
import util
import model
from body import Body


openpose2coco_order = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def output_json(candidate, subset, fname='out.json'):
    num_skeleton = subset.shape[0]
    skeleton_results = []
    for i in range(num_skeleton):
        skeleton = []
        for j in openpose2coco_order:
            if int(subset[i][j])==-1:
                x, y, confidence = -1, -1, -1
            else:
                x, y, confidence, index = candidate[int(subset[i][j])]
            skeleton.append([int(x),int(y),confidence])
        skeleton_results.append({'keypoints':skeleton})
    with open(fname, 'w') as f:
        json.dump(skeleton_results, f)
    return skeleton_results
    
def inference(body_estimation, test_image, image_out_name, json_out_name):
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    cv2.imwrite(image_out_name,canvas)
    skeleton_results = output_json(candidate, subset, fname=json_out_name)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/blob/linjli/debug_output/video_sythesis/dataset/toy_dataset', help='dataset root')
    parser.add_argument('--checkpoint_path', type=str, default='body_pose_model.pth') # wholebody
    
    args = parser.parse_args()
    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    print('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    return args

def get_vid_folders(args):
    vid_folders = os.listdir(args.dataset_root)
    return vid_folders

def get_img_list(args, vid_folders, i):
    vid_name = vid_folders[i]
    folder_path = os.path.join(args.dataset_root, vid_name)
    files = os.listdir(folder_path)
    # Filter the list to only include image files
    image_files = [file for file in files if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
    return folder_path, image_files

def main():
    args = cli()
    print('LOAD: OpenPose Model')
    body_estimation = Body(args.checkpoint_path)

    todo_list = get_vid_folders(args)
    print('total videos:', len(todo_list))

    for index in range(len(todo_list)):
        print('processing video ID: {:05d}'.format(index))
        try:
            dir_path, img_list = get_img_list(args, todo_list, index)

            json_out_folder_name = '{}/openpose_json'.format(dir_path)
            os.makedirs(json_out_folder_name, exist_ok=True)
            image_out_folder_name = '{}/openpose_vis'.format(dir_path)
            os.makedirs(image_out_folder_name, exist_ok=True)

            for item in tqdm(img_list):
                filename = item
                image_path = os.path.join(dir_path, filename)
                # Add the new folder to the directory path
                new_dir_path = os.path.join(dir_path, 'openpose_json')
                # Combine the new directory path with the filename to form the new file path
                json_out_name = os.path.join(new_dir_path, filename) + '.json'
                # Add the new folder to the directory path
                new_dir_path = os.path.join(dir_path, 'openpose_vis')
                # Combine the new directory path with the filename to form the new file path
                image_out_name = os.path.join(new_dir_path, filename) + '.jpg'

                if not os.path.isfile(json_out_name):
                    inference(body_estimation, image_path, image_out_name, json_out_name)
                else:
                    print('Found file exist:', json_out_name)

        except Exception as e: 
            print(e)
            pass

if __name__ == '__main__':
    main()

