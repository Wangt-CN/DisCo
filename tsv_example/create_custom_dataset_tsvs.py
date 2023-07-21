import torch
from collections import defaultdict
import sys
import os
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
from utils.common import encoded_from_img
from utils.tsv_io import tsv_writer
from PIL import Image
import json, cv2, math, yaml
import numpy as np
import argparse

def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')

def load_image(path):
    if os.path.exists(path):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        image = None
    return image

def load_video_paths(args):
    folder_path = args.root_folder
    vid_list = os.listdir(folder_path)
    video_paths = [ folder_path + '/' + vid_name for vid_name in vid_list]
    video_paths.sort()
    return video_paths

def load_frame_paths(video_folder):
    frm_list = os.listdir(video_folder)
    frame_paths = [ video_folder + '/' + frm_name for frm_name in frm_list if 'png' in frm_name or 'jpg' in frm_name]
    frame_paths.sort()
    return frame_paths

def get_frm_list(args):
    video_paths = load_video_paths(args)
    all_frm_paths = []
    for vid in video_paths:
        all_frm_paths += load_frame_paths(vid)
    return all_frm_paths


def main(args):
    split = args.split 
    image_path_list = get_frm_list(args)

    if args.debug:
        image_path_list = image_path_list[:10]

    ###############################################################################
    # process images.tsv
    print('generating images.tsv')
    def gen_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            video_key = image_path.split("/")[-2]
            image_key = f"Custom_{video_key}_{image_name}"
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            row = [image_key, encoded_from_img(image)]
            yield(row)
    tsv_writer(gen_row(image_path_list), f"{args.output_folder}/{split}_images.tsv")

    ###############################################################################
    # process vid2line.tsv
    print('generating vid2line.tsv')
    video2line = defaultdict(list)
    for i, image_path in enumerate(image_path_list):
        image_name = image_path.split("/")[-1]
        video_key = image_path.split("/")[-2]
        video2line[video_key].append(i)
    vid2st_ed_line = {key: [min(video2line[key]), max(video2line[key])] for key in video2line}
    def gen_vis2line_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            video_key = image_path.split("/")[-2]
            image_key = f"Custom_{video_key}_{image_name}"
            # image = load_image(f"./tiktok_datasets/{image_path}")
            # if image is None:
            #     print(f"image does not exists: {image_path}")
            #     continue
            row = [image_key, vid2st_ed_line[video_key]]
            yield(row)
    tsv_writer(gen_vis2line_row(image_path_list), f"{args.output_folder}/{split}_vid2line.tsv")

    ###############################################################################
    # process pose.tsv
    print('generating poses.tsv')
    # draw the body keypoint and lims
    def draw_bodypose(canvas, coco_keypoints):
        pose = [
                coco_keypoints[0], # Nose (0)
                list((np.asarray(coco_keypoints[5]) + np.asarray(coco_keypoints[6]))/2), # Neck (1)
                coco_keypoints[6], # RShoulder (2)
                coco_keypoints[8], # RElbow (3)
                coco_keypoints[10], # RWrist (4)
                coco_keypoints[5], # LShoulder (5)
                coco_keypoints[7], # LElbow (6)
                coco_keypoints[9], # LWrist (7)
                coco_keypoints[12], # RHip (8)
                coco_keypoints[14], # RKnee (9)
                coco_keypoints[16], # RAnkle (10)
                coco_keypoints[11], # LHip (11)
                coco_keypoints[13], # LKnee (12)
                coco_keypoints[15], # LAnkle (13)
                coco_keypoints[2], # REye (14)
                coco_keypoints[1], # LEye (15)
                coco_keypoints[4], # REar (16)
                coco_keypoints[3], # LEar (17)
            ]  # openpose_keypoints

        stickwidth = 4
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        canvas = np.zeros_like(canvas)

        for i in range(18):
            x, y = pose[i][0:2]
            if x>=0 and y>=0: 
                cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
                # cv2.putText(canvas, '%d'%(i), (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for limb_idx in range(17):
            cur_canvas = canvas.copy()
            index_a = limbSeq[limb_idx][0]-1
            index_b = limbSeq[limb_idx][1]-1

            if pose[index_a][0]<0 or pose[index_b][0]<0 or pose[index_a][1]<0 or pose[index_b][1]<0:
                continue

            Y = [pose[index_a][0], pose[index_b][0]]
            X = [pose[index_a][1], pose[index_b][1]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limb_idx])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        # Convert color space from BGR to RGB
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        # Create PIL image object from numpy array
        canvas = Image.fromarray(canvas)
        return canvas

    def gen_row_pose(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            video_key = image_path.split("/")[-2]
            image_key = f"Custom_{video_key}_{image_name}"
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            # anno_pose_path = "./tiktok_datasets/" + image_path.replace("/images/", "/openpose_json/")+".json"
            anno_pose_path = "{}/openpose_json/{}.json".format(('/').join(image_path.split("/")[0:-1]),image_name)
            pose_without_visibletag = []
            f = open(anno_pose_path,'r')
            d = json.load(f)
            # if there is a valid openpose skeleton, load it
            if len(d)>0:
                for j in range(17):
                    x = d[0]['keypoints'][j][0]
                    y = d[0]['keypoints'][j][1]
                    pose_without_visibletag.append([x,y])
            else: # if there is not valid openpose skeleton, add a dummy one
                for j in range(17):
                    x = -1
                    y = -1
                    pose_without_visibletag.append([x,y])      
            pose_image = draw_bodypose(image, pose_without_visibletag)
            row = [image_key, encoded_from_img(pose_image)]
            yield(row)

    tsv_writer(gen_row_pose(image_path_list), f"{args.output_folder}/{split}_poses.tsv")

    ###############################################################################
    # process mask.tsv
    print('generating masks.tsv')
    def load_groundedsam_mask(path):
        # print(path)
        try:
            if os.path.exists(path):
                img = load_image(path)
                img = np.asarray(img)
                bk = img[:,:,:]==[68,0,83]
                fg = (bk==False)
                fg = fg*255.0
                mask = fg.astype(np.uint8)
                ipl_mask = Image.fromarray(mask)
            else:
                ipl_mask = None

        except Exception as e: 
            print(e)
            ipl_mask = None

        return ipl_mask

    def gen_row_mask(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            video_key = image_path.split("/")[-2]
            image_key = f"Custom_{video_key}_{image_name}"
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            anno_mask_path = "{}/groundsam/{}.mask.jpg".format(('/').join(image_path.split("/")[0:-1]),image_name)
            mask_image = load_groundedsam_mask(anno_mask_path)
            mask_image = mask_image.resize(image.size)

            valid = True
            if mask_image is None:
                valid = False
                mask_image = np.zeros_like(image)
       
            row = [image_key, encoded_from_img(mask_image), valid]
            yield(row)

    tsv_writer(gen_row_mask(image_path_list), f"{args.output_folder}/{split}_masks.tsv")

    ###############################################################################
    # process yaml file

    all_field = ['img', 'masks', 'poses', 'vid2line']
    out_cfg = {'composite': False}

    for field in all_field:
        tsvfilename = f'{args.split}_{field}.tsv'
        out_cfg[field] = tsvfilename

    write_to_yaml_file(out_cfg, os.path.join(args.output_folder, args.split + '.yaml'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder",
                        type=str, default='./tiktok_datasets')
    parser.add_argument("--output_folder",
                        type=str, default='./blob_dir/debug_output/video_sythesis/dataset')
    parser.add_argument("--split",
                        type=str, default='train')
    parser.add_argument("--debug",
                        type=bool, default=False)
    args = parser.parse_args()
    main(args)
