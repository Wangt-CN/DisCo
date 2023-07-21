import cv2
import numpy as np
import os
from tqdm import tqdm
# Define video settings


folder_name = ''
save_folder_name = ''

sub_folder_name_list = os.listdir(folder_name)

for idx, sub_folder_name in enumerate(sub_folder_name_list):
    print(sub_folder_name)
    # video_img_source_path = os.path.join(folder_name, sub_folder_name, 'eval_step_7080', 'pred_gs3.0_scale-cond1.0-ref1.0')
    video_img_source_path = os.path.join(folder_name, sub_folder_name, 'eval_step_7200', 'pred_gs3.0_scale-cond1.0-ref1.0')
    # video_img_source_path = os.path.join(folder_name, sub_folder_name, 'eval_step_17790', 'pred_gs3.0_scale-cond1.0-ref1.0')
    if save_folder_name is None:
        save_path = video_img_source_path
    else: # use specific save folder
        save_path = os.path.join(save_folder_name, sub_folder_name, 'pred_gs3.0_scale-cond1.0-ref1.0')
        os.makedirs(save_path, exist_ok=True)

    # if exists
    video_save_path = "{}_vid_output.mp4".format(save_path)
    if os.path.isfile(video_save_path) and os.path.getsize(video_save_path)>10000:
        print('has done, skip this')
        continue

    fps = 30
    size = (256, 256)
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    # Get a list of all files in the folder
    folder_path = video_img_source_path
    try:
        files = os.listdir(folder_path)
    except:
        print('load frame wrong, skip this')
        continue

    # Initialize video writer
    video_writer = cv2.VideoWriter(video_save_path, codec, fps, size)
    # Filter the list to only include image filesd
    image_files = [file for file in files if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
    num_frm = len(image_files)

    for image_fname_ in image_files:
        image_fname = '{}/{}'.format(folder_path,image_fname_)
        frame = cv2.imread(image_fname)
        # Write frames to video
        video_writer.write(frame)

    # Release resources
    video_writer.release()
