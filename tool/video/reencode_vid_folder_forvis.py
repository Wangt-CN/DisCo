
import os


folder_name = ''
save_folder_name = ''

sub_folder_name_list = os.listdir(folder_name)
sub_folder_name_list = [file for file in sub_folder_name_list if not 'json' in file]
sub_folder_name_list.sort()

for idx, sub_folder_name in enumerate(sub_folder_name_list):
    print(sub_folder_name)

    video_source_path = os.path.join(folder_name, sub_folder_name, 'cond_vid_output.mp4')
    video_target_path = os.path.join(folder_name, sub_folder_name, 'cond_vid_output_reencode.mp4')

    assert os.path.exists(video_source_path)

    if os.path.exists(video_target_path):
        print('remove the previous file')
        os.remove(video_target_path)
    os.system(f'ffmpeg -i {video_source_path} -c:v libx264 -c:a aac -strict -2 {video_target_path}')



