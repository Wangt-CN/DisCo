import os
import shutil
import argparse
import os
import shutil
def remakeFileTree(base_dir):
    startname = os.path.basename(base_dir).split('_')[0]
    refer_control_dir = os.path.join(base_dir, 'ref_control')
    cond_dir = os.path.join(base_dir, 'cond')
    video_name = os.path.basename(base_dir)
    print(startname)

    c_numbers = set()
    for sub_folder in [refer_control_dir, cond_dir]:
        if os.path.exists(sub_folder):
            for file_name in os.listdir(sub_folder):
                if file_name.startswith(startname) and 'c0' in file_name and file_name.endswith('.png'):
                    c_number = file_name.split('_')[2]
                    c_numbers.add(c_number)

    for c_number in c_numbers:
        c_folder = video_name.replace("cALL", c_number) #'gKR_sBM_{c_number}_d28_mKR1_ch07_copy' 
        for sub_folder in ['ref_control', 'cond']:
            target_sub_dir = os.path.join(base_dir, c_folder, sub_folder)
            os.makedirs(target_sub_dir, exist_ok=True)
            source_sub_dir = os.path.join(base_dir, sub_folder)
            if os.path.exists(source_sub_dir):
                for file_name in os.listdir(source_sub_dir):
                    if file_name.startswith(startname) and c_number in file_name and file_name.endswith('.png'):
                        shutil.move(
                            os.path.join(source_sub_dir, file_name),
                            target_sub_dir
                        )
    shutil.rmtree(refer_control_dir)
    shutil.rmtree(cond_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type=str, required=True)
    args = parser.parse_args()
    remakeFileTree(args.parent_dir)