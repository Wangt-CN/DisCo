# genenrate gt with (512,512) 

### for environment without ffemge
# video name: gKR_sBM_cAll_d28_mKR1_ch07
video=$1
name=$( echo "$video" | cut -d. -f1 |  rev | cut -d'/' -f1 | rev)
folder_path="video/${name}"
output_folder="frame/${video}_inference"
bg_folder="frame/${video}_bg"
if [ -d "$output_folder" ]; then
    rm -r "$output_folder"
fi
mkdir -p "$output_folder"

for video_file in "$folder_path"/*.mp4; do
    video_name=$(basename -s .mp4 "$video_file")
    ffmpeg -i "$video_file" -vf "fps=25" -q:v 1 -start_number 0 ${output_folder}/"$video_name"_%04d.png
done

### bounding box 
python dataset_utils/groundino_demo.py  --IMAGE_PATH ${output_folder}

### seg bg sequence
python ./annotator/grounded-sam/run_single.py --todo_folder_list ${output_folder} --savepath ${bg_folder}

### mv video
testdata_dir="testdata/${video}"
testdata_dir_cond="testdata/${video}/ref_control"

if [ -d "$testdata_dir_cond" ]; then
    rm -r "$testdata_dir_cond"
fi
mkdir -p "$testdata_dir_cond"
mv ${bg_folder}/* ${testdata_dir_cond}/

#"testdata/caixukun_cALL"
# python dataset_utils/remakeFileTree.py --parent_dir ${testdata_dir} 
