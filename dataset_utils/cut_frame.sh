# genenrate gt with (512,512) 
# source ~/.profile
# conda activate sd 
video_name=$1
name=$( echo "$video_name" | cut -d. -f1 |  rev | cut -d'/' -f1 | rev)
folder_path="video/frame/${name}"
# if [ -d "$folder_path" ]; then
#     rm -r "$folder_path"
# fi
# mkdir -p "$folder_path"
# echo ${folder_path}
# ffmpeg -i vAlkaid123456ideo/${video_name} -vf "fps=30" -q:v 1 -start_number 0 ${folder_path}/"$name"_%04d.png

# python dataset_utils/groundino_demo.py  --IMAGE_PATH ${folder_path}

python ./annotator/grounded-sam/run_single.py --todo_folder_list ${folder_path}