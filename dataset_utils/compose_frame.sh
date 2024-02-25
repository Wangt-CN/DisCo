start_number=$1
end_number=$2
image_path=$3
output_gif=$4
video_name=$5

# list_file="mylist.txt"
# rm -f $list_file # 如果文件已经存在，先删除
# for i in $(seq -f "%04g" $start_number $end_number); do
#     echo "$image_path/${video_name}_$i.png" >> $list_file
# done

ffmpeg -framerate 30 -f concat -safe 0 -i $list_file -vsync vfr $output_gif

# rm $list_file

# chmod +x create_gif.sh 

# bash dataset_utils/compose_frame.sh 0 185 testdata/caixukun_cALL/caixukun_c01/inference output_video.gif cai_c01