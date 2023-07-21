export PYTHONUNBUFFERED=TRUE
python tool/video/gen_vid_folder_forvis.py --start_idx 0 --end_idx 200 --gen_idx 1 > tool/video/runsh/cond_log11.txt &
python tool/video/gen_vid_folder_forvis.py --start_idx 200 --end_idx 400 --gen_idx 1 > tool/video/runsh/cond_log12.txt &
python tool/video/gen_vid_folder_forvis.py --start_idx 400 --end_idx 600 --gen_idx 1 > tool/video/runsh/cond_log13.txt &
#python tool/video/gen_vid_folder_forvis.py --start_idx 600 --end_idx 800 > tool/video/runsh/cond_log4.txt &
#python tool/video/gen_vid_folder_forvis.py --start_idx 800 --end_idx 1000 > tool/video/runsh/cond_log5.txt &
#python tool/video/gen_vid_folder_forvis.py --start_idx 1000 --end_idx 1200 > tool/video/runsh/cond_log6.txt &
#python tool/video/gen_vid_folder_forvis.py --start_idx 1200 --end_idx 1400 > tool/video/runsh/cond_log7.txt &
#python tool/video/gen_vid_folder_forvis.py --start_idx 1400 --end_idx 1700 > tool/video/runsh/cond_log8.txt &

python tool/video/gen_vid_folder_forvis.py --start_idx 0 --end_idx 200 --gen_idx 0 > tool/video/runsh/log11.txt &
python tool/video/gen_vid_folder_forvis.py --start_idx 200 --end_idx 400 --gen_idx 0 > tool/video/runsh/log12.txt &
python tool/video/gen_vid_folder_forvis.py --start_idx 400 --end_idx 600 --gen_idx 0 > tool/video/runsh/log13.txt &