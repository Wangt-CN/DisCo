exp_folder=$1
pred_folder="${2:-${exp_folder}/pred_gs3.0_scale-cond1.0-ref1.0}"
gt_folder=${3:-${exp_folder}/gt}

echo ${pred_folder}
echo ${gt_folder}
# # L1 SSIM LPIPS and PSNR
python  tool/metrics/metric_center.py --root_dir ./blob_dir/debug_output/video_sythesis --path_gen ${pred_folder}/ --path_gt ${gt_folder} --type l1 ssim lpips psnr  clean-fid --write_metric_to ${exp_folder}/metrics_l1_ssim_lpips_psnr.json

# Pytorch-FID
python -m pytorch_fid ${pred_folder}/ ${gt_folder} --device cuda:0

# DEPRECITED due to unstable gif generation code
#  Generate GIFs of 16 frames, 3 fps
# multi process first 
# python  tool/video/gen_gifs_for_fvd.py --root_dir ./blob_dir/debug_output/video_sythesis --path_gen ${pred_folder}/ --path_gt ${gt_folder} --gif_frames 16 --gif_fps 3 --num_workers 4
# single process to get the missing ones
# python  tool/video/gen_gifs_for_fvd.py --root_dir ./blob_dir/debug_output/video_sythesis --path_gen ${pred_folder}/ --path_gt ${gt_folder} --gif_frames 16 --gif_fps 3 --num_workers 1

#  FVD eval
# the root dir should be the dir containing the fvd pretrain model (resnet-50-kinetics and i3d)
python  tool/metrics/metric_center.py --root_dir /home/kevintw/code/disco --path_gen ${pred_folder}/ --path_gt ${gt_folder}/ --type fid-vid fvd  --write_metric_to ${exp_folder}/metrics_fid-vid_fvd.json --number_sample_frames 16 --sample_duration 16
