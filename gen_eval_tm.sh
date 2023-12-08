exp_folder=$1
pred_folder="${2:-${exp_folder}/pred_gs3.0_scale-cond1.0-ref1.0}"
gt_folder=${3:-${exp_folder}/gt}

echo ${pred_folder}
echo ${gt_folder}

# merge frames
if [ -d "${pred_folder}_frames" ]; then
  rm -r "${pred_folder}_frames"
fi
if [ -d "${gt_folder}_frames" ]; then
  rm -r "${gt_folder}_frames"
fi
python tool/merge_subfolder.py -i ${pred_folder} -o ${pred_folder}_frames
python tool/merge_subfolder.py -i ${gt_folder} -o ${gt_folder}_frames

# Pytorch-FID
python -m pytorch_fid ${pred_folder}_frames ${gt_folder}_frames --device cuda:0 > ${exp_folder}/pytorch_fid.txt
cat ${exp_folder}/pytorch_fid.txt

# L1 SSIM LPIPS and PSNR
python tool/metrics/metric_center.py --root_dir blob_dir/debug_output/video_sythesis --path_gen ${pred_folder}_frames --path_gt ${gt_folder}_frames --type l1 ssim lpips psnr clean-fid --write_metric_to ${exp_folder}/metrics_l1_ssim_lpips_psnr.json

# generate MP4s
python tool/video/yz_gen_gifs_for_fvd_subfolders.py -i ${pred_folder} -o ${pred_folder}_16framemp4 --fps 3 --format mp4
python tool/video/yz_gen_gifs_for_fvd_subfolders.py -i ${gt_folder} -o ${gt_folder}_16framemp4 --fps 3 --format mp4

# FVD eval
python tool/metrics/metric_center.py --root_dir blob_dir/debug_output/video_sythesis --path_gen ${pred_folder}_16framemp4 --path_gt ${gt_folder}_16framemp4 --type fid-vid fvd dtssd --write_metric_to ${exp_folder}/metrics_fid-vid_fvd_dtssd_16frames.json --number_sample_frames 16 --sample_duration 16

# generate videos
python tool/video/yz_gen_vid_subfolders.py -i ${pred_folder} --interval 16