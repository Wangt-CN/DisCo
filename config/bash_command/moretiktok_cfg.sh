## visualization command for more tiktok data (cfg)
## you need to revise the path in command to your own path
export WANDB_ENABLE=0
NCCL_ASYNC_ERROR_HANDLING=0 mpirun -np 8 python finetune_sdm_yaml.py \
--cf config/ref_attn_clip_combine_controlnet/tiktok_S256L16_xformers_tsv.py \
--eval_visu --root_dir /home4/wangtan/disco_new/DisControl/output \
--local_train_batch_size 64 \
--local_eval_batch_size 64 \
--log_dir exp_test/moretiktok_ft_cfg \
--epochs 20 --deepspeed --eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 \
--fix_dist_seed \
--loss_target "noise" \
--tiktok_data_root ./tiktok_datasets \
--train_yaml /home4/wangtan/disco_data_online/TSV_data/composite_offset/train_TiktokDance-poses-masks.yaml \
--val_yaml /home4/wangtan/disco_data_online/TSV_data/composite_offset/new10val_TiktokDance-poses-masks.yaml \
--unet_unfreeze_type "null" \
--guidance_scale 1.5 \
--drop_ref 0.05 \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local \
--combine_use_mask \
--conds "poses" "masks" \
--pretrained_model /home4/wangtan/home3_file/disco_data_online/pretrain_model/checkpoint_ft/moretiktok_cfg/mp_rank_00_model_states.pt  \
--eval_save_filename MoreTikTok_cfg