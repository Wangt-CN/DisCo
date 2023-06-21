# DisCo: Disentangled Control for Referring Human Dance Generation in Real World

<a href='https://disco-dance.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='xxx'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://a8b0b9c5d9ee9d6c62.gradio.live/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/alJKsj3JpBo)

[Tan Wang*](https://wangt-cn.github.io/),  [Linjie Li*](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en),  [Kevin Lin*](https://scholar.google.com/citations?hl=en&user=LKSy1kwAAAAJ),  [Chung-Ching Lin](https://scholar.google.com/citations?hl=en&user=legkbM0AAAAJ),  [Zhengyuan Yang](https://scholar.google.com/citations?hl=en&user=rP02ve8AAAAJ),  [Hanwang Zhang](https://scholar.google.com/citations?hl=en&user=YG0DFyYAAAAJ),  [Zicheng Liu](https://scholar.google.com/citations?hl=en&user=bkALdvsAAAAJ),  [Lijuan Wang](https://scholar.google.com/citations?hl=en&user=cDcWXuIAAAAJ)

**Nanyang Technological University  |  Microsoft Azure AI**

[![DisCo: Disentangled Control for Referring Human Dance Generation in Real World](https://res.cloudinary.com/marcomontalbano/image/upload/v1686644061/video_to_markdown/images/youtube--alJKsj3JpBo-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/alJKsj3JpBo "DisCo: Disentangled Control for Referring Human Dance Generation in Real World")





## :fire: News

* **[2023.06.21]** DisCo Human Image Editing Demo is released! Have a try!
* **[2023.06.21]** We release the human-specific fine-tuning code for reference.  Come and build your own dance model!
* **[2023.06.21]**  Release the code for general fine-tuning and human-specific fine-tuning .
* **[2023.06.21]** We release the human attribute pre-trained checkpoint and the fine-tuning checkpoint.



## :paintbrush: DEMO 

[[Online Gradio Demo]](https://a8b0b9c5d9ee9d6c62.gradio.live/) (Video dance generation demo is on the way!)

<p align="center">
  <img src="figures/demo.gif" width="90%" height="90%">
</p>







## :notes: Introduction

In this project, we introduce **DisCo** as a generalized referring human dance generation toolkit, which supports both **human image & video generation** with **multiple usage cases** (pre-training, fine-tuning, and human-specific fine-tuning), especially good in real-world scenarios.



#### It achieves:

- Current SOTA results for referring human dance generation, especially outperforming existing methods in terms of **generalizability to the real world scenarios**.

- Extensive usage cases and applications (see [project page](https://disco-dance.github.io/index.html) for more details). 

- An easy-to-follow framework, supporting **efficient training** (x-formers, FP16 training, deepspeed, wandb) and **a wide range of possible research directions** (pre-training -> fine-tuning -> human-specific fine-tuning).

  

#### With this project, you can get:

- *\[User\]*: Just try our online demo! Or deploy the model inference locally. 
- *\[Researcher\]*: An easy-to-use codebase for re-implementation.
- *\[Researcher\]*: A large amount of research directions for further improvement.





## Getting Started

### Installation

```sh
pip install --user torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user progressbar psutil pymongo simplejson yacs boto3 pyyaml ete3 easydict deprecated future django orderedset python-magic datasets h5py omegaconf einops ipdb
pip install --user --exists-action w -r requirements.txt
pip install git+https://github.com/microsoft/azfuse.git


## for acceleration
pip install --user deepspeed==0.6.3
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```



### Data Preparation

##### 1. Human Attribute Pre-training

We create a human image subset (700K Images) filtered from existing image corpus for human attribute pre-training:

| Dataset  | COCO (Single Person) | TikTok | DeepFashion2 | SHHQ-1.0 | LAION-Human |
| -------- | :------------------: | :----: | :----------: | :------: | :---------: |
| **Size** |         20K          |  90K   |     296K     |   40K    |    240K     |

##### 2. Fine-tuning with Disentangled Control

We use the [TikTok dataset](https://www.yasamin.page/hdnet_tiktok) for the fine-tuning. 

We have already pre-processed the tiktok data with the efficient TSV format which can be downloaded **[here (Google Cloud)](https://console.cloud.google.com/storage/browser/disco-data-share)**. (Note that we only use the 1st frame of each TikTok video as the reference image.)

The data folder structure should be like:

```
Data Root
└── composite_offset/
    ├── train_xxx.yaml  # The path need to be then specified in the training args
    └── val_xxx.yaml
    ...
└── TikTokDance/
    ├── xxx_images.tsv
    └── xxx_poses.tsv
    ...
```





### Human Attribute Pre-training (Code Coming Soon)

<p align="center">
  <img src="figures/pretrain.gif" width="80%" height="80%">
</p>



**Pre-trained Model Checkpoint: [Google Cloud](https://storage.googleapis.com/disco-checkpoint-share/checkpoint_pretrain/0.7m_pretrain/mp_rank_00_model_states.pt)**





### Fine-tuning with Disentangled Control 

![Image](figures/ft1.gif)

![Image](figures/ft2.gif)



#### 1. Modify the config file

Download the `sd-image-variations-diffusers` from official [diffusers repo](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) and put it according to the config file `pretrained_model_path`. Or you can also choose to modify the `pretrained_model_path`.



#### 2. w/o Classifier-Free Guidance (CFG)

**Training:**

[*To enable WANDB, set up the wandb key in `utils/lib.py`]

```python
AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 python finetune_sdm_yaml.py --cf config/ref_attn_clip_combine_controlnet/tiktok_S256L16_xformers_tsv.py \
--do_train --root_dir /home1/wangtan/code/ms_internship2/github_repo/run_test \ 
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--log_dir exp/tiktok_ft \ 
--epochs 20 --deepspeed \
--eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--train_yaml /home/wangtan/data/disco/yaml_file/train_TiktokDance-poses-masks.yaml \
--val_yaml /home/wangtan/data/disco/yaml_file/new10val_TiktokDance-poses-masks.yaml \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--conds "poses" "masks" \
--stage1_pretrain_path /path/to/pretrained_model_checkpoint/mp_rank_00_model_states.pt 
```

**Evaluation:**

We use `gen_eval.sh` to one-stop get the evaluation metrics for {exp_dir_path}/{exp_folder_name})

```
bash gen_eval.sh {exp_dir_path} {exp_folder_name}
```

##### Model Checkpoint (Google Cloud): [TikTok Training Data (FID-FVD: 20.2)]( xxx) | [More TikTok-Style Training Data (FID-FVD: 18.7)](https://storage.googleapis.com/disco-checkpoint-share/checkpoint_ft/moretiktok_nocfg/mp_rank_00_model_states.pt)




#### 3. w/ Classifier-Free Guidance (CFG) [CFG can bring a slightly better results]

**Training (add the following args into the training script of w/o CFG):**

```
--drop_ref 0.05 # probability to dropout the reference image during training
--guidance_scale 1.5 # the scale of the CFG
```

**Evaluation:**

We use `gen_eval.sh` to one-stop get the evaluation metrics for {exp_dir_path}/{exp_folder_name})

```
bash gen_eval.sh {exp_dir_path} {exp_folder_name}
```

##### Model Checkpoint (Google Cloud): [TikTok Training Data (FID-FVD: 18.8)](https://storage.googleapis.com/disco-checkpoint-share/checkpoint_ft/tiktok_cfg/mp_rank_00_model_states.pt) | [More TikTok-Style Training Data (FID-FVD: 15.7)](https://storage.googleapis.com/disco-checkpoint-share/checkpoint_ft/moretiktok_cfg/mp_rank_00_model_states.pt)





### Human-Specific Fine-tuning

![Image](figures/human_specific_ft.gif)



#### 1. Prepare dataset that you want to use for training

- Prepare a human-specific video or a set of human images

- Use Grounded-SAM and OpenPose to obtain human mask and human skeleton for each training image (See [PREPRO.MD](PREPRO.MD) for more details)

  

#### 2. Run the following script for human-specific fine-tuning:

For parameter tuning, recommend to first tune the `learning-rate` and `unet_unfreeze_type`.

```python
AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 python finetune_sdm_yaml.py \
--cf config/ref_attn_clip_combine_controlnet_imgspecific_ft/webtan_S256L16_xformers_upsquare.py --do_train --root_dir /path/of/saving/root \
--local_train_batch_size 32 --local_eval_batch_size 32 --log_dir exp/human_specific_ft/ \
--epochs 20 --deepspeed --eval_step 500 --save_step 500 --gradient_accumulate_steps 1 \
--learning_rate 1e-3  --fix_dist_seed  --loss_target "noise" \
--unet_unfreeze_type "crossattn" --refer_sdvae --ref_null_caption False --combine_clip_local --combine_use_mask --conds "poses" "masks" \
--freeze_pose True --freeze_background False \
--pretrained_model /path/to/the/ft_model_checkpoint \
--ft_iters 500 --ft_one_ref_image False --ft_idx dataset/folder/name --strong_aug_stage1 True --strong_rand_stage2 True
```





## Release Plan

- [x] Code for "Fine-tuning with Disentangled Control"
- [x] Code for "Human-Specific Fine-tuning"
- [x] Model Checkpoints for Pre-training and Fine-tuning
- [x] HuggingFace Demo
- [ ] Code for "Human Attribute Pre-training"





## Citation	

If you use our work in your research, please cite: 

```
@article{disco,
title={DisCo: Disentangled Control for Referring Human Dance Generation in Real World},
author={Wang, Tan and Li, Linjie and Lin, Kevin and Lin, Chung-Ching and Yang, Zhengyuan and Liu, Zicheng and Wang, Lijuan},
website={https://disco-dance.github.io/},
year={2023}
}
```

