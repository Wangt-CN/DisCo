Here we provide the scripts for reference to automatically process the foreground mask and generate the human skeleton.

The target dataset which need to be processed should be like:

```
Target Dataset
└── 001/
    ├── 0000.png
    └── 0001.png
    ...
└── 002/
    ├── 0000.png
    └── 0001.png 
    ...
```



### Grounded-SAM

#### Installation

```
# 1. Follow the official repo (https://github.com/IDEA-Research/Grounded-Segment-Anything) for package installation

# 2. download the pre-trained weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

#### Run

```
python ./annotator/grounded-sam/run.py --dataset_root /path/to/target/dataset
```





### OpenPose

#### Installation

```
# 1. Follow the official openpose website for package installation

# 2. download the pre-trained weights
wget -O body_pose_model.pth https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth?dl=1  
```

#### Run

```
python ./annotator/openpose/run.py --dataset_root /path/to/target/dataset
```

