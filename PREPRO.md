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
# 1. Follow the pytorch-openpose repo (https://github.com/Hzzone/pytorch-openpose) for package installation

# 2. download the pre-trained weights
wget -O body_pose_model.pth https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth?dl=1  
```

#### Run

```
python ./annotator/openpose/run.py --dataset_root /path/to/target/dataset
```





### TSV preparation

After running Grounded SAM and OpenPose, prepare the data structured as below:

```
Target Dataset
└── 001/
    ├── 0000.png
    ├── 0001.png
    ├── ...
    ├────── groundsam
    |       ├── 000001.png.mask.jpg
    |       ├── 000002.png.mask.jpg
    |       └── ...     
    └────── openpose_json
            ├── 000001.png.json
            ├── 000002.png.json
            └── ...
└── 002/
    ├── 0000.png
    ├── 0001.png
    ├── ...
    ├────── groundsam
    |       ├── 000001.png.mask.jpg
    |       ├── 000002.png.mask.jpg
    |       └── ...     
    └────── openpose_json
            ├── 000001.png.json
            ├── 000002.png.json
            └── ...
```

We also provide a preprocessed toy dataset as example. You may find the example dataset in folder `./tsv_example/toy_dataset`

Run the following script to convert your dataset to tsv format

```
python ./tsv_example/create_custom_dataset_tsvs.py --split train --root_folder PATH_TO_YOUR_DATASET_FOLDER --output_folder PATH_TO_DESIRED_FOLDER 
```

For instance, `PATH_TO_YOUR_DATASET_FOLDER=./tsv_example/toy_dataset` and `PATH_TO_DESIRED_FOLDER=./tsv_example/toy_dataset/tsv`


### TSV visualization

Once the tsv files are generated, we provide a jupyter notebook for data visualization. Please refer to `visualize_tsv.ipynb` for details.