import argparse
import os, sys
# Add the current directory to sys.path
pythonpath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(pythonpath)
print(pythonpath)
import copy
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(json_out_folder_name, image_out_folder_name, numpy_out_folder_name, filename, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    # np.save(os.path.join(numpy_out_folder_name, filename + '.mask.npy'), mask_img.numpy()) # too slow for I/O

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(image_out_folder_name, filename + '.mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    mask = mask_img.numpy().astype(np.uint8)
    fname = os.path.join(image_out_folder_name, filename + '.mask.png')
    Image.fromarray(mask).save(fname)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(json_out_folder_name, filename + '.mask.json'), 'w') as f:
        json.dump(json_data, f)
    

def get_filelist(todo_list, i):
    folder_path = todo_list[i]
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    # Filter the list to only include image files
    image_files = ['{}/{}'.format(folder_path,file) for file in files if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
    return folder_path, image_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, default="./annotator/grounded-sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="/blob/keli/debug_output/vid_syn/grounded_sam/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="/blob/keli/debug_output/vid_syn/grounded_sam/sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--partition", type=int, required=True
    )
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--text_prompt", type=str, default="person")
    parser.add_argument("--todo_folder_list", type=str, default="tiktok_scale_smalllist.txt")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    todo_list = []
    with open(args.todo_folder_list, 'r') as file:
        for line in file:
            todo_list.append(line.strip())


    for index in range(args.partition, (args.partition+10)):
        print('processing video ID: {:05d}'.format(index))
        try:
            folder_path, img_list = get_filelist(todo_list, index)

            json_out_folder_name = '{}/groundsam_json'.format(folder_path)
            os.makedirs(json_out_folder_name, exist_ok=True)
            image_out_folder_name = '{}/groundsam_vis'.format(folder_path)
            os.makedirs(image_out_folder_name, exist_ok=True)
            numpy_out_folder_name = '{}/groundsam_npy'.format(folder_path)
            os.makedirs(numpy_out_folder_name, exist_ok=True)

            for item in tqdm(img_list):
                image_path = item
                # Split the file path into a directory path and a filename
                dir_path, filename = os.path.split(image_path)
                npy_out_name = os.path.join(numpy_out_folder_name, filename + '.mask.npy')

                if not os.path.isfile(npy_out_name):
                    # load image
                    image_pil, image = load_image(image_path)

                    # run grounding dino model
                    boxes_filt, pred_phrases = get_grounding_output(
                        model, image, text_prompt, box_threshold, text_threshold, device=device
                    )

                    # initialize SAM
                    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    predictor.set_image(image)

                    size = image_pil.size
                    H, W = size[1], size[0]
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cpu()
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

                    masks, _, _ = predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes.to(device),
                        multimask_output = False,
                    )
                    save_mask_data(json_out_folder_name, image_out_folder_name, numpy_out_folder_name, filename, masks, boxes_filt, pred_phrases)
        except Exception as e: 
            print(e)
            pass


