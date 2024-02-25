from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import shutil
model = load_model("annotator/grounded-sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "annotator/grounded-sam/checkpoints/groundingdino_swint_ogc.pth")
TEXT_PROMPT = "human."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--IMAGE_PATH', type=str, required=True, )
    args = parser.parse_args()
    return args
                        
def annotate(image_source: str, boxes: torch.Tensor) -> Image:
    image = Image.open(image_source)
    w, h = image.size
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    x = abs(xyxy[0][0] - xyxy[0][2])/2.0
    y = abs(xyxy[0][1] - xyxy[0][3])/2.0
    x_mean = abs(xyxy[0][0] + xyxy[0][2])/2.0
    y_mean = abs(xyxy[0][1] + xyxy[0][3])/2.0
    side = max(x, y)
    bbox = [x_mean-side, y_mean-side, x_mean+side, y_mean+side]
    cropped_image = image.crop(bbox)
    return cropped_image.resize((512,512), Image.BICUBIC)

class CutDataset(Dataset):  
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.names = [f for f in os.listdir(self.folder_path)]
        # self.save_path = self.folder_path.replace("input", "output")

        # if os.path.exists(self.save_path):
        #     shutil.rmtree(self.save_path)
        # os.mkdir(self.save_path)
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        file_name = self.names[idx]
        image_path = os.path.join(self.folder_path, file_name)
        _, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        annotated_frame = annotate(image_source=image_path, boxes=boxes)
        #annotated_frame.save(os.path.join(self.save_path, file_name))
        annotated_frame.save(image_path)
        return boxes
if __name__ == "__main__":
    args = parse_args()
    pose_data = CutDataset(args.IMAGE_PATH)
    batch_size = 1024 #len(pose_data.file_names)
    data_loader = DataLoader(pose_data, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(data_loader):
        _=batch
