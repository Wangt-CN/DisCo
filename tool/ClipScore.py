import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import clip
from config import *


class CLIPSimilarity(nn.Module):
    def __init__(self, model_filename):
        super(CLIPSimilarity, self).__init__()
        self.model_filename = model_filename
        self.model, self.p = clip.load(model_filename, device='cpu')
        self.model = self.model.eval()

    def forward(self, text, image, batch_size=None):
        '''
        text: [X]
          for example,  ['str_1', 'str_2', ..., 'str_X']
        image: [Y, c, w, h]

        return: [X, Y]
        '''
        device = image.device
        img_input = F.interpolate(image, size=224)
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        img_input = (img_input.clamp(min=-1.0, max=1.0) + 1.0) / 2.0
        img_input -= image_mean[:, None, None]
        img_input /= image_std[:, None, None]
        text_input = clip.tokenize(text, truncate=True).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input).float()
            image_features = self.model.encode_image(img_input).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logit_scale = 1.0  # Default value is models.logit_scale.exp()
            if batch_size:
                text_features = rearrange(text_features, '(b x) d -> b x d', b=batch_size)
                image_features = rearrange(image_features, '(b y) d -> b d y', b=batch_size)
            else:
                image_features = image_features.t()
            logits_per_text = logit_scale * torch.matmul(text_features, image_features)
        similarity = logits_per_text
        return similarity


class BaseDataset(Dataset):
    def __init__(self, img_list, txt_list):

        self.img_list = img_list
        self.txt_list = txt_list

        assert len(img_list) == len(txt_list), 'len_img must equal len_txt!'

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img_path = self.img_list[i]
        caption = self.txt_list[i]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)

        sample = {"label_imgs": image, 'input_text': caption}
        return sample


def compute_clip_score(model_pth, img_list, txt_list, batch_size=32, cuda_device=0):

    device = torch.device(f'cuda:{cuda_device}')
    model = CLIPSimilarity(model_pth).to(device)

    data = BaseDataset(img_list, txt_list)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=32)

    clip_sim_total = 0
    num = 0
    for sample in tqdm(loader):
        imgs = sample['label_imgs'].to(device)
        caps = sample['input_text']

        clip_sim = model(caps, imgs, batch_size=len(caps))

        clip_sim_total += clip_sim.mean().item()
        num += 1

    return clip_sim_total/num
