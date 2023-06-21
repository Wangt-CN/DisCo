import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset, SequentialSampler, DataLoader, RandomSampler
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from PIL import Image
from tqdm import tqdm

def inception_score(imgs, cuda=True, batch_size=32, splits=10):
    N = len(imgs)
    print(f'detect images {N}')
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=16)


    # Load inception models
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm(enumerate(dataloader, 0)):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class ISDataset(Dataset):
    def __init__(self, dir):
        self.args = dir
        self.fileroot = dir
        self.filelist = os.listdir(self.fileroot)

        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, i):
        try:
            img_name = self.filelist[i]
            img_path = os.path.join(self.fileroot, img_name)
            image = self.transform(img_path)
            return image
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (self.filelist[i], e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


def calculate_is(img_dir, batch_size=32, splits=10):
    data = ISDataset(img_dir)
    mean, std = inception_score(data, cuda=True, batch_size=batch_size, splits=splits)
    return mean, std


