from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from PIL import Image
import glob
import os
import numpy as np


class CreateDatasets(Dataset):
    def __init__(self, root_path, img_size, mode):
        if mode == 'train':
            A_img_path = os.path.join(root_path, 'trainA')
            B_img_path = os.path.join(root_path, 'trainB')
        elif mode == 'test':
            A_img_path = os.path.join(root_path, 'testA')
            B_img_path = os.path.join(root_path, 'testB')
        else:
            raise NotImplementedError('mode {} is error}'.format(mode))

        self.A_img_list = glob.glob(A_img_path + '/*.jpg')
        self.B_img_list = glob.glob(B_img_path + '/*.jpg')
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.A_img_list)

    def __getitem__(self, item):
        A_index = item % len(self.A_img_list)
        A_img = Image.open(self.A_img_list[A_index])
        B_index = np.random.randint(0, len(self.B_img_list) - 1)
        B_img = Image.open(self.B_img_list[B_index])
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        return A_img, B_img
