import os
import glob
import copy
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

# from sampler import BalancedBatchSampler

class MVTecAD(Dataset):

    def __init__(self, root='/home/dataset/mvtec', category='bottle', transform=None, mode='train'):
        self.root = root
        self.category = category
        self.transform = transform
        self.mode = mode
        self.data_dir = os.path.join(root, category, 'test')
        
        ano_class_dir = [p for p in Path(self.data_dir).iterdir() if p.is_dir()]
        ano_img_paths = []
        for dir_path in ano_class_dir[0:-1]:
            ano_img_paths.extend([str(p) for p in dir_path.iterdir() if p.is_file()])
        ano_labels = [torch.tensor(1) for _ in range(len(ano_img_paths))]

        normal_img_paths = [str(p) for p in Path(self.data_dir+'/good').iterdir() if p.is_file()]
        normal_labels = [torch.tensor(0) for _ in range(len(normal_img_paths))]

        # 正常が少ないときは増やす
        if len(ano_labels) > len(normal_labels):
            diff_num = np.abs(len(ano_labels)-len(normal_labels))
            add_normal_path = [str(p) for p in Path(os.path.join(root, category,'train','good')).iterdir() if p.is_file()]
            normal_img_paths.extend(add_normal_path[0:diff_num])
            normal_labels.extend([torch.tensor(0) for _ in range(diff_num)])

            extra_add_num = int(len(normal_img_paths)/2)
            normal_img_paths.extend(add_normal_path[diff_num:diff_num+extra_add_num])
            normal_labels.extend([torch.tensor(0) for _ in range(extra_add_num)])

        all_img_paths = normal_img_paths + ano_img_paths
        all_labels = normal_labels + ano_labels

        train_img_paths, test_img_paths, train_labels, test_labels = train_test_split(
            all_img_paths, all_labels, test_size=0.3, random_state=42)
        self.train_imgs = train_img_paths
        self.train_labels = train_labels
        self.test_imgs = test_img_paths
        self.test_labels = test_labels 
        # import pdb;pdb.set_trace()

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(self.train_imgs[index]).convert('RGB')
            img = self.transform(img)
            label = self.train_labels[index]
            path = self.train_imgs[index]
        else:
            img = Image.open(self.test_imgs[index]).convert('RGB')
            img = self.transform(img)
            label = self.test_labels[index]
            path = self.test_imgs[index]
        return img, label, path

    def __len__(self):
        if self.mode=="train":
            return len(self.train_imgs)
        else:
            return len(self.test_imgs)


class mvtecAD_loader():
    def __init__(self,category):
        self.category = category



    def run(self,transform,mode):
        if mode=='train':
            self.train_dataset=MVTecAD(category=self.category, transform=transform, mode='train')
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=16, 
                shuffle=True, 
                num_workers=4,
                drop_last=True)
            return self.train_loader
        else:
            self.test_dataset=MVTecAD(category=self.category, transform=transform, mode='test')
            self.test_loader = DataLoader(
                self.test_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=4)
            return self.test_loader

    def get_img_num(self,mode):
        if mode=='train':
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)



if __name__ == '__main__':
    transform_img = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    print('setup loader')
    loader = mvtecAD_loader(category='bottle',percentage=0.5)
    train_dataset = loader.run(transform=transform_img,mode='train')