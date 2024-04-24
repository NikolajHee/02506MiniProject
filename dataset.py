
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from skimage import io
import glob
import torch
from skimage.transform import resize
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder


"""
Potential Questions:
    - The images are too large
    - Should the augmentations be calculated once and saved or on the flow
"""


class TRAIN_EM(Dataset):
    """
    Labelled Training Data
        From the 'Segmentation of neuronal structures in EM stacks'
    """
    def __init__(self, data_path, resize_=None, patch_size=None):
        data_path = os.path.join(data_path, 'EM_ISBI_Challenge')

        self.train_images_path = sorted(glob.glob(os.path.join(data_path, 'train_images', '*.png')))

        self.train_labels_path = sorted(glob.glob(os.path.join(data_path, 'train_labels', '*.png')))
        
        self.resize = resize_
        self.patch_size = patch_size
        self.images = []
        self.labels = []

        if self.patch_size is not None:
            self.prepare_patches()

    def prepare_patches(self):
        for img_path, lbl_path in zip(self.train_images_path, self.train_labels_path):
            img = io.imread(img_path)
            lbl = io.imread(lbl_path)
            if self.resize:
                img = resize(img, self.resize)
                lbl = resize(lbl, self.resize)

            img_patches = self.extract_patches(img)
            lbl_patches = self.extract_patches(lbl)
            self.images.extend(img_patches)
            self.labels.extend(lbl_patches)

    def extract_patches(self, image):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        patches = image.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)
        return patches.contiguous().view(-1, self.patch_size, self.patch_size)

    def __len__(self):
        if self.patch_size: 
            return len(self.images)
        else:   
            return len(self.train_images_path)

    def __getitem__(self, idx):
        if self.patch_size is not None:
            img = self.images[idx]
            lbl = self.labels[idx]
            return (img.view(1,self.patch_size,self.patch_size) / 255, lbl.view(1,self.patch_size,self.patch_size) / 255)
        else:
            return (torch.tensor(io.imread(self.train_images_path[idx])).view(1, 512, 512)/255, 
                    torch.tensor(io.imread(self.train_labels_path[idx])).view(1, 512, 512)/255)
        
class TEST_EM(Dataset):
    """
    Unlabelled Test Data
        From the 'Segmentation of neuronal structures in EM stacks'
    """
    def __init__(self, data_path, resize_=None, patch_size=None):
        data_path = os.path.join(data_path, 'EM_ISBI_Challenge')

        self.test_images_path = glob.glob(os.path.join(data_path, 'test_images', '*.png'))

        self.resize = resize_
        self.patch_size = patch_size
        self.images = []

        if self.patch_size is not None:
            self.prepare_patches()

    def prepare_patches(self):
        for img_path in self.test_images_path:
            img = io.imread(img_path)
            if self.resize:
                img = resize(img, self.resize)
            img_patches = self.extract_patches(img)
            self.images.extend(img_patches)

    def extract_patches(self, image):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        patches = image.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)
        return patches.contiguous().view(-1, self.patch_size, self.patch_size)


    def __len__(self):
        if self.patch_size: 
            return len(self.images)
        else:   
            return len(self.test_images_path)

    
    def __getitem__(self, idx):
        if self.patch_size is not None:
            img = self.images[idx]
            return img.view(1,self.patch_size,self.patch_size) / 255
        else:
            torch.tensor(io.imread(self.test_images_path[idx])).view(1, 512, 512)/255

        
if __name__ == '__main__':

    train_dataset = TRAIN_EM('', patch_size=128)

    # for training in batches:
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=5,
                            shuffle=True)
    
    for i, batch in enumerate(train_loader):
        print(f"batch {i}")
        print(f"shape of image: {batch[0].shape}")
        print(f"shape of label: {batch[1].shape}")
        print(f"type of image: {type(batch[0])}")
        print(f"datatype of image: {batch[0].dtype}")
        break

    
    # Alternatively (without the use of batches)
    print(f'the dataset size is {len(train_dataset)}')
    for i in range(len(train_dataset)):
        print(f"image {i}")
        print(f"shape of image: {train_dataset[i][0].shape}")
        print(f"shape of label: {train_dataset[i][1].shape}")
        print(f"type of image: {type(batch[0])}")
        print(f"datatype of image: {batch[0].dtype}")

        break
    
    #print(f"max: {np.max(train_dataset[i][0])}, min: {np.min(train_dataset[i][0])}")

