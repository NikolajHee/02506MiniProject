
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
from data_augmentation import mirror_horizontal, mirror_vertical, elastic_transform
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision import transforms
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
    def __init__(self, data_path, indices=None, elastic=False, mirror_h=False, mirror_v=False, total_augment=False):
        data_path = os.path.join(data_path, 'EM_ISBI_Challenge')
        self.train_images_path = sorted(glob.glob(os.path.join(data_path, 'train_images', '*.png')))
        self.train_labels_path = sorted(glob.glob(os.path.join(data_path, 'train_labels', '*.png')))

        assert self.train_images_path != [], \
            "No training images found, please check the path to the training images: " \
            f"{data_path}/train_images/*.png"

        if indices is not None: 
            # Subsetting the paths with the provided indices
            self.train_images_path = [self.train_images_path[i] for i in indices]
            self.train_labels_path = [self.train_labels_path[i] for i in indices]

        self.elastic = elastic
        self.mirror_h = mirror_h
        self.mirror_v = mirror_v
        self.total_augment = total_augment
        self.alpha = 70 #140 before
        self.sigma = 5
        self.images = []
        self.labels = []

        self.prepare_augmentations()
    

    def prepare_augmentations(self):
        for img_path, lbl_path in zip(self.train_images_path, self.train_labels_path):
            img = io.imread(img_path)
            lbl = io.imread(lbl_path)
            self.images.append(img)
            self.labels.append(lbl)
                
            if self.elastic:
                seed = np.random.randint(0, 10000)
                aug_img = elastic_transform(image=img, alpha=self.alpha, sigma=self.sigma, seed=seed)
                aug_lbl = elastic_transform(image=lbl, alpha=self.alpha, sigma=self.sigma, seed=seed)

                self.images.append(aug_img)
                self.labels.append(aug_lbl)
            
            if self.mirror_h: 
                mirrored_img = img[:, ::-1].copy()
                mirrored_lbl = lbl[:, ::-1].copy()

                self.images.append(mirrored_img)
                self.labels.append(mirrored_lbl)
            
            if self.mirror_v: 
                mirrored_img = img[::-1, :].copy()
                mirrored_lbl = lbl[::-1, :].copy()

                self.images.append(mirrored_img)
                self.labels.append(mirrored_lbl)

            if self.total_augment: 
                seed = np.random.randint(0, 10000)
                total_aug_img = elastic_transform(image=img, alpha=self.alpha, sigma=self.sigma, seed=seed)
                total_aug_lbl = elastic_transform(image=lbl, alpha=self.alpha, sigma=self.sigma, seed=seed)
                total_aug_img = total_aug_img[::-1, ::-1].copy()
                total_aug_lbl = total_aug_lbl[::-1, ::-1].copy()

                self.images.append(total_aug_img)
                self.labels.append(total_aug_lbl)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = self.images[idx]
        lbl = self.labels[idx]

        # Normalize and convert to tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255
        lbl = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0) / 255
        
        return (img, lbl)

        # return (img.view(1,self.patch_size,self.patch_size) / 255, lbl.view(1,self.patch_size,self.patch_size) / 255)

class TEST_EM(Dataset):
    """
    Unlabelled Test Data
        From the 'Segmentation of neuronal structures in EM stacks'
    """
    def __init__(self, data_path, resize_=None, patch_size=None):
        data_path = os.path.join(data_path, 'EM_ISBI_Challenge')

        self.test_images_path = glob.glob(os.path.join(data_path, 'test_images', '*.png'))

        assert self.test_images_path != [], \
            "No test images found, please check the path to the test images: " \
            f"{data_path}/test_images/*.png"

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
            return torch.tensor(io.imread(self.test_images_path[idx])).view(1, 512, 512)/255

        
if __name__ == '__main__':

    train_dataset = TRAIN_EM('', elastic=False, mirror_v=False, total_augment=True)

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
    

    plt.imshow(train_dataset[1][1].squeeze())
    # plt.imshow(train_dataset[1][1].squeeze())
    #print(f"max: {np.max(train_dataset[i][0])}, min: {np.min(train_dataset[i][0])}")

