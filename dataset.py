
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from skimage import io
import glob
import torch
from skimage.transform import resize
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
    def __init__(self, data_path, resize_=None):
        data_path = os.path.join(data_path, 'EM_ISBI_Challenge')

        self.train_images_path = glob.glob(os.path.join(data_path, 'train_images', '*.png'))

        self.train_labels_path = glob.glob(os.path.join(data_path, 'train_labels', '*.png'))
        
        self.resize = resize_

    def __len__(self):
        return len(self.train_images_path)

    def __getitem__(self, idx):
        if self.resize is None:
            return (torch.tensor(io.imread(self.train_images_path[idx])).view(1, 512, 512)/255, 
                    torch.tensor(io.imread(self.train_labels_path[idx])).view(1, 512, 512)/255)
        return (torch.tensor(resize(io.imread(self.train_images_path[idx]), self.resize)).view(1, self.resize[0], self.resize[1])/255,
                torch.tensor(resize(io.imread(self.train_labels_path[idx]), self.resize)).view(1, self.resize[0], self.resize[1])/255)


class TEST_EM(Dataset):
    """
    Unlabelled Test Data
        From the 'Segmentation of neuronal structures in EM stacks'
    """
    def __init__(self, data_path, resize_=None):
        data_path = os.path.join(data_path, 'EM_ISBI_Challenge')

        self.test_images_path = glob.glob(os.path.join(data_path, 'test_images', '*.png'))

        self.resize = resize_

    def __len__(self):
        return len(self.test_images_path)

    
    def __getitem__(self, idx):
        if self.resize is None:
            return torch.tensor(io.imread(self.test_images_path[idx])).view(1, 512, 512)/255
        return torch.tensor(resize(io.imread(self.test_images_path[idx]), self.resize)).view(1, self.resize[0], self.resize[1])/255


if __name__ == '__main__':

    train_dataset = TRAIN_EM('')

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
    for i in range(len(train_dataset)):
        print(f"image {i}")
        print(f"shape of image: {train_dataset[i][0].shape}")
        print(f"shape of label: {train_dataset[i][1].shape}")
        print(f"type of image: {type(batch[0])}")
        print(f"datatype of image: {batch[0].dtype}")

        break
    
    #print(f"max: {np.max(train_dataset[i][0])}, min: {np.min(train_dataset[i][0])}")

