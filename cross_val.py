import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch.nn.functional as F
from dataset import TRAIN_EM, TEST_EM
import numpy as np

from model_v3 import CNN_FOR_SEGMENTATION

def dice_coefficient(pred, target):
    '''
    Computes the dice coefficient
    '''
    smooth = 1.
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def cross_validate_model(model_name : str, dataset, k_folds=5, epochs=10, batch_size=16, learning_rate=0.001):
    '''
    Function to easily perform cross-validation on CNN
    
    Args:
        model_name: a string indication a neural network (CNN) (suboptimal implementation, but quick fix)
        dataset: dataset of choice
        k_folds: number of folds in the CV
        epochs: number of epochs to train in each fold
        batch_size: batch_size to use
        learning_rate: optimizer learning rate
    '''
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # criterion = nn.BCELoss()  # Modify if using a different loss function
    criterion = nn.BCEWithLogitsLoss()
    fold_performance = []

    fold_tqdm = tqdm(enumerate(kf.split(dataset)), total=k_folds, desc="CV Folds Progress")
    # for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    for fold, (train_idx, val_idx) in fold_tqdm:
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

        if model_name == "v3":
            model = CNN_FOR_SEGMENTATION().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        total_dice = 0
        count = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                output = (output > 0.5).float() # apply threshold to convert probabilities to binary output
                dice_score = dice_coefficient(output, y_batch)
                total_dice += dice_score.item()
                count += 1

        fold_dice = total_dice / count
        fold_performance.append(fold_dice)
        print(f"Fold {fold+1}, Dice Coefficient: {fold_dice}")

    return fold_performance


if __name__ == '__main__':
    dataset = TRAIN_EM('')

    cross_validate_model(model_name="v3", dataset=dataset, k_folds=8, epochs=200, batch_size=16, learning_rate=0.001)
    