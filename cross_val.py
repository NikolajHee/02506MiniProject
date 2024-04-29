import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import glob, os
import torch.nn.functional as F
from dataset_v2 import TRAIN_EM, TEST_EM
import numpy as np



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

def rand_error(y, y_hat):
    # not optimal implementation, but quick fix
    from sklearn.metrics import rand_score

    n, c, img_dim = y.shape[0], y.shape[1], y.shape[2]

    y_hat = (y_hat > 0.5).reshape(n, img_dim * img_dim).cpu().numpy()
    
    y = y.int().reshape(n, img_dim * img_dim).cpu().numpy()

    RI = np.zeros(n)

    for i in range(n):
        RI[i] = rand_score(y_hat[i], y[i])

    return np.mean(1 - RI)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def cross_validate_model(model_name : str, indices, k_folds=5, epochs=10, batch_size=16, learning_rate=0.001):
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
    criterion = nn.BCELoss()
    fold_performance = []

    # fold_tqdm = tqdm(enumerate(kf.split(dataset)), total=k_folds, desc="CV Folds Progress")
    # for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

    train_loss = np.zeros((k_folds, epochs))
    test_loss = np.zeros((k_folds, epochs))
    dice_scores = np.zeros((k_folds, epochs))
    rand_error_scores = np.zeros((k_folds, epochs))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_dataset = TRAIN_EM(data_path, indices=train_idx, elastic=True, mirror_h=True, mirror_v=True, total_augment=True)
        val_dataset = TRAIN_EM(data_path, indices=val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if model_name == "v3":
            from model_v3 import CNN_FOR_SEGMENTATION
            model = CNN_FOR_SEGMENTATION().to(device)
            model.apply(init_weights) # initialize weights
            model.train()

        if model_name == 'v4':
            from model_v4 import CNN_FOR_SEGMENTATION
            model = CNN_FOR_SEGMENTATION().to(device)
            model.apply(init_weights) # initialize weights
            model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()

            temp_train_loss_list = np.zeros(len(train_loader))
            temp_test_loss_list = np.zeros(len(val_loader))
            temp_dice_list = np.zeros(len(val_loader))
            temp_rand_error_list = np.zeros(len(val_loader))


            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()


                temp_train_loss_list[i] = loss.item()
            
            train_loss[fold, epoch] = np.mean(temp_train_loss_list)

            model.eval()
            with torch.no_grad():
                for j, (X_batch, y_batch) in enumerate(val_loader):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    
                    loss = criterion(output, y_batch)

                    output = (output > 0.5).float() # apply threshold to convert probabilities to binary output

                    

                    dice_score = dice_coefficient(output, y_batch)

                    rand_error_score = rand_error(output, y_batch)

                    temp_test_loss_list[j] = loss.item()
                    temp_rand_error_list[j] = rand_error_score
                    temp_dice_list[j] = dice_score.item()
            

            test_loss[fold, epoch] = np.mean(temp_test_loss_list)
            dice_scores[fold, epoch] = np.mean(temp_dice_list)
            rand_error_scores[fold, epoch] = np.mean(temp_rand_error_list)

        print(f"Fold {fold+1}",  
              f"Train loss: {train_loss[fold, epoch]}",  
              f"Test loss: {test_loss[fold, epoch]}", 
              f"Dice Score: {dice_scores[fold, epoch]}", 
              f"Rand Error: {rand_error_scores[fold, epoch]}.")


    return {"train_loss": train_loss, \
            "test_loss": test_loss,  
            "dice_scores": dice_scores, 
            "rand_error": rand_error_scores}


if __name__ == '__main__':
    dataset = TRAIN_EM('')

    data_path = ""
    full_indices = list(range(len(glob.glob(os.path.join(data_path, 'EM_ISBI_Challenge/train_images', '*.png')))))
    result = cross_validate_model(model_name="v4", indices=full_indices, k_folds=8, epochs=15, batch_size=16, learning_rate=0.001)

    from utils import cool_plots

    cool_plots()
    
    epoch, batches = result['train_loss'].shape

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))

    for i in range(8):
        plt.plot(result['train_loss'][i], label=f'Fold {i+1}', alpha=0.5)
    plt.title('Binary Classification Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




