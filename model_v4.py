import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TRAIN_EM, TEST_EM
import numpy as np  

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'



class CNN_FOR_SEGMENTATION(nn.Module):
    """
    CNN for segmentation
    - 2 Convolutional layers in the encoder
    - 2 Deconvolutional layers in the decoder
    - 1 final layer
    """

    def __init__(self, ):
        super(CNN_FOR_SEGMENTATION, self).__init__() 

        # no skip connections

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        # Final layer
        self.final = nn.Conv2d(8, 1, kernel_size=1)  # Output one channel for binary mask

        self.sigmoid = nn.Sigmoid()


    
    def forward(self, x):
        # Pass through the encoder
        #print(x.shape)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x = self.encoder3(x2)

        # Pass through the decoder
        x = self.decoder1(x)
        x = self.decoder2(torch.cat((x, x2), dim=1))
        x = self.decoder3(torch.cat((x, x1), dim=1))

        # Pass through the final layer
        x = self.final(x)

        return self.sigmoid(x)


    
# TRAINING
def train(model, dataloader, optimizer, criterion, num_epochs):
    """
    Train the model
    - model: CNN model
    - dataloader: DataLoader
    - optimizer: torch.optim
    - criterion: loss function
    - num_epochs: int
    """

    model.train()
    loss_ = np.zeros((num_epochs, len(dataloader)))
    for epoch in range(num_epochs):
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()


            loss_[epoch, i] = loss.item()

        print(f'Epoch {epoch+1}, Loss: {loss_[epoch].mean()}')

    return loss_









if __name__ == '__main__':
    # own imports

    

    from utils import cool_plots
    cool_plots()

    # parameters
    PATCH_SIZE = 128
    learning_rate = 0.0005
    batch_size = 8
    num_epochs = 100
    save_path = 'pictures'
    data_path = '02506MiniProject'
    seed = 0
    

    from utils import random_seed
    random_seed(seed)
    
    # model
    model = CNN_FOR_SEGMENTATION().to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    criterion = nn.BCELoss()


    # data
    img_dim = PATCH_SIZE if PATCH_SIZE else 512

    train_data, test_data = TRAIN_EM(data_path, patch_size=PATCH_SIZE), TEST_EM(data_path, patch_size=PATCH_SIZE)


    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = batch_size,
                                               shuffle = True)
    
    # training
    loss = train(model=model, 
                 dataloader=train_loader, 
                 optimizer=optimizer, 
                 criterion=criterion,
                 num_epochs=num_epochs)

    model.eval()
    # save loss
    from utils import save_loss

    save_loss(loss, save_path)

    # ROC curve
    from utils import roc_curve_

    roc_curve_(model=model, 
               data=train_data, 
               device=DEVICE,
               save_path=save_path)
    

    # plot train and test images
    from utils import plot_train_images

    plot_train_images(model=model,
                      data=train_data,
                      N_images=10,
                      save_path=save_path,
                      device=DEVICE)


    from utils import plot_test_images

    plot_test_images(model=model,
                     data=test_data,
                     N_images=10,
                     save_path=save_path,
                     device=DEVICE)

