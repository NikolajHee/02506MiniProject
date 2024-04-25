import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TRAIN_EM, TEST_EM
import numpy as np  

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = DoubleConv(1, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        
        x = self.conv_last(x)
        return self.sigmoid(x)

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
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # Final layer
        self.final = nn.Conv2d(8, 1, kernel_size=1)  # Output one channel for binary mask

        self.sigmoid = nn.Sigmoid()


    
    def forward(self, x):
        # Pass through the encoder
        #print(x.shape)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        # Pass through the decoder
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)

        # Pass through the final layer
        x = self.final(x)

        return self.sigmoid(x)


    def loss(self, y_hat, y):
        return - (y * torch.log(y_hat) + (1-y) * torch.log(1-y_hat))
    
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
    import matplotlib.pyplot as plt

    PATCH_SIZE = None
    if PATCH_SIZE == None:
        img_dim = 512
    else:
        img_dim = PATCH_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN_FOR_SEGMENTATION().to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    criterion = nn.BCELoss()

    data = TRAIN_EM('', patch_size=PATCH_SIZE)

    # fig, axs = plt.subplots(1,2, figsize=(9,9))

    # axs[0].imshow(data[2][0].squeeze())
    # axs[1].imshow(data[2][1].squeeze())

    # plt.show()


    train_loader = torch.utils.data.DataLoader(dataset = data,
                                               batch_size = 16,
                                               shuffle = True)
    

    loss = train(model, train_loader, optimizer, criterion, 200)

    plt.plot(np.mean(loss, axis=1))
    plt.savefig('loss.png')
    plt.show()

    

    N_images = 8

    fig, axs = plt.subplots(8,3, figsize=(9,9))

        
    for i in range(N_images):
        y_hat = model.forward(data[i][0].view(1,img_dim,img_dim).to(DEVICE))

        predict = y_hat.squeeze().detach().cpu().numpy() > 0.5

        accuracy = np.mean(predict == data[i][1].view(img_dim,img_dim).numpy())

        axs[i, 0].imshow(data[i][0].squeeze())
        axs[i, 1].imshow(y_hat.squeeze().detach().cpu().numpy() > 0.5)
        axs[i, 1].set_title(f"accuracy: {accuracy:.2f}")
        axs[i, 2].imshow(data[i][1].view(img_dim,img_dim))
    plt.savefig('train.png')
    plt.show()


    data_ = TEST_EM('', patch_size=PATCH_SIZE)


    N_images = 8

    fig, axs = plt.subplots(8,2, figsize=(9,9))


    for i in range(N_images):
        y_hat = model.forward(data_[i].view(1,img_dim,img_dim).to(DEVICE))
        axs[i, 0].imshow(y_hat.squeeze().detach().cpu().numpy() > 0.5)
        axs[i, 1].imshow(data_[i].squeeze())
    plt.savefig('test.png')
    plt.show()
