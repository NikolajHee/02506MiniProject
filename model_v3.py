import torch
import torch.nn as nn
from dataset import TRAIN_EM, TEST_EM
import numpy as np

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"


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
            nn.ReLU(inplace=True),
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
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

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

    def __init__(self):
        super(CNN_FOR_SEGMENTATION, self).__init__()

        # no skip connections

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        # Final layer
        self.final = nn.Conv2d(
            8,
            1,
            kernel_size=1,
        )  # Output one channel for binary mask

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass through the encoder
        # print(x.shape)
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

        print(f"Epoch {epoch+1}, Loss: {loss_[epoch].mean()}")

    return loss_


if __name__ == "__main__":
    # own imports

    from utils import cool_plots

    cool_plots()

    # parameters
    PATCH_SIZE = 128
    learning_rate = 0.0005
    batch_size = 8
    num_epochs = 100
    save_path = "pictures"
    data_path = ""
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

    train_data, test_data = (
        TRAIN_EM(data_path, patch_size=PATCH_SIZE),
        TEST_EM(data_path, patch_size=PATCH_SIZE),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    # training
    loss = train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
    )

    model.eval()
    # save loss
    from utils import save_loss

    save_loss(loss, save_path)

    # ROC curve
    from utils import roc_curve_

    roc_curve_(
        model=model,
        data=train_data,
        device=DEVICE,
        save_path=save_path,
    )

    # plot train and test images
    from utils import plot_train_images

    plot_train_images(
        model=model,
        data=train_data,
        N_images=10,
        save_path=save_path,
        device=DEVICE,
    )

    from utils import plot_test_images

    plot_test_images(
        model=model,
        data=test_data,
        N_images=10,
        save_path=save_path,
        device=DEVICE,
    )
