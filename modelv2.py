import torch
from torch import nn
import numpy as np
from dataset import TRAIN_EM, TEST_EM

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"


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

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
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

        # Pass through the decoder
        x = self.decoder1(x)
        x = self.decoder2(x)

        # Pass through the final layer
        x = self.final(x)

        return self.sigmoid(x)

    def loss(self, y_hat, y):
        return -(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))


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
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_FOR_SEGMENTATION().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCELoss()

    data = TRAIN_EM("02506MiniProject", patch_size=128)

    # fig, axs = plt.subplots(1,2, figsize=(9,9))

    # axs[0].imshow(data[2][0].squeeze())
    # axs[1].imshow(data[2][1].squeeze())

    # plt.show()

    train_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=16,
        shuffle=True,
    )

    loss = train(model, train_loader, optimizer, criterion, 100)

    plt.plot(np.mean(loss, axis=1))
    plt.savefig("loss.png")
    plt.show()

    N_images = 8

    fig, axs = plt.subplots(8, 3, figsize=(9, 9))

    for i in range(N_images):
        X, y = data[i]
        y_hat = model.forward(X.to(DEVICE))

        predict = y_hat.squeeze().detach().cpu().numpy() > 0.5

        accuracy = np.mean(predict == y.squeeze().numpy())

        axs[i, 0].imshow(X.squeeze())
        axs[i, 1].imshow(predict)
        axs[i, 1].set_title(f"accuracy: {accuracy:.2f}")
        axs[i, 2].imshow(y.squeeze())

    plt.savefig("train.png")
    plt.show()

    model.eval()

    def precision_recall(y, y_hat):
        tp = y & y_hat
        fp = ~y & y_hat
        # fn = y & ~y_hat
        tn = ~y & ~y_hat

        tpr = tp.sum() / (tn.sum() + fp.sum())
        fpr = fp.sum() / (tn.sum() + fp.sum())

        return tpr, fpr

    from sklearn.metrics import roc_curve, roc_auc_score

    y = torch.tensor([])

    y_hat = torch.tensor([])

    for i in range(len(data)):
        X, y_ = data[i]
        y_hat_ = model.forward(X.to(DEVICE))

        y = torch.cat((y, y_.flatten()))
        y_hat = torch.cat((y_hat, y_hat_.flatten().detach().cpu()))

    fpr, tpr, thresholds = roc_curve(y.numpy(), y_hat.numpy())
    auc = roc_auc_score(y.numpy(), y_hat.numpy())

    plt.title(f"AUC: {auc}")
    plt.plot(fpr, tpr, linestyle="--", label="CNN")
    plt.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        linestyle="--",
        label="Random",
    )
    plt.legend()
    plt.show()

    data_ = TEST_EM("", patch_size=128)

    N_images = 8

    fig, axs = plt.subplots(8, 2, figsize=(9, 9))

    for i in range(N_images):
        y_hat = model.forward(data_[i].view(1, 128, 128).to(DEVICE))
        axs[i, 0].imshow(y_hat.squeeze().detach().cpu().numpy() > 0.5)
        axs[i, 1].imshow(data_[i].squeeze())
    plt.savefig("test.png")
    plt.show()
