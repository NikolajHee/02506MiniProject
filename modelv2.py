
import torch
from torch import nn
from dataset import TRAIN_EM, TEST_EM
import numpy as np  

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'



class CNN_FOR_SEGMENTATION(nn.Module):
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
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # Final layer
        self.final = nn.Conv2d(8, 1, kernel_size=1)  # Output one channel for binary mask

        self.sigmoid = nn.Sigmoid()


    
    def forward(self, x):
        # Pass through the encoder
        x = self.encoder1(x)
        x = self.encoder2(x)

        # Pass through the decoder
        x = self.decoder1(x)
        x = self.decoder2(x)

        # Pass through the final layer
        x = self.final(x)

        return self.sigmoid(x)


    def loss(self, y_hat, y):
        return - (y * torch.log(y_hat) + (1-y) * torch.log(1-y_hat))
    


def train(model, dataloader, optimizer, criterion, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN_FOR_SEGMENTATION().to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    criterion = nn.BCELoss()

    data = TRAIN_EM('', patch_size=128)

    # fig, axs = plt.subplots(1,2, figsize=(9,9))

    # axs[0].imshow(data[2][0].squeeze())
    # axs[1].imshow(data[2][1].squeeze())

    # plt.show()


    train_loader = torch.utils.data.DataLoader(dataset = data,
                                               batch_size = 1,
                                               shuffle = True)
    

    train(model, train_loader, optimizer, criterion, 10)

    

    N_images = 8

    fig, axs = plt.subplots(8,3, figsize=(9,9))


    for i in range(N_images):
        y_hat = model.forward(data[i][0].view(1,128,128).to(device))

        predict = y_hat.squeeze().detach().cpu().numpy() > 0.5

        accuracy = np.mean(predict == data[i][1].view(128,128).numpy())

        axs[i, 0].imshow(y_hat.squeeze().detach().cpu().numpy())
        axs[i, 1].imshow(y_hat.squeeze().detach().cpu().numpy() > 0.5)
        axs[i, 2].imshow(data[i][1].view(128,128))


    plt.show()
        




    





