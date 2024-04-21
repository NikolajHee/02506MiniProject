import torch
from torch import nn
from dataset import TRAIN_EM, TEST_EM

from torch.utils.data import DataLoader



class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__() 
        self.a1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.a2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.a3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
        )


        self.z1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )


        self.b1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU()
        )

        self.z2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU()
        )

        self.output = nn.Conv2d(64, 2, kernel_size=1)

        # softmax
        self.softmax = nn.Softmax(dim=1)



    
    def forward(self, x):
        x = self.a1(x)
        x1 = x

        x = self.a2(x)
        x2 = x

        x = self.a3(x)

        # x3 = x
        x = self.z1(x)

        x = self.b1(torch.cat((x, x2), 1))

        x = self.z2(x)
        x = self.b2(torch.cat((x, x1), 1))

        x = self.output(x)
        
        x = self.softmax(x)
 
        return x
    

    def train(self,
              train_loader,
              num_epochs):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=1e-5)
        loss = nn.CrossEntropyLoss()

        loss_save = []

        for epoch in range(num_epochs):
            epoch_loss = []
            for (X,y) in train_loader:
                optimizer.zero_grad()
                y_hat = self.forward(X.float())
                y = torch.cat((y==0, y==1), dim=1).float()
                train_loss = loss(y_hat.view(X.size(0), 2, -1), y.view(X.size(0), 2, -1))
                epoch_loss.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

            print(f'Epoch: {epoch+1}, Loss: {sum(epoch_loss)/len(epoch_loss)}')

            loss_save.append(sum(epoch_loss)/len(epoch_loss))
        
        return loss_save



            



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2)
    # data_1 = TRAIN_EM('02506MiniProject')
    # data_2 = TRAIN_EM('02506MiniProject', resize_=(128, 128))

    # axs[0].imshow(data_1[0][0].squeeze())
    # axs[1].imshow(data_2[0][0].squeeze())

    # plt.show()


    data = TRAIN_EM('02506MiniProject', resize_=(128, 128))

    model = model()
    train_loader = DataLoader(dataset=data,
                            batch_size=5,
                            shuffle=True)
    

    
    #print(model.forward(data[0][0].view(1, 1, 512, 512).float()).shape)
    # print('hej')

    loss = model.train(train_loader, 10)

    

    plt.plot(loss)
    plt.show()


    data2 = TEST_EM('02506MiniProject', resize_=(128, 128))

    y_hat = model.forward(data2[0].view(1, 1, 128, 128).float())

    labelled_image = y_hat.argmax(1).squeeze().detach().numpy()
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(labelled_image.squeeze())
    axs[1].imshow(data2[0].squeeze())
    plt.show()










