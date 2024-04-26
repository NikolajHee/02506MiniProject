import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def cool_plots():
    """
    Function to set the style of the plots to the science style.
    """
    try:
        import scienceplots
    except:
        print("Plots will not be in the science style.")
    import matplotlib.pyplot as plt
    plt.style.use('science')
    plt.rcParams.update({'figure.dpi': '200'})
    plt.rcParams.update({"legend.frameon": True})

def random_seed(random_seed):
    """
    Function to seed the data-split and backpropagation (to enforce reproducibility)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed+1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed+2)
    random.seed(random_seed+3)



def roc_curve_(model, data, device, save_path=None):
    from sklearn.metrics import roc_curve, roc_auc_score

    y = torch.tensor([])

    y_hat = torch.tensor([])

    for i in range(len(data)):
        X, y_ = data[i]
        y_hat_ = model.forward(X.to(device))

        y = torch.cat((y, y_.flatten()))
        y_hat = torch.cat((y_hat, y_hat_.flatten().detach().cpu()))

    fpr, tpr, thresholds = roc_curve(y.numpy(), y_hat.numpy())
    auc = roc_auc_score(y.numpy(), y_hat.numpy())

    plt.title(f'AUC: {auc}')
    plt.plot(fpr, tpr, linestyle='--', label='CNN')
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', label='Random')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.show()



def save_loss(loss, save_path):

    epoch, batches = loss.shape

    plt.figure(figsize=(8, 4))

    plt.plot(np.arange(0, epoch*batches, step=batches), np.mean(loss, axis=1), label='Epoch mean')
    plt.plot(loss.reshape(-1)[batches//2:epoch*batches-batches//2], label='Batch loss', alpha=0.5)


    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.xticks(np.arange(0, epoch*batches, step=batches*10), np.arange(1, epoch+1, 10))
    plt.title('Binary Classifcation Entropy Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.show()


def plot_train_images(model, data, N_images, save_path, device):
    _, img_dim, _ = data[0][0].shape

    assert N_images < len(data)

    indexes = np.random.choice(len(data), N_images, replace=False)

    for i in range(N_images):
        fig, ax = plt.subplots(1, 2, figsize=(4,2))

        image, label = data[indexes[i]]

        y_hat = model.forward(image.view(1, img_dim, img_dim).to(device))

        predict = y_hat.squeeze().detach().cpu().numpy() > 0.5

        accuracy = np.mean(predict == label.view(img_dim, img_dim).numpy())


        ax[0].imshow(label.view(img_dim,img_dim), cmap='viridis')
        ax[0].set_title("$y_{" + str(indexes[i]) + "}$")
        ax[0].axis('off')
        ax[1].imshow(y_hat.squeeze().detach().cpu().numpy() > 0.5, cmap='viridis')
        ax[1].set_title("$\hat{y}_{" + str(indexes[i]) + "}$")
        ax[1].axis('off')
        
        plt.savefig(os.path.join(save_path, f'train_{indexes[i]}.png'))
        #plt.show()


def plot_test_images(model, data, N_images, save_path, device):
    _, img_dim, _ = data[0].shape

    indexes = np.random.choice(len(data), N_images, replace=False)

    for i in range(N_images):
        fig, ax = plt.subplots(1, 2, figsize=(4, 2))

        image = data[indexes[i]]

        y_hat = model.forward(image.view(1, img_dim, img_dim).to(device))
        ax[0].imshow(image.squeeze())
        ax[0].set_title("$X_{" + str(indexes[i]) + "}$")
        ax[0].axis('off')
        ax[1].imshow(y_hat.squeeze().detach().cpu().numpy() > 0.5)
        ax[1].set_title("$\hat{y}_{" + str(indexes[i]) + "}$")
        ax[1].axis('off')
        
        
        plt.savefig(os.path.join(save_path, f'test_{indexes[i]}.png'))
        #plt.show()
