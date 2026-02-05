from data_augmentation import elastic_transform

from utils import cool_plots

from dataset_v2 import TRAIN_EM

import numpy as np
import matplotlib.pyplot as plt


data = TRAIN_EM("")


alpha = 70
sigma = 5
seed = 0

img, lbl = data[0]
img = np.array(img.squeeze())
lbl = np.array(lbl.squeeze())


cool_plots()


aug_img = elastic_transform(image=img, alpha=alpha, sigma=sigma, seed=seed)
aug_lbl = elastic_transform(image=lbl, alpha=alpha, sigma=sigma, seed=seed)


mirrored_img_1 = img[:, ::-1].copy()
mirrored_lbl = lbl[:, ::-1].copy()


mirrored_img_2 = img[::-1, :].copy()
mirrored_lbl = lbl[::-1, :].copy()


fig, axs = plt.subplots(1, 4)

axs[0].imshow(img)
axs[0].axis("off")
axs[0].set_title("Original")

axs[1].imshow(aug_img)
axs[1].axis("off")
axs[1].set_title("Elastic")

axs[2].imshow(mirrored_img_1)
axs[2].axis("off")
axs[2].set_title("Mirror H")

axs[3].imshow(mirrored_img_2)
axs[3].axis("off")
axs[3].set_title("Mirror V")

plt.tight_layout()
plt.savefig("pictures/augmentation.png")
plt.show()
