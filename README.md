# 02506MiniProject
A PyTorch Convolutional Neural Network (CNN) trained on the data 'Segmentation of neuronal structures in EM stacks1'

# Project plan
1) Create dataloader (**done**)
2) Split input images into smaller patches, e.g. 128x128 (**done**)
3) Set up simple version of U-net (**in progress**)
4) Set up use of precision/recall/F-score
5) Work on data augmentation (**in progress**)
6) Work on more complex version of model
7) Implement cross-validation

# Ideas for direction
- Transfer learning, to utiliez informaiton learned from other datasets
- Splitting the images up in smaller images, such that we get more data
- data augmentation (mirroring, turning, )
- Sliding frame (creating an training image for each movement of the frame)


# Papers
- https://arxiv.org/pdf/1505.04597.pdf
- Crowdsourcing the creation of image segmentation algorithms for connectomics. Frontiers in neuroanatomy, Ignacio Arganda-Carreras
