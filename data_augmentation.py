"""
This script performs a selection of data augmentation types, e.g. mirroring the image.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates


def mirror_input(input, dim: int):
    if not isinstance(dim, int):
        raise TypeError(
            "dimension argument must be an integer: either 0 (along the horizontal axis) or 1 (along the vertical axis)",
        )

    return torch.flip(input, [dim])


def mirror_horizontal(input):
    # return mirror_input(input, 0)
    return np.fliplr(input)


def mirror_vertical(input):
    return mirror_input(input, 1)


def elastic_transform(image, alpha, sigma, seed):
    """
    Parameters:
        image (numpy.ndarray): The image to be transformed.
        alpha: Scale of the transformation. Controls the intensity of the deformation.
        sigma: Standard deviation of the Gaussian filter - determines the smoothness of the deformation.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: The deformed image.
    """
    random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing="ij")
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1).reshape(shape)
    return distorted_image


if __name__ == "__main__":
    # Testing:
    x = torch.arange(8).view(4, 2)
    # print(x)
    # print(mirror_horizontal(x))
