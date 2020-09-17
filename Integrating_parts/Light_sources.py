from os.path import join

import PIL
import numpy
from matplotlib import patches

try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def non_max_suppression(dim, result, c_image, color):
    x = []
    y = []
    image_height, image_width = result.shape[:2]
    for i in range(0, image_height - dim, dim):
        for j in range(0, image_width - dim, dim):
            index = np.argmax(result[i:i + dim, j:j + dim])
            a = np.amax(result[i:i + dim, j:j + dim])
            if a > 80:
                x.append(index // dim + i)
                y.append(index % dim + j)
                c_image[index // dim + i, index % dim + j] = color
    return x, y


def red_picture(im_red, filter_kernel, c_image):
    grad = sg.convolve2d(im_red, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    x_red, y_red = non_max_suppression(14, result, c_image, [255, 0, 0])
    return x_red, y_red



def green_picture(im_green, filter_kernel, c_image):
    grad = sg.convolve2d(im_green, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    x_green, y_green = non_max_suppression(18, result, c_image, [0, 255, 0])
    return x_green, y_green


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    c_image = np.array(Image.open(c_image))

    im_red = c_image[:, :, 0]
    im_green = c_image[:, :, 1]
    filter_kernel = [
        [-1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225,
         -1 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225,
         -1 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, 11 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225, -2 / 225,
         -2 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225,
         -1 / 225, -1 / 225, -1 / 225, -1 / 225],
        [-1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225, 1 / 225, -1 / 225, -1 / 225, -1 / 225, -1 / 225,
         -1 / 225, -1 / 225, -1 / 225, -1 / 225]]
    x_red, y_red = red_picture(im_red, filter_kernel, c_image)
    x_green, y_green = green_picture(im_green, filter_kernel, c_image)

    return x_red, y_red, x_green, y_green