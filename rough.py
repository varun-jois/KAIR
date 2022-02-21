
import imageio as iio
import numpy as np
import torch
import math


def loss1(lea, hr):
    return lea + abs(1 - np.exp(lea - hr * .03))


def loss2(lea, hr):
    return abs(1 - np.exp(lea - hr * .03))


# read an image
h = iio.imread('/home/varun/PhD/super_resolution/datasets/DIV2K_valid_HR_randSample/0813.png')
h = np.asarray(h)
h = h / 255

# create noise
noise = lambda h: np.random.randint(0, 256, size=h.shape) / 255

# L1 loss
print(f'L1 loss with noise {np.abs(noise(h) - h).mean()}')
print(f'L1 loss with zeros {np.abs(np.zeros(h.shape) - h).mean()}')
print(f'L1 loss with ones {np.abs(np.ones(h.shape) - h).mean()}')
