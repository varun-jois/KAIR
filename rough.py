
import imageio as iio
import numpy as np
import torch
import math
from models.loss import PerceptualLoss


def loss1(lea, hr):
    return lea + abs(1 - np.exp(lea - hr * .03))


def loss2(lea, hr):
    return abs(1 - np.exp(lea - hr * .03))


# read an image
h = iio.imread('/home/varun/PhD/super_resolution/datasets/DIV2K_valid_HR_randSample/0813.png')
h = np.asarray(h)
h = h / 255
h = torch.from_numpy(h)
h = h.permute((2, 0, 1))
h = h.unsqueeze(0)
h = h.to('cuda')

# create noise
#noise = lambda h: np.random.randint(0, 256, size=h.shape) / 255
noise = torch.randint_like(h, low=0, high=256) / 255
noise = noise.to('cuda')

# L1 loss
print(f'L1 loss with noise {np.abs(noise(h) - h).mean()}')
print(f'L1 loss with zeros {np.abs(np.zeros(h.shape) - h).mean()}')
print(f'L1 loss with ones {np.abs(np.ones(h.shape) - h).mean()}')

pl = PerceptualLoss(feature_layer=34).to('cuda')
loss = pl(noise, h)
