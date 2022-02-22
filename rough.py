
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
# h = h.permute((2, 0, 1))
# h = h.unsqueeze(0)
# h = h.to('cuda')

# read image E
e_32 = iio.imread('/home/varun/PhD/super_resolution/KAIR/experiments/exp_14/290ep_hr65k/0813_290ep_hr65k_31.64_E_A.png')
e_32 = np.asarray(e_32)
e_32 = e_32 / 255

e_17 = iio.imread('/home/varun/PhD/super_resolution/KAIR/experiments/exp_26/10ep_hr1/0813_10ep_hr1_16.96_E.png')
e_17 = np.asarray(e_17)
e_17 = e_17 / 255

# l1 distances
print(f'L1 loss with e_32 {np.abs(e_32 - h).mean()}')
print(f'L1 loss with e_17 {np.abs(e_17 - h).mean()}')


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
