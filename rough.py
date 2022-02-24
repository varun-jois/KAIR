
import imageio as iio
import numpy as np
import torch
from models.loss import PerceptualLoss

######################
# For GPU
######################

# read an image
h = iio.imread('/home/varun/sr/datasets/practice/0813_44.png')
h = np.asarray(h)
h = h / 255
b, h = h[:, :(h.shape[1] // 2)], h[:, (h.shape[1] // 2):]
b = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to('cuda')
h = torch.from_numpy(h).permute(2, 0, 1).unsqueeze(0).to('cuda')

# read image l
l = iio.imread('/home/varun/sr/datasets/practice/0813_30.png')
l = np.asarray(l)
l = l / 255
_, l = l[:, :(l.shape[1] // 2)], l[:, (l.shape[1] // 2):]
l = torch.from_numpy(l).permute(2, 0, 1).unsqueeze(0).to('cuda')

# load perceptual loss
pl = PerceptualLoss(feature_layer=35).to('cuda')

# L1 loss
print(f'Perceptual loss with psnr 44 : {pl(b, h).item()}')
print(f'L1 loss with zeros {np.abs(np.zeros(h.shape) - h).mean()}')
print(f'L1 loss with ones {np.abs(np.ones(h.shape) - h).mean()}')

######################
# For my laptop
######################

# read H
h = iio.imread('/home/varun/PhD/super_resolution/datasets/DIV2K_valid_HR_randSample/0813.png')
h = np.asarray(h)
h = h / 255

# read E
e = iio.imread('/home/varun/PhD/super_resolution/KAIR/experiments/exp_30/50ep_hr1/0813_50ep_hr1_26.2_E.png')
e = np.asarray(e)
e = e / 255

# read e_a
e_a = iio.imread('/home/varun/PhD/super_resolution/KAIR/experiments/exp_30/50ep_hr1/0813_50ep_hr1_25.82_E_A.png')
e_a = np.asarray(e_a)
e_a = e_a / 255

print(f'L1 loss with e {np.abs(e - h).mean()}')
print(f'L1 loss with e_a {np.abs(e_a - h).mean()}')


