
import imageio as iio
import numpy as np
import torch
from models.loss import PerceptualLoss

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
