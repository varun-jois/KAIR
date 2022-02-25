
import imageio as iio
import numpy as np
import torch
from models.loss import PerceptualLoss
from torchvision.transforms.functional import center_crop
from torchvision.utils import save_image
from models import model_plain_aug
from models.network_rrdbnet_augmentor import RRDBNET_AUG
from models.network_rrdbnet import RRDBNet

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


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
l = iio.imread('/home/varun/sr/datasets/practice/0813_5.png')
l = np.asarray(l)
l = l / 255
_, l = l[:, :(l.shape[1] // 2)], l[:, (l.shape[1] // 2):]
l = torch.from_numpy(l).permute(2, 0, 1).unsqueeze(0).to('cuda')

# read hr image
hr = iio.imread('/home/varun/sr/datasets/practice/0813.png')
hr = np.asarray(hr)
hr = hr / 255
hr = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0).to('cuda')

# load models
aug = RRDBNET_AUG()
aug = aug.to('cuda')
state_dict = torch.load('/home/varun/sr/KAIR/superresolution/aug_x4_rrdb/models/20000_A.pth')
aug.load_state_dict(state_dict)

# load the generator
gen = RRDBNet()
gen = gen.to('cuda')
state_dict = torch.load('/home/varun/sr/KAIR/superresolution/aug_x4_rrdb/models/20000_G.pth')
gen.load_state_dict(state_dict)

l_a = aug(hr)

# load perceptual loss
pl = PerceptualLoss(feature_layer=34).to('cuda')

# L1 loss
print(f'Perceptual loss with psnr 44 : {pl(b.float(), h.float()).item()}')
print(f'Perceptual loss with psnr 30 : {pl(b.float(), l.float()).item()}')
print(f'L1 loss with zeros {np.abs(np.zeros(h.shape) - h).mean()}')
print(f'L1 loss with ones {np.abs(np.ones(h.shape) - h).mean()}')

######################
# For my laptop
######################

# read H
h = iio.imread('/home/varun/PhD/super_resolution/datasets/DIV2K_valid_HR_randSample/0813.png')
h = np.asarray(h)
h = h / 255

# read small
e = iio.imread('/home/varun/PhD/super_resolution/KAIR/experiments/exp_30/20ep_hr1/0813_20ep_hr1_34.26.png')
e = np.asarray(e)
e = e / 255
e = torch.from_numpy(e).permute(2, 0, 1).unsqueeze(0)

# find centre
height

print(f'L1 loss with e {np.abs(e - h).mean()}')
print(f'L1 loss with e_a {np.abs(e_a - h).mean()}')


