#########
import os

import numpy as np
from PyQt5.QtCore import QLibraryInfo
import cv2

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
import torch
from models.network_rrdbnet_augmentor import RRDBNET_AUG
import matplotlib.pyplot as plt
import utils.utils_image as util


# load the model
aug = RRDBNET_AUG()
state_dict = torch.load('superresolution/sraug_x4_psnr/models/35000_A.pth')
aug.load_state_dict(state_dict)


# get an image
paths = util.get_image_paths('/home/varun/sr/datasets/DIV2K/DIV2K_valid_HR')
for i in range(10):
    H_path = paths[i]
    img = util.imread_uint(H_path, 3)  # or manually set path   '/home/varun/PhD/super_resolution/vrj_data/div2k_0112.png'
    img = util.modcrop(img, 4)
    img_H = util.uint2single(img)
    # plt.imshow(img_H)

    # use matlab's bicubic downsampling
    img_L = util.imresize_np(img_H, 1 / 4, True)

    # downsample with augmentor
    imgt = util.uint2tensor4(img)
    oom = False
    try:
        with torch.no_grad():
            img_L_A = aug(imgt)
    except RuntimeError:  # Out of memory
        oom = True
    if oom:
        continue

    # convert back to uint and save
    img_L = util.single2uint(img_L)
    img_L_A = util.tensor2uint(img_L_A)

    # calculate psnr
    psnr = util.calculate_psnr(img_L, img_L_A)
    # print(psnr)

    # save image
    final = np.concatenate((img_L, img_L_A), axis=1)
    util.imwrite(final, f'aug_images/{i}_{psnr}.png')

    # lets look at both
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(img_L)
    # ax[1].imshow(img_L_A)
    # ax[0].set_xlabel('Bicubic')
    # ax[1].set_xlabel('Augmentor')
    # fig.savefig(f'aug_images/{i}.png', bbox_inches='tight')

    print(f'Saved image {i}')

# calculate psnr
# psnr = util.calculate_psnr(img_L, img_L_A)
# print(psnr)
