#########
import os
import glob
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QLibraryInfo
import cv2

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
import torch
from models.network_rrdbnet_augmentor import RRDBNET_AUG
from models.network_rrdbnet import RRDBNet
import matplotlib.pyplot as plt
import utils.utils_image as util


def compare_augmentor_models():

    # image paths
    paths = util.get_image_paths('/home/varun/sr/datasets/DIV2K/DIV2K_valid_HR_randSample')
    # hr_steps = {1: 40000, 4: 65000, 16: 68000, 64: 72000, 256: 76000, 1024: 80000, 4096: 84000, 16384: 88000}
    hr_steps = {'10k_hr1': 10_000, '20k_hr2': 20_000, '30k_hr3': 30_000, '40k_hr4': 40_000}
    dir = '/home/varun/sr/KAIR/aug_images'

    # the images to evaluate
    idx = range(3)  # np.random.randint(0, 10, 5)

    for hr, step in hr_steps.items():

        # load the augmentor
        model_name = f'{step}_A.pth'
        aug = RRDBNET_AUG()
        state_dict = torch.load(os.path.join('/home/varun/sr/KAIR/superresolution/sraug_x4_psnr/models', model_name))
        aug.load_state_dict(state_dict)

        # load the generator
        model_name = f'{step}_G.pth'
        gen = RRDBNet()
        state_dict = torch.load(os.path.join('/home/varun/sr/KAIR/superresolution/sraug_x4_psnr/models', model_name))
        gen.load_state_dict(state_dict)
        print('loaded model')

        # create the directory
        folder = os.path.join(dir, str(hr))
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i in idx:
            # pick a random pic
            H_path = paths[i]
            img_name = Path(H_path).stem
            img = util.imread_uint(H_path, 3)  # or manually set path   '/home/varun/PhD/super_resolution/vrj_data/div2k_0112.png'
            img = util.modcrop(img, 4)
            img_H = util.uint2single(img)

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

            # evaluate generator for bicubic and augmentor data
            oom = False
            try:
                with torch.no_grad():
                    img_E = gen(util.single2tensor4(img_L))
                    img_E_A = gen(img_L_A)
            except RuntimeError:  # Out of memory
                oom = True
            if oom:
                print(f'Failed to generate image.')
                continue

            # convert back to uint and save
            img_L = util.single2uint(img_L)
            img_L_A = util.tensor2uint(img_L_A)
            img_E = util.tensor2uint(img_E)
            img_E_A = util.tensor2uint(img_E_A)

            # calculate psnr
            psnr = round(util.calculate_psnr(img_L, img_L_A), 2)
            psnr_E = round(util.calculate_psnr(img_E, img), 2)
            psnr_E_A = round(util.calculate_psnr(img_E_A, img), 2)

            # save image
            final = np.concatenate((img_L, img_L_A), axis=1)
            file = os.path.join(folder, f'{img_name}_{hr}_{psnr}.png')
            util.imwrite(final, file)

            file = os.path.join(folder, f'{img_name}_{hr}_{psnr_E}_E.png')
            util.imwrite(img_E, file)

            file = os.path.join(folder, f'{img_name}_{hr}_{psnr_E_A}_E_A.png')
            util.imwrite(img_E_A, file)

            # save the original HR image
            # file = os.path.join(folder, f'{img_name}.png')
            # util.imwrite(img, file)

        print(f'Finished {hr}')


def compare_with_JPEG(hard_ratio, quality_factor=90):
    from utils.utils_blindsr import add_JPEG_noise
    images = glob.glob(f'/home/varun/PhD/super_resolution/aug_images/{hard_ratio}/*')
    psnrs = []
    for image in images:
        img_name = os.path.basename(image)
        img = util.imread_uint(image, 3)
        img_L = img[:, :(img.shape[1] // 2), :]
        img_J = add_JPEG_noise(util.uint2single(img_L), quality_factor)
        psnr = round(util.calculate_psnr(img_L, util.single2uint(img_J)), 2)
        psnrs.append(psnr)
        print(f'{img_name}: {psnr}')
    print(f'Average psnr: {sum(psnrs) / len(psnrs)}')


#  def test_generator():


if __name__ == '__main__':
    compare_augmentor_models()
