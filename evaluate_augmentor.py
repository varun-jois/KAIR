#########
import os
import glob
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QLibraryInfo
import sys
import cv2

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
import torch
from models.network_rrdbnet_augmentor import RRDBNET_AUG
from models.network_rrdbnet import RRDBNet
import matplotlib.pyplot as plt
import utils.utils_image as util
from utils import utils_option as option
import glob


def compare_augmentor_models(device, model_name):

    # image paths
    paths = util.get_image_paths('/home/varun/sr/datasets/DIV2K/DIV2K_valid_HR_randSample')
    # hr_steps = {1: 40000, 4: 65000, 16: 68000, 64: 72000, 256: 76000, 1024: 80000, 4096: 84000, 16384: 88000}
    hr_steps = {'25ep_hr8': 5_000, '50ep_hr8': 10_000, '75ep_hr16': 15_000, '100ep_hr16': 20_000}
    dir = '/home/varun/sr/KAIR/aug_images'
    model_dir = f'/home/varun/sr/KAIR/superresolution/{model_name}/models'

    # the images to evaluate
    idx = range(3)  # np.random.randint(0, 10, 5)

    for hr, step in hr_steps.items():

        # load the augmentor
        model_name = f'{step}_A.pth'
        aug = RRDBNET_AUG()
        aug = aug.to(device)
        state_dict = torch.load(os.path.join(model_dir, model_name))
        aug.load_state_dict(state_dict)

        # load the generator
        model_name = f'{step}_G.pth'
        gen = RRDBNet()
        gen = gen.to(device)
        state_dict = torch.load(os.path.join(model_dir, model_name))
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
            imgt = util.uint2tensor4(img).to(device)
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
                    img_E = gen(util.single2tensor4(img_L).to(device))
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


def test_augmentor(device, model_name):
    # image paths
    paths = util.get_image_paths('/home/varun/sr/datasets/DIV2K/DIV2K_valid_HR_randSample')
    dir = '/home/varun/sr/KAIR/aug_images'
    model_dir = f'/home/varun/sr/KAIR/superresolution/{model_name}/models'

    # load the aug model
    init_iter_A, init_path_A = option.find_last_checkpoint(model_dir, net_type='A')
    aug = RRDBNET_AUG()
    aug = aug.to(device)
    state_dict = torch.load(init_path_A)
    aug.load_state_dict(state_dict)
    aug.eval()
    print('loaded augmentor')

    # load the gen model
    init_iter_G, init_path_G = option.find_last_checkpoint(model_dir, net_type='G')
    gen = RRDBNet()
    gen = gen.to(device)
    state_dict = torch.load(init_path_G)
    gen.load_state_dict(state_dict)
    gen.eval()
    print('loaded generator')

    # the images to evaluate
    idx = range(10)  # np.random.randint(0, 10, 5)

    for i in idx:
        # pick a random pic
        H_path = paths[i]
        img_name = Path(H_path).stem
        img = util.imread_uint(H_path,
                               3)  # or manually set path   '/home/varun/PhD/super_resolution/vrj_data/div2k_0112.png'
        img = util.modcrop(img, 4)
        img_H = util.uint2single(img)

        # use matlab's bicubic downsampling
        img_L = util.imresize_np(img_H, 1 / 4, True)

        # downsample with augmentor
        imgt = util.uint2tensor4(img).to(device)
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
                img_E = gen(util.single2tensor4(img_L).to(device))
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
        file = os.path.join(dir, f'{img_name}_{psnr}.png')
        util.imwrite(final, file)

        file = os.path.join(dir, f'{img_name}_{psnr_E}_E.png')
        util.imwrite(img_E, file)

        file = os.path.join(dir, f'{img_name}_{psnr_E_A}_E_A.png')
        util.imwrite(img_E_A, file)

        # save the original HR image
        # file = os.path.join(folder, f'{img_name}.png')
        # util.imwrite(img, file)


def test_generator(device, model_name):
    # image paths
    #paths = util.get_image_paths('/home/varun/sr/datasets/DIV2K/DIV2K_valid_HR_randSample')
    paths = util.get_image_paths('/home/varun/sr/datasets/misc')
    dir = '/home/varun/sr/KAIR/gen_images'
    model_dir = f'/home/varun/sr/KAIR/superresolution/{model_name}/models'

    # load the model
    # try to get the E model otherwise G
    _, init_path = option.find_last_checkpoint(model_dir, net_type='E')
    if init_path is None:
        _, init_path = option.find_last_checkpoint(model_dir, net_type='G')
        print('Loaded the G model')
    gen = RRDBNet()
    gen = gen.to(device)
    state_dict = torch.load(init_path)
    gen.load_state_dict(state_dict)
    gen.eval()
    print('loaded model')

    # the images to evaluate
    idx = range(3)  # np.random.randint(0, 10, 5)

    for i in idx:
        # pick a random pic
        H_path = paths[i]
        img_name = Path(H_path).stem
        img = util.imread_uint(H_path,
                               3)  # or manually set path   '/home/varun/PhD/super_resolution/vrj_data/div2k_0112.png'
        img = util.modcrop(img, 4)
        img_H = util.uint2single(img)

        # use matlab's bicubic downsampling
        img_L = util.imresize_np(img_H, 1 / 4, True)

        # get the output of bicubic upsampling
        img_B = util.imresize_np(img_L, 4, True)

        # evaluate generator for bicubic and augmentor data
        oom = False
        try:
            with torch.no_grad():
                img_E = gen(util.single2tensor4(img_L).to(device))
        except RuntimeError:  # Out of memory
            oom = True
        if oom:
            print(f'Failed to generate image.')
            continue

        # convert back to uint and save
        img_L = util.single2uint(img_L)
        img_B = util.single2uint(img_B)
        img_E = util.tensor2uint(img_E)

        # calculate psnr
        psnr_E = round(util.calculate_psnr(img_E, img), 2)
        psnr_B = round(util.calculate_psnr(img_B, img), 2)

        # save image
        file = os.path.join(dir, f'{img_name}_L.png')
        util.imwrite(img_L, file)

        # one file with all three
        file = os.path.join(dir, f'{img_name}_BEH.png')
        three = np.concatenate((img_B, img_E, img), axis=1)
        util.imwrite(three, file)

        file = os.path.join(dir, f'{img_name}_{psnr_B}_B.png')
        util.imwrite(img_B, file)

        file = os.path.join(dir, f'{img_name}_{psnr_E}_E.png')
        util.imwrite(img_E, file)

        file = os.path.join(dir, f'{img_name}_H.png')
        util.imwrite(img, file)

        # save the original HR image
        # file = os.path.join(folder, f'{img_name}.png')
        # util.imwrite(img, file)

    print(f'Finished')


if __name__ == '__main__':
    mode = sys.argv[1]
    model_name = sys.argv[2]
    device = torch.device('cuda')
    if mode == '0':
        test_augmentor(device, model_name)
    elif mode == '1':
        test_generator(device, model_name)
    elif mode == '3':
        compare_augmentor_models(device, model_name)
    else:
        raise ValueError
