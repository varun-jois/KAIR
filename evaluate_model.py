"""
Runs the trained model on the testsets and retrieves the average PSNR scores.

Author: Varun
"""

#########
import os
import pandas as pd
from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
import sys
import torch
from models.network_rrdbnet import RRDBNet
import utils.utils_image as util
from utils import utils_option as option
from datetime import datetime


TEST_DIR = '/home/varun/sr/datasets'
MODEL_DIR = '/home/varun/sr/KAIR/superresolution'
TESTSETS = ['Set5', 'Set14', 'B100', 'Urban100', 'RealSR']  # ['RealSR']
METRICS = ['psnr', 'ssim']


def main(model_names):
    device = torch.device('cuda')
    results = {m: {t: {x: 0 for x in METRICS} for t in TESTSETS} for m in model_names}
    scores = []

    for m in model_names:
        print(f'Starting testing for {m}')

        # load model
        if 'rrdb' in m:
            gen = RRDBNet()
        else:
            gen = RRDBNet()
        gen = gen.to(device)
        # try to get the E model other wise G
        # _, init_path = option.find_last_checkpoint(os.path.join(MODEL_DIR, m, 'models'), net_type='E')
        # if init_path is None:
        _, init_path = option.find_last_checkpoint(os.path.join(MODEL_DIR, m, 'models'), net_type='G')
        print('Loaded the G model')
        gen.load_state_dict(torch.load(init_path), strict=True)
        gen.eval()

        # get scaling factor
        sf = m.split('_')[1].upper()
        border = int(sf[-1])  # shave boader to calculate PSNR and SSIM

        for t in TESTSETS:
            # the paths for the images
            if t == 'RealSR':
                L_paths = util.get_image_paths(f'/home/varun/sr/datasets/RealSR_V3/test/LR/{sf}')
            else:
                L_paths = util.get_image_paths(os.path.join(TEST_DIR, t, 'LR_bicubic', sf))
                H_path_dir = os.path.join(TEST_DIR, t, 'HR')

            # counters for the metrics
            psnr_total = 0
            ssim_total = 0

            for img in L_paths:
                name, _ = os.path.splitext(os.path.basename(img))
                img_L = util.imread_uint(img, 3)
                imgt = util.uint2tensor4(img_L).to(device)

                # run it through the model
                oom = False
                try:
                    with torch.no_grad():
                        img_E = gen(imgt)
                except RuntimeError:  # Out of memory
                    oom = True
                if oom:
                    print(f'Failed to generate image.')
                    raise
                img_E = util.tensor2uint(img_E)

                # load the HR image
                if t == 'RealSR':
                    img_H_path = f'/home/varun/sr/datasets/RealSR_V3/test/HR/{sf}/{name[:-3]}HR.png'
                else:
                    img_H_path = os.path.join(H_path_dir, f'{name[:-2]}.png')
                img_H = util.imread_uint(img_H_path, 3)
                img_H = util.modcrop(img_H, int(sf[-1]))

                # calculate the metrics
                psnr = util.calculate_psnr(img_E, img_H, border=border)
                psnr_total += psnr
                ssim_total += util.calculate_ssim(img_E, img_H, border=border)

                # store the individual psnr scores
                scores.append({'model': m, 'image': img_H_path, 'psnr': psnr})

            # store the results
            results[m][t]['psnr'] = psnr_total / len(L_paths)
            results[m][t]['ssim'] = ssim_total / len(L_paths)
            print(f'\t\tFinished {t}')

    # save the results
    df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in results.items()}, axis=1).T
    now = datetime.now()
    file_name = now.strftime('%y%m%d_%H%M%S') + '.csv'
    df.to_csv(f'/home/varun/sr/KAIR/results/{file_name}')

    df = pd.DataFrame(scores)
    df.sort_values(by=['model', 'psnr'], inplace=True)
    df.to_csv('/home/varun/sr/KAIR/results/psnr_scores.csv', index=False)


if __name__ == '__main__':
    model_names = [sys.argv[i] for i in range(1, len(sys.argv))]
    main(model_names)
