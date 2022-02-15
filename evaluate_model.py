"""
Runs the trained model on the testsets and retrieves the average PSNR scores.

Author: Varun
"""

#########
import os
import glob
from pathlib import Path
import numpy as np
import sys
import cv2
import torch
from models.network_rrdbnet_augmentor import RRDBNET_AUG
from models.network_rrdbnet import RRDBNet
import utils.utils_image as util
from utils import utils_option as option

TEST_DIR = '/home/varun/sr/datasets'
MODEL_DIR = '/home/varun/sr/KAIR/superresolution'
TESTSETS = ['Set5', 'Set14', 'B100', 'Urban100']


def main(model_names):
    device = torch.device('cuda')
    results = { for m in model_names for metric in ['']}

    for m in model_names:

        # load model
        if 'rrdb' in m:
            gen = RRDBNet()
        else:
            gen = RRDBNet()
        gen = gen.to(device)
        init_iter_G, init_path_G = option.find_last_checkpoint(os.path.join(MODEL_DIR, m, 'models'), net_type='G')
        gen.load_state_dict(torch.load(init_path_G), strict=True)
        gen.eval()




        print('loaded model')





    pass


if __name__ == '__main__':
    model_names = [sys.argv[i] for i in range(1, len(sys.argv))]
    main(model_names)
