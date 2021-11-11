'''
Augmentor module for SRAUG.

Author: Varun Jois
'''

import functools
import torch.nn as nn
from models.network_rrdbnet import make_layer, RRDB


class RRDBNET_AUG(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, gc=32, nb1=1, nb2=1, nb3=10):
        '''
        The augmentor used for creating augmented data.
        Parameters
        ----------
        in_nc : Number of channels for the input.
        out_nc : Number of channels for the output.
        nf : Number of convolution features (kernels) for the RRDB block.
        gc : The growth of the number of channels within the RRDB block.
        nb1 : The number of RRDB blocks before the first maxpool downsampling.
        nb2 : The number of RRDB blocks before the second maxpool downsampling.
        nb3 : The number of RRDB blocks before the after the last maxpool downsampling.
        '''
        super(RRDBNET_AUG, self).__init__()
        RRDB_block = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_1 = make_layer(RRDB_block, nb1)
        self.RRDB_1_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.RRDB_2 = make_layer(RRDB_block, nb2)
        self.RRDB_2_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.RRDB_3 = make_layer(RRDB_block, nb3)
        self.RRDB_3_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        body = self.RRDB_1_conv(self.RRDB_1(fea))
        fea = self.lrelu(fea + body)

        # downsample
        fea = self.mp(fea)
        body = self.RRDB_2_conv(self.RRDB_2(fea))
        fea = self.lrelu(fea + body)

        # downsample
        fea = self.mp(fea)
        body = self.RRDB_3_conv(self.RRDB_3(fea))
        fea = self.lrelu(fea + body)

        out = self.conv_last(fea)

        return out
