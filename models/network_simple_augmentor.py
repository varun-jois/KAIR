
import torch
import torch.nn as nn


class Simple_AUG(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=5):
        super(Simple_AUG, self).__init__()
        self.c1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.c3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.c4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.c5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.c6 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        fea = self.lrelu(self.c1(x))
        fea = self.lrelu(self.c2(fea))
        fea = self.mp(fea)
        fea = self.lrelu(self.c3(fea))
        fea = self.lrelu(self.c4(fea))
        fea = self.mp(fea)
        fea = self.lrelu(self.c5(fea))
        out = self.c6(fea)
        return out
