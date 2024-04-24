import numpy as np
import os
#import optuna
import torch
import sys
import matplotlib.pyplot as plt

import torch.nn as nn
from einops import rearrange

sys.path.append('../')
from hf_diffusion import SinusoidalPositionEmbeddings

class model_o3_err(nn.Module):
    #time-dependent version of model_o3_err
    def __init__(self, hidden, dr, channels, dim=64, time_embed_dim=256):
        super(model_o3_err, self).__init__()
        #time_embedding common to all blocks
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # input: 1x256x256 ---------------> output: (2*hidden)x128x128
        self.C01 = nn.Conv2d(channels, 2 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.time_dep0 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2*hidden)) #TODO: ADD einsum here to go from b c to b c 1 1 
        self.B01 = nn.BatchNorm2d(2 * hidden)
        self.B02 = nn.BatchNorm2d(2 * hidden)
        self.B03 = nn.BatchNorm2d(2 * hidden)

        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(2 * hidden, 4 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4 * hidden, 4 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4 * hidden, 4 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.time_dep1 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 4*hidden))
        self.B11 = nn.BatchNorm2d(4 * hidden)
        self.B12 = nn.BatchNorm2d(4 * hidden)
        self.B13 = nn.BatchNorm2d(4 * hidden)

        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(4 * hidden, 8 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8 * hidden, 8 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8 * hidden, 8 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.time_dep2 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 8*hidden))
        self.B21 = nn.BatchNorm2d(8 * hidden)
        self.B22 = nn.BatchNorm2d(8 * hidden)
        self.B23 = nn.BatchNorm2d(8 * hidden)

        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8 * hidden, 16 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16 * hidden, 16 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16 * hidden, 16 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.time_dep3 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 16*hidden))
        self.B31 = nn.BatchNorm2d(16 * hidden)
        self.B32 = nn.BatchNorm2d(16 * hidden)
        self.B33 = nn.BatchNorm2d(16 * hidden)

        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16 * hidden, 32 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32 * hidden, 32 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32 * hidden, 32 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.time_dep4 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 32*hidden))
        self.B41 = nn.BatchNorm2d(32 * hidden)
        self.B42 = nn.BatchNorm2d(32 * hidden)
        self.B43 = nn.BatchNorm2d(32 * hidden)

        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(32 * hidden, 64 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64 * hidden, 64 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64 * hidden, 64 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.time_dep5 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 64 * hidden))
        self.B51 = nn.BatchNorm2d(64 * hidden)
        self.B52 = nn.BatchNorm2d(64 * hidden)
        self.B53 = nn.BatchNorm2d(64 * hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64 * hidden, 128 * hidden, kernel_size=4, stride=1, padding=0,
                             padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(128 * hidden)

        self.P0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1 = nn.Linear(128 * hidden, 64 * hidden)
        self.FC2 = nn.Linear(64 * hidden, 12)

        self.dropout = nn.Dropout(p=dr)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, image, time):
        time_embedded = self.time_mlp(time)
        #print(time.shape, time_embedded.shape)
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x += rearrange(self.time_dep0(time_embedded), "b c -> b c 1 1")
        x = self.LeakyReLU(self.B03(self.C03(x)))


        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x += rearrange(self.time_dep1(time_embedded), "b c -> b c 1 1")
        x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x += rearrange(self.time_dep2(time_embedded), "b c -> b c 1 1")
        x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x += rearrange(self.time_dep3(time_embedded), "b c -> b c 1 1")
        x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x += rearrange(self.time_dep4(time_embedded), "b c -> b c 1 1")
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x += rearrange(self.time_dep5(time_embedded), "b c -> b c 1 1")
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0], -1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        #the model returns 12 values (regardless of whether you actually want to train to predict the astro etc)
        #subsetting parameters must be handled externally
        y = torch.clone(x)
        y[:, 6:12] = torch.square(x[:, 6:12])
        return y


if __name__=='__main__':
    tinp = torch.arange(20)
    sinu = SinusoidalPositionEmbeddings(64)
    embtinp = sinu(tinp)
    '''
    plt.figure()
    plt.plot(tinp.numpy()[:], embtinp)
    plt.xlabel('t')
    plt.ylabel('Sinu(t)')'''
    print(3)
