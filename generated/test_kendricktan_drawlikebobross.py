import sys
_module = sys.modules[__name__]
del sys
aae = _module
loader = _module
models = _module
train = _module
trainer = _module
app = _module
preprocess = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


import random


import torch.nn.functional as F


import numpy as np


class Encoder(nn.Module):

    def __init__(self, z_dim, h_dim=128, filter_num=64, channel_num=3):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channel_num, filter_num, 4, 2, 
            1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            filter_num, filter_num * 2, 4, 2, 1, bias=False), nn.
            BatchNorm2d(filter_num * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_num * 2, filter_num * 2, 8, 4, 1, bias=False),
            nn.BatchNorm2d(filter_num * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_num * 2, filter_num, 8, 4, 1, bias=False), nn.
            BatchNorm2d(filter_num), nn.LeakyReLU(0.2, inplace=True))
        self.fc = nn.Sequential(nn.Linear(filter_num * 3 * 3, h_dim), nn.
            ReLU(), nn.Linear(h_dim, z_dim))
        self.z_dim = z_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self, z_dim, filter_num=64, channel_num=3):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(z_dim, filter_num * 8,
            8, 1, 0, bias=False), nn.BatchNorm2d(filter_num * 8), nn.ReLU(
            True), nn.ConvTranspose2d(filter_num * 8, filter_num * 4, 8, 4,
            2, bias=False), nn.BatchNorm2d(filter_num * 4), nn.ReLU(True),
            nn.ConvTranspose2d(filter_num * 4, filter_num * 2, 4, 2, 1,
            bias=False), nn.BatchNorm2d(filter_num * 2), nn.ReLU(True), nn.
            ConvTranspose2d(filter_num * 2, filter_num, 4, 2, 1, bias=False
            ), nn.BatchNorm2d(filter_num), nn.ReLU(True), nn.
            ConvTranspose2d(filter_num, channel_num, 4, 2, 1, bias=False),
            nn.Sigmoid())
        self.z_dim = z_dim

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.
            Linear(z_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.fc(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_dim, h_dim=128, filter_num=64, channel_num=3):
        super(Generator, self).__init__()
        encoder = Encoder(z_dim, h_dim, filter_num, channel_num)
        decoder = Decoder(z_dim, filter_num, channel_num)
        self.conv = encoder.conv
        self.fc = encoder.fc
        self.conv_t = decoder.conv

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv_t(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kendricktan_drawlikebobross(_paritybench_base):
    pass
    def test_000(self):
        self._check(Decoder(*[], **{'z_dim': 4}), [torch.rand([4, 4])], {})

    def test_001(self):
        self._check(Discriminator(*[], **{'z_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

