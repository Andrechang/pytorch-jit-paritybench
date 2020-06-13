import sys
_module = sys.modules[__name__]
del sys
app = _module
config = _module
db = _module
ml = _module
model = _module
utils = _module
trustpilot = _module
items = _module
middlewares = _module
pipelines = _module
settings = _module
spiders = _module
scraper = _module
src = _module
data_loader = _module
focal_loss = _module
model = _module
train = _module

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


import random


import torch


import torch.nn.functional as F


import torch.nn as nn


import numpy as np


from torch.autograd import Variable


from collections import Counter


from torch.utils.data import DataLoader


from torch.utils.data import WeightedRandomSampler


class CharacterLevelCNN(nn.Module):

    def __init__(self):
        super(CharacterLevelCNN, self).__init__()
        self.number_of_characters = 69
        self.extra_characters = ''
        self.alphabet = (
            'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+ =<>()[]{}'
            )
        self.max_length = 1014
        self.number_of_classes = 3
        self.dropout_input_p = 0
        self.dropout_input = nn.Dropout2d(self.dropout_input_p)
        self.conv1 = nn.Sequential(nn.Conv1d(self.number_of_characters +
            len(self.extra_characters), 256, kernel_size=7, padding=0), nn.
            ReLU(), nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7,
            padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU(), nn.MaxPool1d(3))
        input_shape = 128, self.max_length, self.number_of_characters + len(
            self.extra_characters)
        self.output_dimension = self._get_conv_output(input_shape)
        self.fc1 = nn.Sequential(nn.Linear(self.output_dimension, 1024), nn
            .ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.
            Dropout(0.5))
        self.fc3 = nn.Linear(1024, self.number_of_classes)
        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    def get_model_parameters(self):
        return {'alphabet': self.alphabet, 'extra_characters': self.
            extra_characters, 'number_of_characters': self.
            number_of_characters, 'max_length': self.max_length,
            'num_classes': self.number_of_classes}

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CharacterLevelCNN(nn.Module):

    def __init__(self, args, number_of_classes):
        super(CharacterLevelCNN, self).__init__()
        self.dropout_input = nn.Dropout2d(args.dropout_input)
        self.conv1 = nn.Sequential(nn.Conv1d(args.number_of_characters +
            len(args.extra_characters), 256, kernel_size=7, padding=0), nn.
            ReLU(), nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7,
            padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            padding=0), nn.ReLU(), nn.MaxPool1d(3))
        input_shape = 128, args.max_length, args.number_of_characters + len(
            args.extra_characters)
        self.output_dimension = self._get_conv_output(input_shape)
        self.fc1 = nn.Sequential(nn.Linear(self.output_dimension, 1024), nn
            .ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.
            Dropout(0.5))
        self.fc3 = nn.Linear(1024, number_of_classes)
        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_MarwanDebbiche_post_tuto_deployment(_paritybench_base):
    pass
