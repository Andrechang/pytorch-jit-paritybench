import sys
_module = sys.modules[__name__]
del sys
_ext = _module
stnm = _module
functions = _module
gridgen = _module
modules = _module
gridgen = _module
stnm = _module
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


from torch.nn.modules.module import Module


import torch


from torch.autograd import Variable


import numpy as np


import random


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


class AffineGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = torch.Tensor(self.height, self.width, 3)
        for i in range(self.height):
            self.grid.select(2, 0).select(0, i).fill_(-1 + float(i) / (self
                .height - 1) * 2)
        for j in range(self.width):
            self.grid.select(2, 1).select(1, j).fill_(-1 + float(j) / (self
                .width - 1) * 2)
        self.grid.select(2, 2).fill_(1)

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()
        batchgrid_temp = self.batchgrid.view(-1, self.height * self.width, 3)
        batchgrid_temp.contiguous()
        input_temp = torch.transpose(input1, 1, 2)
        input_temp.contiguous()
        output_temp = torch.bmm(batchgrid_temp, input_temp)
        output = output_temp.view(-1, self.height, self.width, 2)
        output.contiguous()
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            grad_input1 = grad_input1.cuda()
        grad_output_temp = grad_output.contiguous()
        grad_output_view = grad_output_temp.view(-1, self.height * self.
            width, 2)
        grad_output_view.contiguous()
        grad_output_temp = torch.transpose(grad_output_view, 1, 2)
        grad_output_temp.contiguous()
        batchgrid_temp = self.batchgrid.view(-1, self.height * self.width, 3)
        batchgrid_temp.contiguous()
        grad_input1 = torch.baddbmm(grad_input1, grad_output_temp,
            batchgrid_temp)
        return grad_input1


class AffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            identity = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]],
                dtype=np.float32))
            batch_identity = torch.zeros([input.size(0), 2, 3])
            for i in range(input.size(0)):
                batch_identity[i] = identity
            batch_identity = Variable(batch_identity)
            loss = torch.mul(input - batch_identity, input - batch_identity)
            loss = torch.sum(loss, 1)
            loss = torch.sum(loss, 2)
            return self.f(input), loss.view(-1, 1)


class CylinderGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = (1 + torch.cos(input1)) / 2
        output = torch.zeros(torch.Size([input1.size(0), self.height, self.
            width, 2]))
        if not self.input1.is_cuda:
            for i in range(self.input1.size(0)):
                x = self.input1[i][0]
                low = int(np.ceil(self.width * self.input1[i][0]))
                frac = self.width * self.input1[i][0] - low
                interp = frac * 2 * (1 - x) + (1 - frac) * 2 * -x
                output[(i), :, :, (1)] = torch.zeros(self.grid[:, :, (1)].
                    size())
                if low <= self.width and low > 0:
                    output[(i), :, :low, (1)].fill_(2 * (1 - x))
                if low < self.width and low >= 0:
                    output[(i), :, low:, (1)].fill_(2 * -x)
                output[(i), :, :, (1)] = output[(i), :, :, (1)] + self.grid[:,
                    :, (1)]
                output[(i), :, :, (0)] = self.grid[:, :, (0)]
        else:
            print('not implemented')
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        if not grad_output.is_cuda:
            for i in range(self.input1.size(0)):
                grad_input1[i] = -torch.sum(torch.sum(grad_output[(i), :, :,
                    (1)], 1)) * torch.sin(self.input1[i]) / 2
        else:
            print('not implemented')
        return grad_input1 * self.lr


class CylinderGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(CylinderGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = CylinderGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            return self.f(input), torch.mul(input, input).view(-1, 1)


class AffineGridGenV2(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(AffineGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if input1.is_cuda:
            self.batchgrid = self.batchgrid
        output = torch.bmm(self.batchgrid.view(-1, self.height * self.width,
            3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.
            width, 2)
        return output


class CylinderGridGenV2(Module):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.
            grid.size())
        for i in range(input.size(0)):
            self.batchgrid[(i), :, :, :] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        input_u = input.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output0 = self.batchgrid[:, :, :, 0:1]
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (self.batchgrid[:, :,
            :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (
            np.pi / 2)
        output = torch.cat([output0, output1], 3)
        return output


class DenseAffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.mul(self.batchgrid, input1[:, :, :, 0:3])
        y = torch.mul(self.batchgrid, input1[:, :, :, 3:6])
        output = torch.cat([torch.sum(x, 3), torch.sum(y, 3)], 3)
        return output


class DenseAffine3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class DenseAffine3DGridGen_rotate(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen_rotate, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1, input2):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = input2.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen_with_mask(Module):

    def __init__(self, height, width, lr=1, aux_loss=False, ray_tracing=False):
        super(Depth3DGridGen_with_mask, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.ray_tracing = ray_tracing
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if depth.is_cuda:
            self.batchgrid = self.batchgrid
            self.batchgrid3d = self.batchgrid3d
        x_ = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y_ = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        rotate_z = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.
            width, 1) * np.pi
        x = x_ * torch.cos(rotate_z) - y_ * torch.sin(rotate_z)
        y = x_ * torch.sin(rotate_z) + y_ * torch.cos(rotate_z)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        if depth.is_cuda:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                cuda.FloatTensor) * (y.ge(0).type(torch.cuda.FloatTensor) -
                y.lt(0).type(torch.cuda.FloatTensor))
        else:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).
                type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class STNMFunction(Function):

    def forward(self, canvas, fgimg, fggrid, fgmask):
        self.canvas = canvas
        self.fgimg = fgimg
        self.fggrid = fggrid
        self.fgmask = fgmask
        output = torch.zeros(canvas.size()[0], canvas.size()[1], canvas.
            size()[2], canvas.size()[3])
        if not canvas.is_cuda:
            print('only support cuda now!')
        else:
            output = output.cuda()
            stnm.BilinearSamplerBHWD_updateOutput_cuda(canvas, fgimg,
                fggrid, fgmask, output)
        return output

    def backward(self, grad_output):
        grad_canvas = torch.zeros(self.canvas.size())
        grad_fgimg = torch.zeros(self.fgimg.size())
        grad_fggrid = torch.zeros(self.fggrid.size())
        grad_fgmask = torch.zeros(self.fgmask.size())
        if not grad_output.is_cuda:
            print('only support cuda now!')
        else:
            grad_output = grad_output.contiguous()
            grad_canvas = grad_canvas.cuda().contiguous()
            grad_fgimg = grad_fgimg.cuda().contiguous()
            grad_fggrid = grad_fggrid.cuda().contiguous()
            grad_fgmask = grad_fgmask.cuda().contiguous()
            stnm.BilinearSamplerBHWD_updateGradInput_cuda(self.canvas, self
                .fgimg, self.fggrid, self.fgmask, grad_canvas, grad_fgimg,
                grad_fggrid, grad_fgmask, grad_output)
        return grad_canvas, grad_fgimg, grad_fggrid, grad_fgmask


class STNM(Module):

    def __init__(self):
        super(STNM, self).__init__()
        self.f = STNMFunction()

    def forward(self, canvas, fgimg, fggrid, fgmask):
        return self.f(canvas, fgimg, fggrid, fgmask)


parser = argparse.ArgumentParser()


opt = parser.parse_args()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jwyang_lr_gan_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Depth3DGridGen(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Depth3DGridGen_with_mask(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

