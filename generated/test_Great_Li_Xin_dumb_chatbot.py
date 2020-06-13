import sys
_module = sys.modules[__name__]
del sys
chatbot = _module
checkpoint_selector = _module
custom_token = _module
data_utils = _module
masked_cross_entropy = _module
model = _module
model_utils = _module
prerequisites = _module
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


import torch


from torch.nn import functional


import random


import torch.nn as nn


import torch.nn.functional as func


import torch.nn.init as weight_init


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import math


from torch import optim


from torch.nn.utils import clip_grad_norm_


_global_config['TRAIN'] = 4


class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(weight_init.xavier_uniform(torch.tensor(1,
                self.hidden_size)), requires_grad=False)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.batch_score(hidden, encoder_outputs)
        return func.softmax(attn_energies, dim=1).unsqueeze(1)

    def batch_score(self, hidden, encoder_outputs):
        if self.method == 'dot':
            encoder_outputs = encoder_outputs.permute(1, 2, 0)
            return torch.bmm(hidden.transpose(0, 1), encoder_outputs).squeeze(1
                )
        elif self.method == 'general':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            energy = self.attn(encoder_outputs.view(-1, self.hidden_size)
                ).view(length, batch_size, self.hidden_size)
            return torch.bmm(hidden.transpose(0, 1), energy.permute(1, 2, 0)
                ).squeeze(1)
        elif self.method == 'concat':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            attn_input = torch.cat((hidden.repeat(length, 1, 1),
                encoder_outputs), dim=2)
            energy = self.attn(attn_input.view(-1, 2 * self.hidden_size)).view(
                length, batch_size, self.hidden_size)
            return torch.bmm(self.v.repeat(batch_size, 1, 1), energy.
                permute(1, 2, 0)).squeeze(1)


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1,
        bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=
            dropout, bidirectional=bidirectional)
        if USE_CUDA:
            self.gru = self.gru

    def forward(self, inputs_seqs, input_lens, hidden=None):
        embedded = self.embedding(inputs_seqs)
        packed = pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self
                .hidden_size:]
        return outputs, hidden


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, attn_method, num_layers=1,
        dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout
            )
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        if USE_CUDA:
            self.gru = self.gru
            self.concat = self.concat
            self.out = self.out
        self.attn = Attn(attn_method, hidden_size)

    def forward(self, input_seqs, last_hidden, encoder_outputs):
        batch_size = input_seqs.size(0)
        embedded = self.embedding(input_seqs).unsqueeze(0)
        output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        concat_input = torch.cat((output.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden, attn_weights


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Great_Li_Xin_dumb_chatbot(_paritybench_base):
    pass