import sys
_module = sys.modules[__name__]
del sys
conf = _module
main = _module
model = _module
splitcross = _module
utils = _module
model = _module
train = _module
util = _module
setup = _module
conftest = _module
test_count = _module
test_imdb = _module
test_iwslt = _module
test_multi30k = _module
test_penn_treebank = _module
test_reverse = _module
test_simple_qa = _module
test_smt = _module
test_snli = _module
test_trec = _module
test_ud_pos = _module
test_wikitext_2 = _module
test_wmt = _module
test_zero_to_zero = _module
test_encoder = _module
test_label_encoder = _module
test_character_encoder = _module
test_delimiter_encoder = _module
test_moses_encoder = _module
test_spacy_encoder = _module
test_static_tokenizer_encoder = _module
test_subword_encoder = _module
test_subword_tokenizer = _module
test_text_encoder = _module
test_treebank_encoder = _module
test_word_encoder = _module
test_accuracy = _module
test_bleu = _module
nn = _module
test_attention = _module
test_cnn_encoder = _module
test_lock_dropout = _module
test_weight_drop = _module
test_balanced_sampler = _module
test_bptt_batch_sampler = _module
test_bptt_sampler = _module
test_bucket_batch_sampler = _module
test_deterministic_sampler = _module
test_distributed_batch_sampler = _module
test_distributed_sampler = _module
test_noisy_sorted_sampler = _module
test_oom_batch_sampler = _module
test_repeat_sampler = _module
test_sorted_sampler = _module
test_download = _module
test_random = _module
test_utils = _module
word_to_vector = _module
test_bpemb = _module
test_char_n_gram = _module
test_fast_text = _module
test_glove = _module
torchnlp = _module
_third_party = _module
lazy_loader = _module
weighted_random_sampler = _module
datasets = _module
count = _module
imdb = _module
iwslt = _module
multi30k = _module
penn_treebank = _module
reverse = _module
simple_qa = _module
smt = _module
snli = _module
trec = _module
ud_pos = _module
wikitext_2 = _module
wmt = _module
zero = _module
download = _module
encoders = _module
encoder = _module
label_encoder = _module
text = _module
character_encoder = _module
default_reserved_tokens = _module
delimiter_encoder = _module
moses_encoder = _module
spacy_encoder = _module
static_tokenizer_encoder = _module
subword_encoder = _module
subword_text_tokenizer = _module
text_encoder = _module
treebank_encoder = _module
whitespace_encoder = _module
metrics = _module
accuracy = _module
bleu = _module
attention = _module
cnn_encoder = _module
lock_dropout = _module
weight_drop = _module
random = _module
samplers = _module
balanced_sampler = _module
bptt_batch_sampler = _module
bptt_sampler = _module
bucket_batch_sampler = _module
deterministic_sampler = _module
distributed_batch_sampler = _module
distributed_sampler = _module
noisy_sorted_sampler = _module
oom_batch_sampler = _module
repeat_sampler = _module
sorted_sampler = _module
utils = _module
aliases = _module
bpemb = _module
char_n_gram = _module
fast_text = _module
glove = _module
pretrained_word_vectors = _module

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


import math


import numpy as np


import torch


import torch.nn as nn


from collections import defaultdict


from functools import partial


import itertools


from torch.utils.data import DataLoader


from torch.utils.data.sampler import SequentialSampler


import torch.optim as optim


from numpy.testing import assert_almost_equal


import numpy


from collections import namedtuple


from torch.nn import Conv1d


from torch.nn import Linear


from torch.nn import ReLU


from torch.nn import Parameter


import logging


import inspect


import collections


class SplitCrossEntropyLoss(nn.Module):
    """SplitCrossEntropyLoss calculates an approximate softmax"""

    def __init__(self, hidden_size, splits, verbose=False):
        super(SplitCrossEntropyLoss, self).__init__()
        self.hidden_size = hidden_size
        self.splits = [0] + splits + [100 * 1000000]
        self.nsplits = len(self.splits) - 1
        self.stats = defaultdict(list)
        self.verbose = verbose
        if self.nsplits > 1:
            self.tail_vectors = nn.Parameter(torch.zeros(self.nsplits - 1,
                hidden_size))
            self.tail_bias = nn.Parameter(torch.zeros(self.nsplits - 1))

    def logprob(self, weight, bias, hiddens, splits=None,
        softmaxed_head_res=None, verbose=False):
        if softmaxed_head_res is None:
            start, end = self.splits[0], self.splits[1]
            head_weight = None if end - start == 0 else weight[start:end]
            head_bias = None if end - start == 0 else bias[start:end]
            if self.nsplits > 1:
                head_weight = (self.tail_vectors if head_weight is None else
                    torch.cat([head_weight, self.tail_vectors]))
                head_bias = self.tail_bias if head_bias is None else torch.cat(
                    [head_bias, self.tail_bias])
            head_res = torch.nn.functional.linear(hiddens, head_weight,
                bias=head_bias)
            softmaxed_head_res = torch.nn.functional.log_softmax(head_res)
        if splits is None:
            splits = list(range(self.nsplits))
        results = []
        running_offset = 0
        for idx in splits:
            if idx == 0:
                results.append(softmaxed_head_res[:, :-(self.nsplits - 1)])
            else:
                start, end = self.splits[idx], self.splits[idx + 1]
                tail_weight = weight[start:end]
                tail_bias = bias[start:end]
                tail_res = torch.nn.functional.linear(hiddens, tail_weight,
                    bias=tail_bias)
                head_entropy = softmaxed_head_res[:, (-idx)].contiguous()
                tail_entropy = torch.nn.functional.log_softmax(tail_res)
                results.append(head_entropy.view(-1, 1) + tail_entropy)
        if len(results) > 1:
            return torch.cat(results, dim=1)
        return results[0]

    def split_on_targets(self, hiddens, targets):
        split_targets = []
        split_hiddens = []
        mask = None
        for idx in range(1, self.nsplits):
            partial_mask = targets >= self.splits[idx]
            mask = mask + partial_mask if mask is not None else partial_mask
        for idx in range(self.nsplits):
            if self.nsplits == 1:
                split_targets, split_hiddens = [targets], [hiddens]
                continue
            if sum(len(t) for t in split_targets) == len(targets):
                split_targets.append([])
                split_hiddens.append([])
                continue
            tmp_mask = mask == idx
            split_targets.append(torch.masked_select(targets, tmp_mask))
            split_hiddens.append(hiddens.masked_select(tmp_mask.unsqueeze(1
                ).expand_as(hiddens)).view(-1, hiddens.size(1)))
        return split_targets, split_hiddens

    def forward(self, weight, bias, hiddens, targets, verbose=False):
        if self.verbose or verbose:
            for idx in sorted(self.stats):
                None
            None
        total_loss = None
        if len(hiddens.size()) > 2:
            hiddens = hiddens.view(-1, hiddens.size(2))
        split_targets, split_hiddens = self.split_on_targets(hiddens, targets)
        start, end = self.splits[0], self.splits[1]
        head_weight = None if end - start == 0 else weight[start:end]
        head_bias = None if end - start == 0 else bias[start:end]
        if self.nsplits > 1:
            head_weight = (self.tail_vectors if head_weight is None else
                torch.cat([head_weight, self.tail_vectors]))
            head_bias = self.tail_bias if head_bias is None else torch.cat([
                head_bias, self.tail_bias])
        combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if
            len(split_hiddens[i])])
        all_head_res = torch.nn.functional.linear(combo, head_weight, bias=
            head_bias)
        softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res)
        if self.verbose or verbose:
            self.stats[0].append(combo.size()[0] * head_weight.size()[0])
        running_offset = 0
        for idx in range(self.nsplits):
            if len(split_targets[idx]) == 0:
                continue
            if idx == 0:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:
                    running_offset + len(split_hiddens[idx])]
                entropy = -torch.gather(softmaxed_head_res, dim=1, index=
                    split_targets[idx].view(-1, 1))
            else:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:
                    running_offset + len(split_hiddens[idx])]
                if self.verbose or verbose:
                    start, end = self.splits[idx], self.splits[idx + 1]
                    tail_weight = weight[start:end]
                    self.stats[idx].append(split_hiddens[idx].size()[0] *
                        tail_weight.size()[0])
                tail_res = self.logprob(weight, bias, split_hiddens[idx],
                    splits=[idx], softmaxed_head_res=softmaxed_head_res)
                head_entropy = softmaxed_head_res[:, (-idx)]
                indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)
                tail_entropy = torch.gather(torch.nn.functional.log_softmax
                    (tail_res), dim=1, index=indices).squeeze()
                entropy = -(head_entropy + tail_entropy)
            running_offset += len(split_hiddens[idx])
            total_loss = entropy.float().sum(
                ) if total_loss is None else total_loss + entropy.float().sum()
        return (total_loss / len(targets)).type_as(weight)


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.
            d_hidden, num_layers=config.n_layers, dropout=config.dp_ratio,
            bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = inputs.detach().new_zeros(*state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1
            ).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2 * config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size] * 2
        self.out = nn.Sequential(Linear(*lin_config), self.relu, self.
            dropout, Linear(*lin_config), self.relu, self.dropout, Linear(*
            lin_config), self.relu, self.dropout, Linear(seq_in_size,
            config.d_out))

    def forward(self, premise, hypothesis):
        prem_embed = self.embed(premise)
        hypo_embed = self.embed(hypothesis)
        if self.config.fix_emb:
            prem_embed = prem_embed.detach()
            hypo_embed = hypo_embed.detach()
        if self.config.projection:
            prem_embed = self.relu(self.projection(prem_embed))
            hypo_embed = self.relu(self.projection(hypo_embed))
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        if self.attention_type == 'general':
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)
        attention_scores = torch.bmm(query, context.transpose(1, 2).
            contiguous())
        attention_scores = attention_scores.view(batch_size * output_len,
            query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len,
            query_len)
        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        output = self.linear_out(combined).view(batch_size, output_len,
            dimensions)
        output = self.tanh(output)
        return output, attention_weights


class CNNEncoder(torch.nn.Module):
    """ A combination of multiple convolution layers and max pooling layers.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    **Thank you** to AI2 for their initial implementation of :class:`CNNEncoder`. Here is
    their `License
    <https://github.com/allenai/allennlp/blob/master/LICENSE>`__.

    Args:
        embedding_dim (int): This is the input dimension to the encoder.  We need this because we
          can't do shape inference in pytorch, and we need to know what size filters to construct
          in the CNN.
        num_filters (int): This is the output dim for each convolutional layer, which is the number
          of "filters" learned by that layer.
        ngram_filter_sizes (:class:`tuple` of :class:`int`, optional): This specifies both the
          number of convolutional layers we will create and their sizes. The default of
          ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding ngrams of
          size 2 to 5 with some number of filters.
        conv_layer_activation (torch.nn.Module, optional): Activation to use after the convolution
          layers.
        output_dim (int or None, optional) : After doing convolutions and pooling, we'll project the
          collected features into a vector of this size.  If this value is ``None``, we will just
          return the result of the max pooling, giving an output of shape
          ``len(ngram_filter_sizes) * num_filters``.
    """

    def __init__(self, embedding_dim, num_filters, ngram_filter_sizes=(2, 3,
        4, 5), conv_layer_activation=ReLU(), output_dim=None):
        super(CNNEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim
        self._convolution_layers = [Conv1d(in_channels=self._embedding_dim,
            out_channels=self._num_filters, kernel_size=ngram_size) for
            ngram_size in self._ngram_filter_sizes]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)
        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self._output_dim
                )
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self):
        return self._embedding_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (:class:`torch.FloatTensor` [batch_size, num_tokens, input_dim]): Sequence
                matrix to encode.
            mask (:class:`torch.FloatTensor`): Broadcastable matrix to `tokens` used as a mask.
        Returns:
            (:class:`torch.FloatTensor` [batch_size, output_dim]): Encoding of sequence.
        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
        tokens = torch.transpose(tokens, 1, 2)
        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(self._activation(convolution_layer(tokens
                )).max(dim=2)[0])
        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs
            ) > 1 else filter_outputs[0]
        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result


class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False
            ).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + str(self.p) + ')'


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))
    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=
                module.training)
            setattr(module, name_w, w)
        return original_module_forward(*args, **kwargs)
    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        >>> from torchnlp.nn import WeightDrop
        >>> import torch
        >>>
        >>> torch.manual_seed(123)
        <torch._C.Generator object ...
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)
        >>>
        >>> input_ = torch.randn(3, 2)
        >>> hidden_state = torch.randn(3, 2)
        >>> weight_drop_gru(input_, hidden_state)
        tensor(... grad_fn=<AddBackward0>)
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = [('weight_hh_l' + str(i)) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class WeightDropGRU(torch.nn.GRU):
    """
    Wrapper around :class:`torch.nn.GRU` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = [('weight_hh_l' + str(i)) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_PetrochukM_PyTorch_NLP(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Linear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4])], {})

    def test_001(self):
        self._check(Attention(*[], **{'dimensions': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(LockedDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(WeightDropLSTM(*[], **{'input_size': 4, 'hidden_size': 4}), [], {'input': torch.rand([4, 4, 4])})

    @_fails_compile()
    def test_004(self):
        self._check(WeightDropGRU(*[], **{'input_size': 4, 'hidden_size': 4}), [], {'input': torch.rand([4, 4, 4])})

