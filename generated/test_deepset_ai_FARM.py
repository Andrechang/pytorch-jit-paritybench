import sys
_module = sys.modules[__name__]
del sys
conf = _module
qa_formats = _module
conversion_huggingface_models = _module
conversion_huggingface_models_classification = _module
doc_classification = _module
doc_classification_cola = _module
doc_classification_crossvalidation = _module
doc_classification_custom_optimizer = _module
doc_classification_fasttext_LM = _module
doc_classification_multilabel = _module
doc_classification_multilabel_roberta = _module
doc_classification_with_earlystopping = _module
doc_classification_word_embedding_LM = _module
doc_regression = _module
embeddings_extraction = _module
embeddings_extraction_s3e_pooling = _module
evaluation = _module
lm_finetuning = _module
natural_questions = _module
ner = _module
onnx_question_answering = _module
passage_ranking = _module
question_answering = _module
question_answering_crossvalidation = _module
streaming_inference = _module
text_pair_classification = _module
train_from_scratch = _module
train_from_scratch_with_sagemaker = _module
wordembedding_inference = _module
farm = _module
conversion = _module
convert_tf_checkpoint_to_pytorch = _module
BertOnnxModel = _module
OnnxModel = _module
onnx_optimization = _module
bert_model_optimization = _module
data_handler = _module
data_silo = _module
dataloader = _module
dataset = _module
input_features = _module
processor = _module
samples = _module
utils = _module
eval = _module
metrics = _module
msmarco_passage_farm = _module
msmarco_passage_official = _module
squad_evaluation = _module
experiment = _module
file_utils = _module
infer = _module
inference_rest_api = _module
modeling = _module
adaptive_model = _module
language_model = _module
optimization = _module
prediction_head = _module
predictions = _module
tokenization = _module
wordembedding_utils = _module
train = _module
visual = _module
ascii = _module
images = _module
text = _module
run_all_experiments = _module
setup = _module
conftest = _module
convert_result_to_csv = _module
question_answering_accuracy = _module
create_testdata = _module
test_conversion = _module
test_doc_classification = _module
test_doc_classification_distilbert = _module
test_doc_classification_roberta = _module
test_doc_regression = _module
test_inference = _module
test_lm_finetuning = _module
test_natural_questions = _module
test_ner = _module
test_ner_amp = _module
test_processor_saving_loading = _module
test_question_answering = _module
test_s3e_pooling = _module
test_tokenization = _module

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


import logging


from functools import partial


import torch


from torch.utils.data.sampler import SequentialSampler


import copy


import numpy


from torch import nn


from collections import OrderedDict


import numpy as np


import inspect


from torch.nn.parallel import DistributedDataParallel


from torch.nn import DataParallel


import itertools


from scipy.special import expit


from scipy.special import softmax


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn import BCEWithLogitsLoss


EMBEDDING_VOCAB_FILES_MAP = {}


logger = logging.getLogger(__name__)


def load_from_cache(pretrained_model_name_or_path, s3_dict, **kwargs):
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    s3_file = s3_dict[pretrained_model_name_or_path]
    try:
        resolved_file = cached_path(s3_file, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies, resume_download
            =resume_download)
        if resolved_file is None:
            raise EnvironmentError
    except EnvironmentError:
        if pretrained_model_name_or_path in s3_dict:
            msg = "Couldn't reach server at '{}' to download data.".format(
                s3_file)
        else:
            msg = (
                "Model name '{}' was not found in model name list. We assumed '{}' was a path, a model identifier, or url to a configuration file or a directory containing such a file but couldn't find any such file at this path or url."
                .format(pretrained_model_name_or_path, s3_file))
        raise EnvironmentError(msg)
    if resolved_file == s3_file:
        logger.info('loading file {}'.format(s3_file))
    else:
        logger.info('loading file {} from cache at {}'.format(s3_file,
            resolved_file))
    return resolved_file


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if (cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <=
        96 or cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def run_split_on_punc(text, never_split=None):
    """Splits punctuation on a piece of text.
    Function taken from HuggingFace: transformers.tokenization_bert.BasicTokenizer
    """
    if never_split is not None and text in never_split:
        return [text]
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return [''.join(x) for x in output]


OUTPUT_DIM_NAMES = ['dim', 'hidden_size', 'd_model']


class WrappedDataParallel(DataParallel):
    """
    A way of adapting attributes of underlying class to parallel mode. See: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

    Gets into recursion errors. Workaround see: https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class WrappedDDP(DistributedDataParallel):
    """
    A way of adapting attributes of underlying class to distributed mode. Same as in WrappedDataParallel above.
    Even when using distributed on a single machine with multiple GPUs, apex can speed up training significantly.
    Distributed code must be launched with "python -m torch.distributed.launch --nproc_per_node=1 run_script.py"
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False


class PredictionHead(nn.Module):
    """ Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions. """
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific PredictionHead implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create(cls, prediction_head_name, layer_dims, class_weights=None):
        """
        Create subclass of Prediction Head.

        :param prediction_head_name: Classname (exact string!) of prediction head we want to create
        :type prediction_head_name: str
        :param layer_dims: describing the feed forward block structure, e.g. [768,2]
        :type layer_dims: List[Int]
        :param class_weights: The loss weighting to be assigned to certain label classes during training.
           Used to correct cases where there is a strong class imbalance.
        :type class_weights: list[Float]
        :return: Prediction Head of class prediction_head_name
        """
        return cls.subclasses[prediction_head_name](layer_dims=layer_dims,
            class_weights=class_weights)

    def save_config(self, save_dir, head_num=0):
        """
        Saves the config as a json file.

        :param save_dir: Path to save config to
        :type save_dir: str or Path
        :param head_num: Which head to save
        :type head_num: int
        """
        self.generate_config()
        output_config_file = Path(save_dir
            ) / f'prediction_head_{head_num}_config.json'
        with open(output_config_file, 'w') as file:
            json.dump(self.config, file)

    def save(self, save_dir, head_num=0):
        """
        Saves the prediction head state dict.

        :param save_dir: path to save prediction head to
        :type save_dir: str or Path
        :param head_num: which head to save
        :type head_num: int
        """
        output_model_file = Path(save_dir) / f'prediction_head_{head_num}.bin'
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = {}
        for key, value in self.__dict__.items():
            if is_json(value) and key[0] != '_':
                config[key] = value
        config['name'] = self.__class__.__name__
        self.config = config

    @classmethod
    def load(cls, config_file, strict=True, load_weights=True):
        """
        Loads a Prediction Head. Infers the class of prediction head from config_file.

        :param config_file: location where corresponding config is stored
        :type config_file: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        config = json.load(open(config_file))
        prediction_head = cls.subclasses[config['name']](**config)
        if load_weights:
            model_file = cls._get_model_file(config_file=config_file)
            logger.info('Loading prediction head from {}'.format(model_file))
            prediction_head.load_state_dict(torch.load(model_file,
                map_location=torch.device('cpu')), strict=strict)
        return prediction_head

    def logits_to_loss(self, logits, labels):
        """
        Implement this function in your special Prediction Head.
        Should combine logits and labels with a loss fct to a per sample loss.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param labels: labels, can vary in shape and type, depending on task
        :type labels: object
        :return: per sample loss as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def logits_to_preds(self, logits):
        """
        Implement this function in your special Prediction Head.
        Should combine turn logits into predictions.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :return: predictions as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def prepare_labels(self, **kwargs):
        """
        Some prediction heads need additional label conversion.
        E.g. NER needs word level labels turned into subword token level labels.

        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: labels in the right format
        :rtype: object
        """
        raise NotImplementedError()

    def resize_input(self, input_dim):
        """ This function compares the output dimensionality of the language model against the input dimensionality
        of the prediction head. If there is a mismatch, the prediction head will be resized to fit."""
        if 'feed_forward' not in dir(self):
            return
        else:
            old_dims = self.feed_forward.layer_dims
            if input_dim == old_dims[0]:
                return
            new_dims = [input_dim] + old_dims[1:]
            logger.info(
                f'Resizing input dimensions of {type(self).__name__} ({self.task_name}) from {old_dims} to {new_dims} to match language model'
                )
            self.feed_forward = FeedForwardBlock(new_dims)
            self.layer_dims[0] = input_dim
            self.feed_forward.layer_dims[0] = input_dim

    @classmethod
    def _get_model_file(cls, config_file):
        if 'config.json' in str(config_file) and 'prediction_head' in str(
            config_file):
            head_num = int(''.join([char for char in os.path.basename(
                config_file) if char.isdigit()]))
            model_file = Path(os.path.dirname(config_file)
                ) / f'prediction_head_{head_num}.bin'
        else:
            raise ValueError(
                f"This doesn't seem to be a proper prediction_head config file: '{config_file}'"
                )
        return model_file

    def _set_name(self, name):
        self.task_name = name


class FeedForwardBlock(nn.Module):
    """ A feed forward neural network of variable depth and width. """

    def __init__(self, layer_dims, **kwargs):
        super(FeedForwardBlock, self).__init__()
        self.layer_dims = layer_dims
        n_layers = len(layer_dims) - 1
        layers_all = []
        self.output_size = layer_dims[-1]
        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_deepset_ai_FARM(_paritybench_base):
    pass
    def test_000(self):
        self._check(FeedForwardBlock(*[], **{'layer_dims': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})
