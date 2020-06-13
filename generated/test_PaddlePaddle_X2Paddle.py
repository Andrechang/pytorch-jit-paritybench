import sys
_module = sys.modules[__name__]
del sys
setup = _module
check_for_lite = _module
merge_params = _module
x2paddle = _module
convert = _module
core = _module
fluid_code = _module
graph = _module
op_mapper = _module
util = _module
decoder = _module
caffe_decoder = _module
caffe_pb2 = _module
onnx_decoder = _module
paddle_decoder = _module
tf_decoder = _module
caffe_custom_layer = _module
axpy = _module
convolutiondepthwise = _module
detectionoutput = _module
normalize = _module
permute = _module
priorbox = _module
register = _module
roipooling = _module
select = _module
shufflechannel = _module
caffe_op_mapper = _module
caffe_shape = _module
InstanceNormalization = _module
onnx_custom_layer = _module
onnx_directly_map = _module
onnx_op_mapper = _module
paddle_op_mapper = _module
tf_op_mapper_nhwc = _module
optimizer = _module
caffe_optimizer = _module
onnx_optimizer = _module
tf_optimizer = _module
tests = _module

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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_PaddlePaddle_X2Paddle(_paritybench_base):
    pass
