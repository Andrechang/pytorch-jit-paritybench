import torch
try:
    from functorch.compile import compiled_module, tvm_compile, draw_graph_compile
    functorch_en = True
except:
    functorch_en = False

def functorch_tvm(nn):
    fw_compiler = tvm_compile(target='llvm', tuning_logfile='fw_keops')
    bw_compiler = tvm_compile(target='llvm', tuning_logfile='bw_keops')
    return compiled_module(nn, fw_compiler, bw_compiler)

def functorch_draw(nn, name='graph'):
    fw_compiler = draw_graph_compile(name) # need pydot
    bw_compiler = draw_graph_compile(name+'_bwd')
    return compiled_module(nn, fw_compiler, bw_compiler)

compile_functions = {
    'torchscript': torch.jit.script,
    'tvm': functorch_tvm,
    'fxgraph_draw': functorch_draw,
}