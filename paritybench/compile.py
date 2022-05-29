import torch
import torch.fx
try:
    from functorch.compile import compiled_module, draw_graph, draw_graph_compile
except:
    pass

try:
    # you will need to install torchdynamo, functorch
    # see: https://github.com/pytorch/torchdynamo
    import torchdynamo
    from torchdynamo.optimizations import BACKENDS
    torchdynamo_en = True
except:
    torchdynamo_en = False

def my_compiler(gm: torch.fx.GraphModule, example_inputs):
    '''
    add your own compiler for torch.fx.GraphModule
    '''
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

def graph_drawer(name='graph'):
    '''
    draws a graph for torch.fx.GraphModule in .svg
    '''
    def f(gm: torch.fx.GraphModule, inps):
        draw_graph(gm, name)
        return gm
    return f

def functorch_draw(nn, name='graph'):
    fw_compiler = draw_graph_compile(name+'.png')
    bw_compiler = draw_graph_compile(name+'_bwd.png')
    return compiled_module(nn, fw_compiler, bw_compiler)

compile_functions = {
    'torchscript': torch.jit.script,
    'functorch_draw': functorch_draw,
}

if torchdynamo_en:
    compile_functions['fxgraph_draw'] = graph_drawer
    compile_functions['my_compiler'] = my_compiler
    compile_functions.update(BACKENDS)