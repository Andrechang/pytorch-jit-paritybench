import torch
import torch.fx
# you will need to install torchdynamo, functorch

try:
    # see: https://github.com/pytorch/functorch
    from functorch.compile import compiled_module, draw_graph, draw_graph_compile
    functorch_en = True
except:
    functorch_en = False

try:
    # see: https://github.com/pytorch/torchdynamo
    import torchdynamo
    from torchdynamo.optimizations import BACKENDS
    torchdynamo_en = True
except:
    torchdynamo_en = False

try:
    from thop import profile
except:
    pass

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


def backend_execute(nn_module, nn_cls, args, kwargs, main_args, nn_script=None):
    result = torch.empty((1))
    if nn_script:
        result = nn_script(*args, **kwargs)
    elif main_args.compile_mode == 'torchscript':
        torch.jit.script(nn_module)
    elif main_args.compile_mode == 'functorch_draw':
        graph_path = "{}/{}".format(main_args.tests_dir, nn_cls.__name__)
        nn_script = compile_functions[main_args.compile_mode](nn_module, name=graph_path)
        result = nn_script(*args, **kwargs)
    elif main_args.compile_mode == 'fxgraph_draw':
        graph_path = "{}/{}.png".format(main_args.tests_dir, nn_cls.__name__)
        with torchdynamo.optimize(compile_functions[main_args.compile_mode](graph_path)):
            result = nn_module(*args, **kwargs)
    else:
        with torchdynamo.optimize(compile_functions[main_args.compile_mode]):
            result = nn_module(*args, **kwargs)
    metrics = {}
    if main_args.metrics:
        metrics['macs'], metrics['params'] = profile(nn_module, inputs=(*args,))

    return result, metrics

compile_functions = {
    'torchscript': torch.jit.script,
}

if functorch_en:
    compile_functions['functorch_draw'] = functorch_draw

if torchdynamo_en:
    compile_functions['fxgraph_draw'] = graph_drawer
    compile_functions['my_compiler'] = my_compiler
    compile_functions.update(BACKENDS)