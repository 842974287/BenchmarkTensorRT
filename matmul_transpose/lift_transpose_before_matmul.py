import torch
import torch.nn as nn
import torch.fx
import tensorrt as trt
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter, TRTInterpreter, TRTModule, InputTensorSpec

class bmm_permute(nn.Module):
    def forward(self, x, y):
        return torch.bmm(x, y).permute(0, 2, 1)

class permute_bmm(nn.Module):
    def forward(self, x, y):
        return torch.bmm(x.permute(0, 2, 1), y.permute(0, 2, 1))

class wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = permute_bmm()
    
    def forward(self, x, y):
        return self.mod(y, x)

test_inputs = [torch.randn(1, 2, 3), torch.randn(1, 3, 4)]
bp_mod = acc_tracer.trace(bmm_permute(), test_inputs)
pb_mod = acc_tracer.trace(wrapper(), test_inputs, leaf_module_list={permute_bmm})
print(torch.eq(bp_mod(*test_inputs), pb_mod(*test_inputs)))

@tensorrt_converter(permute_bmm)
def convert(network, target, args, kwargs, name):
    x, y = args[0], args[1]
    
    layer = network.add_matrix_multiply(x, trt.MatrixOperation.TRANSPOSE, y, trt.MatrixOperation.TRANSPOSE)
    layer.name = name
    return layer.get_output(0)

def benchmark(i, j, l):
    # print(f"\nshape {i}, {j}, {l}\n")
    test_inputs = [torch.randn(128, i, j).half(), torch.randn(128, j, l).half()]
    interp = TRTInterpreter(bp_mod, input_specs=InputTensorSpec.from_tensors(test_inputs), logger_level=trt.Logger.WARNING)
    bp_trt = TRTModule(*interp.run(max_batch_size=256))
    interp = TRTInterpreter(pb_mod, input_specs=InputTensorSpec.from_tensors(test_inputs))
    pb_trt = TRTModule(*interp.run(max_batch_size=256))

    bp_trt.enable_profiling()
    pb_trt.enable_profiling()
    test_inputs = [i.cuda() for i in test_inputs]
    bp_trt(*test_inputs)
    pb_trt(*test_inputs)
    torch.cuda.synchronize()

n = [32, 128, 512, 1024]
m = [32, 128, 512, 1024]
l = [32, 128, 512, 1024]
for i in n:
    for j in m:
        for k in l:
            benchmark(i, j, k)

