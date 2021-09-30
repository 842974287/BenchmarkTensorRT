import math
import torch
import torch.nn as nn
import torch.fx
import tensorrt as trt
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter, TRTInterpreter, TRTModule, InputTensorSpec


def roundup(x, n=32):
    return n * math.ceil(1.0 * x / n)


def benchmark(n, k, m):

    class Original(nn.Module):
        def __init__(self):
            super().__init__()
            self.y = nn.Parameter(torch.randn(k, m))

        def forward(self, x):
            return torch.matmul(x, self.y)


    class RoundUp(nn.Module):
        def __init__(self):
            super().__init__()
            self.y = nn.Parameter(torch.randn(roundup(k), roundup(m)))

        def forward(self, x):
            return torch.matmul(x, self.y)

    inputs = [torch.randn(1, n, k).half().cuda()]
    round_up_inputs = [torch.randn(1, n, roundup(k)).half().cuda()]
    A = acc_tracer.trace(Original().eval().half().cuda(), inputs)
    B = acc_tracer.trace(RoundUp().eval().half().cuda(), round_up_inputs)


    interp = TRTInterpreter(A, input_specs=InputTensorSpec.from_tensors(inputs), logger_level=trt.Logger.ERROR)
    A_trt = TRTModule(*interp.run(max_batch_size=256))
    interp = TRTInterpreter(B, input_specs=InputTensorSpec.from_tensors(round_up_inputs))
    B_trt = TRTModule(*interp.run(max_batch_size=256))

    A_trt(*inputs)
    B_trt(*round_up_inputs)
    torch.cuda.synchronize()

    A_trt.enable_profiling()
    B_trt.enable_profiling()

    A_trt(*inputs)
    B_trt(*round_up_inputs)
    torch.cuda.synchronize()


n = 2048

for k in range(150, 1000, 32):
    benchmark(n, k, 128)
