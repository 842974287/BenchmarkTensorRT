import torch
import torch.nn as nn
import torch.fx
import tensorrt as trt
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter, TRTInterpreter, TRTModule, InputTensorSpec

n_choice = [32, 512, 1024, 2048]
k_choice = [32, 256, 512, 1280]
a_b_choice = [16, 32, 64, 128]


import torch.fx.experimental.fx_acc.acc_ops as acc_ops

# There's a bug in fx2trt converter for slice_tensor, need to fix it.
@tensorrt_converter(acc_ops.slice_tensor)
def acc_ops_slice_tensor(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"slice_tensor received input {input_val} that is not part "
                           "of the TensorRT region!")

    dims = kwargs["dims"]
    if network.has_implicit_batch_dimension:
        if not len(dims):
            raise RuntimeError("dim argument cannot be empty!")
        if any([dim == 0 for dim in dims]):
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dims}!"
            )
        dims = [d - 1 for d in dims]
    else:
        raise RuntimeError("We don't support slice_tensor with explicit batch dimension yet!")

    start = [0] * len(input_val.shape)
    stride = [1] * len(start)
    output_shape = list(input_val.shape)
    starts = kwargs["starts"]
    stops = kwargs["stops"]
    steps = kwargs["steps"]

    for i, dim in enumerate(dims):
        start[dim] = starts[i]
        stride[dim] = steps[i]
        output_shape[dim] = (stops[i] - starts[i]) // steps[i]

    layer = network.add_slice(input_val, start=start, shape=output_shape, stride=stride)
    layer.name = name
    return layer.get_output(0)


def benchmark(n, k, a, b):
    class BigMatmul(nn.Module):
        def __init__(self):
            super().__init__()
            self.y = nn.Parameter(torch.randn(k, a + b))

        def forward(self, x):
            x = torch.matmul(x, self.y)
            return torch.split(x, [a, b], dim=2)

    class SmallMatmul(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(k, a))
            self.b = nn.Parameter(torch.randn(k, b))

        def forward(self, x):
            a = torch.matmul(x, self.a)
            b = torch.matmul(x, self.b)
            return a, b

    inputs = [torch.randn(1, n, k).half().cuda()]
    big_matmul = acc_tracer.trace(BigMatmul().eval().half().cuda(), inputs)
    small_matmul = acc_tracer.trace(SmallMatmul().eval().half().cuda(), inputs)

    interp = TRTInterpreter(big_matmul, input_specs=InputTensorSpec.from_tensors(inputs), logger_level=trt.Logger.VERBOSE)
    bp_trt = TRTModule(*interp.run(max_batch_size=256))
    interp = TRTInterpreter(small_matmul, input_specs=InputTensorSpec.from_tensors(inputs))
    pb_trt = TRTModule(*interp.run(max_batch_size=256))

    # Warmup
    bp_trt(*inputs)
    pb_trt(*inputs)
    torch.cuda.synchronize()
    
    bp_trt.enable_profiling()
    pb_trt.enable_profiling()

    bp_trt(*inputs)
    pb_trt(*inputs)
    torch.cuda.synchronize()
    
    
for n in n_choice:
    for k in k_choice:
        for a in a_b_choice:
            benchmark(n, k, a, a)
