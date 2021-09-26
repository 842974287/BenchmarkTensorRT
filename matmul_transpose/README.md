## Benchmark lifting up transpose before matmul
### Description
When a matmul operation is followed by a transpose operation that transposes the last two dimensions, we can lift the transpose up before the matmul. Using this formula `(A x B)^t = B^t x A^t`.
In PyTorch code it would be like the following
```
class A(nn.Module):
  def forward(self, x, y):
    return torch.bmm(x, y).permute(0, 2, 1)
    
class B(nn.Module):
  def forward(self, x, y):
    return torch.bmm(y.permute(0, 2, 1), x.permute(0, 2, 1))
```

Module `A` and `B` would produce the same results.
### Benchmark
Let's say `x` is of shape (n, k) and `y` is of shape (k, m), we select the value from [32, 128, 512, 1024] for each of them which gives us a total of 64 combinations. We use TensorRT profiler to get the time spend on the layers. Note that all the benchmark results are gathered in fp16 precision.

**Original Module (A) vs Transformed Module (B)**
![compare](https://user-images.githubusercontent.com/20773414/134827597-002f12cd-ffa2-4c08-ac4c-02a3218511e3.png)
Blue line is the time cost for original module and orange line is the time cost after lifting up the transpose. From the graph it seems like this tranformation would always give us some perf benefits.

**Time Breakdown for Original Module (matmul vs permute)**
![compare](https://user-images.githubusercontent.com/20773414/134827750-cecf7f3b-c226-41cb-82f9-ffca5393c3a0.png)
Blue line is the time spend on matmul layer while orange line is the line spend on transpoes layer.
