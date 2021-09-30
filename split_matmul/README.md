## Benchmark Big Matmul + Split Versus Two Small Matmul
### Description
If we have two matmul operations which share the same lhs input and rhs input is a constant matrix. We can merge these two matmul op to a big matmul op + a split op. Sample code as the following two modules.
```
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
```
These two modules would produce the same results.
### Benchmark
Input `x` is of shape (1, n, k). Weight of the big matmul is of shape (k, a + a) while weights of the small matmul are of shape (k, a). We select n from [32, 512, 1024, 2048], k from [32, 256, 512, 1280] and a from [16, 32, 64, 128]. TensorRT profiler is used to get the time spend on the layers. Note that the benchmark results are gathered in fp16 precision on RTX 2070 super GPU.

**BigMatmul vs SmallMatmul**
![compare](https://user-images.githubusercontent.com/20773414/135383410-33617e3b-9153-41f3-93e2-ae4a6701c2ad.png)
Blue line is the time cost of BigMatmul and orange line is the time cost of SmallMatmul.
It seems really hard to draw any conclusion from this graph. Will need some more experiments. But overall seems like a big matmul would be better if we have a large weight (`k` and `a` are large).

**Need More experiments**

