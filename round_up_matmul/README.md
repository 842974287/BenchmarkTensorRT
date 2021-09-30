## Benchmark Rounding up K dim in Matmul
### Description
For a matmul op like (n, k) x (k, m), TensorRT provides faster kernels when k and m are multiple of 32 (I'm just guessing here, could be 16 or 64. What do I know). 
```
def roundup(x, n=32):
    return n * math.ceil(1.0 * x / n)


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
```
Note that these two modules don't return the same results. **TODO** add another module for benchmark which does padding and split to generate the same result.

### Benchmark
For matmul like (n, k) x (k, m), we fix n to 2048 and m to 128. k is sampled using `range(150, 1000, 32)`.

**Original Module (A) vs RoundUp Module (B)**
![compare](https://user-images.githubusercontent.com/20773414/135391743-5cc9b25c-daf1-4ba6-b053-5b607578e384.png)
Blue line is the time cost for original module and orange line is the time cost after rounding up. Rounding up seems to give a significant boost on the performance. In practice though, in order to get the same results we'll probably end up with more computation to do like padding and split. Maybe we can get away from padding by doing it before inference and postpone the split as later as possible?
