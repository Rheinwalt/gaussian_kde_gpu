# gaussian_kde_gpu

## GPU Gaussian kernel density estimation

The provided function `gaussian_kde_gpu()` is a simplified version of Scipy's `gaussian_kde`. It does not support weights and only uses the default Scott's Rule for bandwidth estimation. However, it leverages the computative power of a CUDA supported GPU via Numba. Given the random variable `p` it will estimate the probability density function (PDF) at query point `q` in the following way:

``` {.python .numberLines}
from gaussian_kde_gpu import gaussian_kde_gpu

density = gaussian_kde_gpu(p, q)
```

For more details see [tutorial](GPU_Gaussian_kernel_density_estimation.ipynb).

### Requirements

- Numpy
- Numba
