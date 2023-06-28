import numpy as np
from numba import cuda
from math import exp


def bandwidth(x):
    """Scott's rule bandwidth."""
    d = x.shape[1]
    f = x.shape[0] ** (-1 / (d + 4))
    H = f * np.eye(d) * np.std(x, axis=0)
    return H*H


@cuda.jit('(float64[:], float64[:,:], float64[:,:], int64, int64, int64, float64[:], float64)')
def cuda_kernel(r, p, q, n, m, d, bw, f):
    """Numba based CUDA kernel."""
    i = cuda.grid(1)
    if i < m:
        for j in range(n):
            arg = 0.
            for k in range(d):
                res = p[j, k] - q[i, k]
                arg += res * res * bw[k]
            arg = f * exp(-arg / 2.)
            r[i] += arg


def gaussian_kde_gpu(p, q, threadsperblock=64):
    """Gaussian kernel density estimation:
       Density of points p evaluated at query points q."""
    n = p.shape[0]
    d = p.shape[1]
    m = q.shape[0]
    assert d == q.shape[1]
    bw = bandwidth(p)
    bwinv = np.diag(np.linalg.inv(bw))
    bwinv = np.ascontiguousarray(bwinv)
    f = (2 * np.pi) ** (-d / 2)
    f /= np.sqrt(np.linalg.det(bw))
    d_est = cuda.to_device(np.zeros(m))
    d_p = cuda.to_device(p)
    d_q = cuda.to_device(q)
    d_bwinv = cuda.to_device(bwinv)
    blockspergrid = (n + threadsperblock - 1) // threadsperblock
    cuda_kernel[blockspergrid, threadsperblock](d_est, d_p, d_q,
                                                n, m, d, d_bwinv, f)
    est = d_est.copy_to_host()
    est /= n
    return est
