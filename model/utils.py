import torch
import subprocess
import time
import numpy as np
from inspect import isfunction

class Timer:
    def __init__(self, device='cuda'):
        self.device = device
        if self.device == 'cuda':
            self._start = torch.cuda.Event(enable_timing=True)
            self._end   = torch.cuda.Event(enable_timing=True)
        else:
            self._start_time = time.time()

    def start(self):
        if self.device == 'cuda':
            self._start.record()
        else:
            self._start_time = time.time()

    def stop(self):
        if self.device == 'cuda':
            self._end.record()
            torch.cuda.synchronize()
            self._elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self._elapsed_ms = (time.time() - self._start_time) * 1.e3

    @property
    def elapsed_ms(self):
        return self._elapsed_ms

    @property
    def elapsed_seconds(self):
        return self._elapsed_ms * 1.e-3

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def batchfy(data, batch_size, channel_size=1, remove_seq_dim=False):
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    if data.ndim != 2:
        raise ValueError('data should be given either in (nx) or (nt, nx) shape')

    data = np.tile(data, (batch_size, channel_size, 1))
    if not remove_seq_dim:
        data = np.expand_dims(data, axis=-2)
    return data

def get_meta():
    """
    Get git version, date, user, and hostname
    """
    meta = ""
                                                                     
    args_dict = {
        #'Git version       : ': ['git', 'rev-parse', 'master'],
        'Date              : ': ['date'],
        'User              : ': ['whoami'],
        'Host name         : ': ['hostname'],
        'Current directory : ': ['pwd'],
    }

    for meta_name, args in args_dict.items():
        stdout = subprocess.run(args=args, encoding='utf-8', stdout=subprocess.PIPE).stdout
        meta += (meta_name + stdout)
                                                                                                                                                                                              
    meta = meta[:-1] # removing the final '\n'
    return meta

def normalize(x, xmax, xmin, scale=1):
    """
    Data range to be
      [0, 1] for scale == 1
      [-1, 1] for scale == 2
    """
    assert scale in [1, 2]

    x -= xmin
    x /= (xmax - xmin)

    if not scale == 1:
        x = scale * (x - 0.5)

    return x

def denormalize(x, xmax, xmin, scale=1):
    """
    Inverse of normalize
    """
    assert scale in [1, 2]

    if not scale == 1:
        x = x / scale + 0.5

    x = x * (xmax - xmin) + xmin
    return x

def standardize(x, mean, std):
    """
    mean to 0 and std to 1
    """
    x -= mean
    x /= std
    return x

def destandardize(x, mean, std):
    """
    Inverse of standardize
    """
    x = (x * std) + mean
    return x

def _rhs(u, F, normalize=None, denormalize=None):
    """
    This equation is supposed to work on the normalized quantity
    So denomralize to the original space and then normalize again
    """
    u = u.clone()

    if exists(denormalize):
        u = denormalize(u)

    u.requires_grad_(True)

    u_p1 = torch.roll(u, -1)
    u_m1 = torch.roll(u,  1)
    u_m2 = torch.roll(u,  2)

    residual = (u_p1 - u_m2) * u_m1 - u + F
    residual_loss = (residual**2).mean()
    du = torch.autograd.grad(residual_loss, u)[0]

    if exists(normalize):
        du = normalize(du)

    return du

def compute_stats_from_arrays(array, mode=None, stats=None):
    """
    Compute stats sequentially from multiple arrays
    """
    if mode == 'std':
        if stats is None:
            raise ValueError('Precomputed stats are not given. They are needed to compute standard deviation')

        # compute std (mean is ready)
        tmp_size = stats.get('size')
        tmp_sum = stats.get('sum')
        mean = tmp_sum / tmp_size
        tmp_var = stats.get('var', 0)
        tmp_var += np.sum( (array - mean)**2 )
        stats = stats.copy()
        stats['var'] = tmp_var

        return stats
    else:
        if stats is None:
            tmp_size = 0
            tmp_sum = 0
            tmp_min, tmp_max = None, None
        else:
            tmp_size = stats.get('size')
            tmp_sum  = stats.get('sum')
            tmp_min  = stats.get('min')
            tmp_max  = stats.get('max')

        # Compute stats from partial data
        tmp_size += array.size
        tmp_sum += np.sum(array)
        tmp_min = np.min(array) if tmp_min is None else np.amin([tmp_min, np.min(array)])
        tmp_max = np.max(array) if tmp_max is None else np.amax([tmp_max, np.max(array)])

        return {'size': tmp_size, 'sum': tmp_sum, 'max': tmp_max, 'min': tmp_min}
