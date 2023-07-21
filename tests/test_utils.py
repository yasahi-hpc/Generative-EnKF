import pytest
import numpy as np
import torch
from simulation.utils import (
    split_dir_and_file,
    sec_to_hh_mm_ss,
    dict2str,
)

from model.utils import (
    batchfy,
    normalize,
    denormalize,
    standardize,
    destandardize,
    compute_stats_from_arrays,
)

@pytest.mark.parametrize('full_path, dirname',
    [('file.json', './'), ('file.json/', './'), ('/file.json', './'), ('tmp/file.json', 'tmp'), ('tmp/sub/file.json', 'tmp/sub')]
)
def test_split_dir_and_file(full_path, dirname):
    _dirname, _filename = split_dir_and_file(full_path)
    
    filename = 'file.json'
    
    assert _dirname == dirname
    assert _filename == filename

@pytest.mark.parametrize('seconds, hh_mm_ss',
    [(5.3, '00:00:05'), (7, '00:00:07'), (61, '00:01:01'), (3601, '01:00:01'), (36001, '10:00:01')]
)
def test_sec_to_hh_mm_ss(seconds, hh_mm_ss):
    hh, mm, ss = sec_to_hh_mm_ss(seconds)
    ss = int(ss)

    assert f'{hh:02}:{mm:02}:{ss:02}' == hh_mm_ss

@pytest.mark.parametrize('dict, expected_str',
    [({'key0': 'value0'}, 'key0: value0'),
     ({'key0': ['value0', 'value1', 'value2']}, 'key0: value0, value1, value2'),
     ({'key0': ['value0-0', 'value0-1'], 'key1': 'value1-0'}, 'key0: value0-0, value0-1\nkey1: value1-0'),
    ]
)
def test_dict2str(dict, expected_str):
    assert dict2str(dict) == expected_str

@pytest.mark.parametrize('shape, batch_size', [((40), 1), ((8, 40), 4)])
def test_batchfy(shape, batch_size):
    if type(shape) is not tuple:
        shape = (1, shape)

    obs = np.random.rand(*shape)
    obs = batchfy(obs, batch_size)

    expected_batch_shape = (batch_size, shape[0], 1, shape[1])

    assert obs.shape == expected_batch_shape

@pytest.mark.parametrize('shape, batch_size, scale', [((40), 1, 1), ((8, 40), 4, 1), ((8, 40), 4, 2)])
def test_normalization(shape, batch_size, scale, factor=3):
    if type(shape) is not tuple:
        shape = (1, shape)

    # Before normalization, some elements of x should be out of [0, 1] range
    x = torch.randn(batch_size, *shape) * factor
    x0 = x.clone() # keep original values

    xmin, xmax = torch.min(x), torch.max(x)
    assert xmax > 1.0 or xmin < 0.0

    x_norm = normalize(x, xmax, xmin, scale=scale)

    # After the normalization, all elements should be [0, 1] or [-1, 1] range
    if scale == 1:
        # Rerange to [0, 1]
        expected_xmin, expected_xmax = 0, 1
    elif scale == 2:
        # Rerange to [-1, 1]
        expected_xmin, expected_xmax = -1, 1

    atol = 1.e-6
    assert torch.all( x_norm <= expected_xmax + atol )
    assert torch.all( x_norm >= expected_xmin - atol )

    # After the denormalization, all the elements should be identical to the original
    x_denorm = denormalize(x_norm, xmax, xmin, scale=scale)
    assert torch.allclose(x0, x_denorm, atol=1.e-3, rtol=1.e-3)

@pytest.mark.parametrize('shape, batch_size', [((40), 1), ((8, 40), 4), ((8, 40), 4)])
def test_standardization(shape, batch_size, factor=3):
    if type(shape) is not tuple:
        shape = (1, shape)

    # Before normalization, some elements of x should be out of [0, 1] range
    x = torch.randn(batch_size, *shape) * factor
    x0 = x.clone() # keep original values

    mean, std = torch.mean(x), torch.std(x)

    x_norm = standardize(x, mean, std)

    # After the normalization, the mean is 0 and std is 1
    expected_mean, expected_std = 0, 1

    atol = 1.e-6
    assert torch.all( torch.abs( torch.mean(x_norm) - expected_mean ) < atol )
    assert torch.all( torch.abs( torch.std(x_norm) - expected_std ) < atol )

    # After the denormalization, all the elements should be identical to the original
    x_denorm = destandardize(x_norm, mean, std)
    assert torch.allclose(x0, x_denorm, atol=1.e-3, rtol=1.e-3)

@pytest.mark.parametrize('shape, num_arrays', [((40), 1), ((8, 40), 1), ((8, 40), 3)])
def test_compute_stats_from_arrays(shape, num_arrays):
    if type(shape) is not tuple:
        shape = (1, shape)
    # Firstly create a random array
    x = np.random.randn(num_arrays, *shape)

    # Compute global stats
    mean, std = np.mean(x), np.std(x)
    min, max = np.min(x), np.max(x)

    list_of_arrays = [x[idx] for idx in range(num_arrays)]

    # Compute execept for std
    stats = None
    for array in list_of_arrays:
        stats = compute_stats_from_arrays(array, mode=None, stats=stats)

    # Compute std
    for array in list_of_arrays:
        stats = compute_stats_from_arrays(array, mode='std', stats=stats)

    stats['mean'] = stats['sum'] / stats['size'] 
    stats['std'] = np.sqrt( stats['var'] / stats['size'] )

    assert np.isclose(stats['mean'], mean)
    assert np.isclose(stats['std'], std)
    assert np.isclose(stats['max'], max)
    assert np.isclose(stats['min'], min)
