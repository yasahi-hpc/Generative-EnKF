"""
Usage
python convert.py --filename dataset_factory.json

Saving the dataset under
<case_name>/dataset/<mode>

shot*.nc includes u (true value), u_obs1(), u_obs2(), u_obs4(), u_obs8(), ... u_obs40()
"""

import time
import argparse
import json
import pathlib
import copy
import torch
import numpy as np
import xarray as xr
from simulation.utils import (
    save_json,
    observe,
    observe_with_zeros,
    make_divisors,
    sec_to_hh_mm_ss,
)
from model.utils import (
    compute_stats_from_arrays,
    _rhs,
)

def parse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-dirname', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='cases/dataset_factory', \
                        type=str, \
                        choices=None, \
                        help='directory of inputfile', \
                        metavar=None
                       )

    parser.add_argument('--filename', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='dataset_factory.json', \
                        type=str, \
                        choices=None, \
                        help='input file name', \
                        metavar=None
                       )

    parser.add_argument('--out_dirname', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='dataset', \
                        type=str, \
                        choices=None, \
                        help='input file name', \
                        metavar=None
                       )

    parser.add_argument('--mode', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='train', \
                        type=str, \
                        choices=None, \
                        help='train, val, or test', \
                        metavar=None
                       )

    parser.add_argument('--start_idx', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0, \
                        type=int, \
                        choices=None, \
                        help='ensemble id to start converstion', \
                        metavar=None
                       )

    parser.add_argument('--end_idx', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0, \
                        type=int, \
                        choices=None, \
                        help='ensemble id to end converstion', \
                        metavar=None
                       )

    parser.add_argument('--chunk_size', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=1, \
                        type=int, \
                        choices=None, \
                        help='Chunk size of each data', \
                        metavar=None
                       )

    args = parser.parse_args()
    return args

def compute_stats():
    stats = None
    stats_grad = None

    modes = [None, 'std']
    for mode in modes:
        # Compute size, sum, min, max
        for idx in range(start_idx, end_idx):
            in_dir = pathlib.Path(case_name) / f'results/ens_idx{idx:03}'
            files = sorted(list(in_dir.glob('u*.nc')))
            ds = xr.open_mfdataset(files, engine='netcdf4')
            u = ds['u'].values
            F = ds.attrs['F']
            du = _rhs(torch.tensor(u), F).numpy()

            stats = compute_stats_from_arrays(u, mode=mode, stats=stats)
            stats_grad = compute_stats_from_arrays(du, mode=mode, stats=stats_grad)

    def add_mean_and_std(stats):
        stats['mean'] = stats['sum'] / stats['size']
        stats['std'] = np.sqrt( stats['var'] / stats['size'] )
        stats['var'] = stats['var'] / stats['size'] # (should be divided by stats['size']-1 ?)
        return stats

    stats = add_mean_and_std(stats)
    stats_grad = add_mean_and_std(stats_grad)
    return stats, stats_grad

if __name__ == '__main__':
    start = time.time()
    args = parse()
    dirname, filename, mode = args.dirname, args.filename, args.mode
    start_idx, end_idx = args.start_idx, args.end_idx
    chunk_size = args.chunk_size
    out_dirname = args.out_dirname

    json_file = pathlib.Path(dirname) / filename
    if not json_file:
        raise FileNotFoundError(f'input file {filename} not found at {dirname}')

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Loading the output data
    # <case_name>/results/ens_idx#/u#.nc
    case_name = json_data['settings']['case_name']
    n_ens = json_data['simulation']['n_ens']

    assert start_idx <= end_idx, f'start_idx {start_idx} must be smaller than end_idx {end_idx}'
    assert end_idx <= n_ens, f'end_idx {end_idx} must be smaller than n_ens {n_ens}'

    dataset_dir = pathlib.Path(out_dirname) / f'{mode}'
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    stats, stats_grad = compute_stats()
    print('stats', stats)
    print('stats_grad', stats_grad)

    dataset_shot_idx = 0
    for idx in range(start_idx, end_idx):
        in_dir = pathlib.Path(case_name) / f'results/ens_idx{idx:03}'
        files = sorted(list(in_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        nx = len(ds['x'])
        F  = ds.attrs['F']
        obs_intervals = make_divisors(nx)

        nb_steps = ( len(ds['time'])-1 ) // chunk_size

        coords = {'x': ds['x'].values}
        for it in range(nb_steps):
            time_slice = slice(it*chunk_size, (it+1)*chunk_size)
            u = ds['u'].isel(time=time_slice).values

            data_vars = {'u': (['time', 'x'], u)}
            for obs_interval in obs_intervals:
                u_obs = observe_with_zeros(u, obs_interval)
                data_vars[f'u_obs{obs_interval}'] = (['time', 'x'], u_obs)

            filename = dataset_dir / f'shot{dataset_shot_idx:06d}.nc'
            attrs = copy.deepcopy(stats)
            attrs['grad_max']  = stats_grad['max']
            attrs['grad_min']  = stats_grad['min']
            attrs['grad_mean'] = stats_grad['mean']
            attrs['grad_var']  = stats_grad['var']
            attrs['grad_std']  = stats_grad['std']
            attrs['simulation_idx'] = idx
            attrs['F'] = F
            shot_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            shot_ds.to_netcdf(filename, engine='netcdf4')

            dataset_shot_idx += 1

    seconds = time.time() - start
    hh, mm, ss = sec_to_hh_mm_ss(seconds=seconds)
    ss = int(ss)

    print(f'It took {hh:02}h{mm:02}m{ss:02}s to construct dataset for Lorenz96')
