"""
    data (simulation): (b, c, w)
    label (observation): (b, c, o)
"""

import xarray as xr
import numpy as np
import torch
import pathlib
from simulation.utils import observe_with_zeros

class Lorenz96Dataset(torch.utils.data.Dataset):
    def __init__(self, path, **kwargs):
        self.files = sorted(list(pathlib.Path(path).glob('shot*.nc')))
        if not self.files:
            raise FileNotFoundError(f'shot*.nc does not exist in {path}')

        self.datanum = len(self.files)
        self.version = kwargs.get('version', 0)
        self.device = kwargs.get('device', 'cpu')
        self.inference_mode = kwargs.get('inference_mode', False)
        self.obs_interval = kwargs.get('obs_interval', 1)
        self.sigma = kwargs.get('sigma', 1)
        self.obs_noise_runtime = kwargs.get('obs_noise_runtime', False)

        # Read the dataset to save the coordinate 
        ds = xr.open_dataset(self.files[0], engine='netcdf4')
        coord_name = f'x_obs{self.obs_interval}'
        self.coords = ds.coords
        self.coords['x_obs'] = ds['x'].values[::self.obs_interval]

        self.F = ds.attrs.get('F', 8)
        u_max  = ds.attrs.get('max', 20)
        u_min  = ds.attrs.get('min', -20)
        u_mean = ds.attrs.get('mean', 2.0)
        u_std  = ds.attrs.get('std',  3.5)

        du_max  = ds.attrs.get('grad_max', 0.115)
        du_min  = ds.attrs.get('grad_min', -0.1)
        du_mean = ds.attrs.get('grad_mean', 0.002)
        du_std  = ds.attrs.get('grad_std',  0.0125)

        # Add coefficients for normalization
        self.norm_dict = {}
        self.norm_dict['u_max']  = u_max
        self.norm_dict['u_min']  = u_min
        self.norm_dict['u_mean'] = u_mean
        self.norm_dict['u_std']  = u_std

        self.norm_dict['du_max']  = du_max
        self.norm_dict['du_min']  = du_min
        self.norm_dict['du_mean'] = du_mean
        self.norm_dict['du_std']  = du_std

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        ds = xr.open_dataset(self.files[idx], engine='netcdf4')

        # (time, x)
        u = ds['u'].isel(time=-1).values
        if self.obs_noise_runtime:
            u_obs = observe_with_zeros(u, obs_interval=self.obs_interval, noise_level=self.sigma)
            u_obs = np.expand_dims(u_obs, axis=0)
        else:
            u_obs = ds[f'u_obs{self.obs_interval}'].values

        # Add channel directions
        # u (time, x), u_obs (time, 1, x)
        u = torch.tensor(np.expand_dims(u, axis=0)).float()
        u_obs = torch.tensor(np.expand_dims(u_obs, axis=1)).float()

        if self.inference_mode:
            return idx, u, u_obs
        else:
            return u, u_obs
