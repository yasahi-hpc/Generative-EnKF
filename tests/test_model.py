import pytest
import pathlib
import torch
import numpy as np
import xarray as xr
from functools import partial
from torch.utils.data import DataLoader
from model.lorentz96_dataset import Lorenz96Dataset
from model.unet import UNet
from model.diffusion import GaussianDiffusion
from model.img_saver import ImageSaver
from model.utils import cycle, batchfy, _rhs, normalize, denormalize

@pytest.mark.parametrize('obs_interval, chunk_size, ddim_sampling_eta',
    [(1, 1, 0.), (1, 1, 0.1), (4, 1, 0.1), (1, 2, 0.1)])
@pytest.mark.parametrize('phys_condition', [False, True])
class TestModel:
    nb_shots: int = 16
    nx: int = 40
    num_channels: int = 1
    F: float = 8
    u_max: float = 20.

    @pytest.fixture(autouse=True)
    def _model(self, obs_interval, chunk_size, ddim_sampling_eta, phys_condition) -> None:
        model_dict = {
            'dim': self.nx,
            'init_dim': self.nx,
            'out_dim': 1,
            'dim_mults': (1,2,4,8),
            'channels': self.num_channels,
            'self_condition': False,
            'obs_condition': True,
            'phys_condition': phys_condition,
            'resnet_block_groups': 8,
            'learned_variance': False,
            'learned_sinusoidal_cond': False,
            'random_fourier_features': False,
            'learned_sinusoidal_dim': 16,
            'obs_interval': obs_interval,
            'chunk_size': chunk_size,
        }

        timesteps = 10
        sampling_timesteps = timesteps-1 if ddim_sampling_eta > 0. else None

        diffusion_dict = {
            'seq_length': self.nx,
            'timesteps': 10, # should be larger for production
            'sampling_timesteps': sampling_timesteps,
            'loss_type': 'l1',
            'objective': 'pred_noise',
            'beta_schedule': 'cosine_beta',
            'p2_loss_weight_gamma': 0.,
            'p2_loss_weight_k': 1,
            'ddim_sampling_eta': ddim_sampling_eta,
        }

        model = UNet(**model_dict)
        self.diffusion_model = GaussianDiffusion(model, **diffusion_dict)

    @pytest.fixture(autouse=True)
    def _dataset(self, chunk_size, tmp_dir) -> None:
        self.dataset_dir = tmp_dir / f'dataset{chunk_size}'
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir(parents=True)

        obs_intervals = [1, 2, 4]
        for i in range(self.nb_shots):
            filename = self.dataset_dir / f'shot{i:06}.nc'
            u = np.random.rand(self.nx)
            x = np.arange(self.nx)
            t = np.arange(chunk_size)

            data_vars = {'u': (['time', 'x'], np.random.rand(chunk_size, self.nx))}
            coords = {'x': x, 
                      'time': t,}
            
            for obs_interval in obs_intervals:
                data_vars[f'u_obs{obs_interval}'] = (['time', 'x'], np.random.rand(chunk_size, self.nx))
                
            shot_ds = xr.Dataset(data_vars=data_vars, coords=coords)
            if not filename.exists():
                shot_ds.to_netcdf(filename, engine='netcdf4')

    @pytest.fixture(autouse=True)
    def _saver(self, obs_interval, tmp_dir) -> None:
        dataset_dict = {
                        'inference_mode': True,
                        'obs_interval': obs_interval,
                       }

        dataset = Lorenz96Dataset(path=self.dataset_dir, **dataset_dict)
        coords = dataset.coords

        # Image saver
        self.fig_dir = tmp_dir / 'imgs'
        saver_dict = {'out_dir': self.fig_dir,
                      'coords': coords,
                      'n_cols': 2}
        self.saver = ImageSaver(**saver_dict)

    @pytest.mark.parametrize('mode', ['train', 'inference'])
    @pytest.mark.parametrize('batch_size', [1,4])
    def test_dataset(self, mode, obs_interval, chunk_size, batch_size, ddim_sampling_eta=None, phys_condition=None) -> None:
        inference_mode = mode == 'inference'

        dataset_dict = {
                        'inference_mode': inference_mode,
                        'obs_interval': obs_interval,
                       }

        dataset = Lorenz96Dataset(path=self.dataset_dir, **dataset_dict)
        x = dataset.coords['x']
        nx = x.shape[0]
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        data_loader = cycle(data_loader)

        # loading more than exists
        nb_steps = 2
        for step in range(nb_steps):
            data_batch = next(data_loader)

            if inference_mode:
                idx, u, u_obs = data_batch
            else:
                u, u_obs = data_batch

            expected_u_shape = (batch_size, self.num_channels, nx)

            # (b, t, c, w)
            expected_u_obs_shape = (batch_size, chunk_size, self.num_channels, nx)

            assert u.shape == expected_u_shape
            assert u_obs.shape == expected_u_obs_shape

    def test_train(self, chunk_size, obs_interval, phys_condition, ddim_sampling_eta=None, batch_size=4) -> None:
        u     = torch.rand((batch_size, self.num_channels, self.nx))
        u_obs = torch.rand((batch_size, chunk_size, self.num_channels, self.nx))

        preprocess = partial(normalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
        postprocess = partial(denormalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
        rhs = partial(_rhs, F=self.F, normalize=preprocess, denormalize=postprocess) if phys_condition else None

        loss = self.diffusion_model(u, labels=u_obs, phys_func=rhs)

    @pytest.mark.parametrize('batch_size', [1,4])
    def test_sampling(self, chunk_size, obs_interval, batch_size, phys_condition, ddim_sampling_eta=None) -> None:
        u     = torch.rand((batch_size, self.num_channels, self.nx))
        u_obs = torch.rand((batch_size, chunk_size, self.num_channels, self.nx))

        preprocess = partial(normalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
        postprocess = partial(denormalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
        rhs = partial(_rhs, F=self.F, normalize=preprocess, denormalize=postprocess) if phys_condition else None

        samples = self.diffusion_model.sample(batch_size, labels=u_obs, phys_func=rhs)
        expected_u_shape = (batch_size, 1, self.nx)

        self.saver.save(u=u, u_obs=u_obs, samples=samples, step=0, mode='test')
        assert samples.shape == expected_u_shape

    @pytest.mark.parametrize('batch_size', [1,4])
    def test_sampling_interface(self, chunk_size, obs_interval, batch_size, phys_condition, ddim_sampling_eta=None) -> None:
        obs = np.random.rand(chunk_size, self.nx)

        # if chunk_size > 1, then the shape would be kept
        # if chunk_size == 1, then 1D data, this is the original interface
        obs = obs.squeeze() 
        obs = batchfy(obs, batch_size)
        
        to_tensor = lambda var: torch.tensor(var).float()
        obs = to_tensor(obs)

        expected_u_obs_shape = (batch_size, chunk_size, self.num_channels, self.nx)
        assert obs.shape == expected_u_obs_shape

        preprocess = partial(normalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
        postprocess = partial(denormalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
        rhs = partial(_rhs, F=self.F, normalize=preprocess, denormalize=postprocess) if phys_condition else None

        samples = self.diffusion_model.sample(batch_size, labels=obs, phys_func=rhs)
        samples = samples.squeeze(-2)
        samples = samples.cpu().numpy()
        expected_u_shape = (batch_size, self.nx)
        assert samples.shape == expected_u_shape

    def test_style_transfer_interface(self, chunk_size, obs_interval, ddim_sampling_eta, phys_condition, batch_size=4) -> None:
        u   = torch.rand(self.nx)
        obs = np.random.rand(chunk_size, self.nx)

        obs = obs.squeeze() 
        obs = batchfy(obs, batch_size)
        u   = batchfy(u, batch_size, remove_seq_dim=True)

        if ddim_sampling_eta > 0.:
            to_tensor = lambda var: torch.tensor(var).float()
            obs = to_tensor(obs)
            u = to_tensor(u)

            expected_u_obs_shape = (batch_size, chunk_size, self.num_channels, self.nx)
            expected_u_shape = (batch_size, self.num_channels, self.nx)

            assert obs.shape == expected_u_obs_shape
            assert u.shape == expected_u_shape

            preprocess = partial(normalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
            postprocess = partial(denormalize, xmax=self.u_max, xmin=-self.u_max, scale=1)
            rhs = partial(_rhs, F=self.F, normalize=preprocess, denormalize=postprocess) if phys_condition else None

            samples = self.diffusion_model.style_transfer(x=u, labels=obs, phys_func=rhs)
            samples = samples.squeeze(-2)
            samples = samples.cpu().numpy()
            expected_u_shape = (batch_size, self.nx)

            assert samples.shape == expected_u_shape
        else:
            pass
