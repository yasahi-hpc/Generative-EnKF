import torch
import copy
import numpy as np
import xarray as xr
from functools import partial
from ._base_trainer import _BaseTrainer
from .unet import UNet
from .diffusion import GaussianDiffusion
from .utils import batchfy, _rhs, normalize, denormalize, standardize, destandardize

class PhysicsInformedDiffusionTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'Physics-Informed-Diffusion'
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.num_resolutions = kwargs.get('num_resolutions', 4)
        self.timesteps = kwargs.get('timesteps', 1000)
        self.loss_type = kwargs.get('loss_type', 'l1')

        # Default parameters for UNet
        self.model_dict = {
            'dim': 40,
            'init_dim': 40,
            'out_dim': 1,
            'dim_mults': (1,2,4,8),
            'channels': 1,
            'self_condition': False,
            'obs_condition': True,
            'phys_condition': True,
            'resnet_block_groups': 8,
            'learned_variance': False,
            'learned_sinusoidal_cond': False,
            'random_fourier_features': False,
            'learned_sinusoidal_dim': 16,
            'obs_interval': self.obs_interval,
            'prob_uncond': self.prob_uncond,
        }

        # Default parameters for diffusion
        self.diffusion_dict = {
            'seq_length': 40,
            'timesteps': self.timesteps,
            'sampling_timesteps': self.sampling_timesteps,
            'loss_type': self.loss_type,
            'objective': 'pred_noise',
            'beta_schedule': 'cosine_beta',
            'p2_loss_weight_gamma': 0.,
            'p2_loss_weight_k': 1,
            'ddim_sampling_eta': self.ddim_sampling_eta,
            'save_intermediate_sampling_imgs': self.save_intermediate_sampling_imgs,
            'result_dir': self.intermediate_sample_dir,
        }

    def _initialize(self, *args, **kwargs):
        # Define models
        self.diffusion_model = self.__get_model()
        self.diffusion_model = self.diffusion_model.to(self.device)

        # Define optimizer
        self.opt = torch.optim.Adam(self.diffusion_model.parameters(), lr=self.lr)

        # Define [0, 1] normalization processes
        if self.preprocess_type == 'normalization':
            u_max, u_min = self.norm_dict['u_max'], self.norm_dict['u_min']
            du_max, du_min = self.norm_dict['du_max'], self.norm_dict['du_min']
            self._preprocess  = partial(normalize, xmax=u_max, xmin=u_min, scale=1)
            self._postprocess = partial(denormalize, xmax=u_max, xmin=u_min, scale=1)
            self._rescale     = partial(normalize, xmax=u_max*100, xmin=u_min*100, scale=1) # In order to make phys guidance in the same range as observation guidance
            #self._rescale     = partial(normalize, xmax=u_max, xmin=u_min, scale=1) # Just scaling no bias
            #self._rescale     = partial(normalize, xmax=du_max-du_min, xmin=0, scale=1) # Just scaling no bias
        elif self.preprocess_type == 'standardization':
            u_mean, u_std = self.norm_dict['u_mean'], self.norm_dict['u_std']
            du_mean, du_std = self.norm_dict['du_mean'], self.norm_dict['du_std']
            self._preprocess  = partial(standardize, mean=u_mean, std=u_std)
            self._postprocess = partial(destandardize, mean=u_mean, std=u_std)
            self._rescale     = partial(standardize, mean=0, std=du_std) # Just scaling no bias
        else:
            raise ValueError(f'Preprocess type should be one of normalization and standardization. {self.preprocess_type} is specified')

    def __get_model(self, checkpoint_idx=-1):
        self.model_dict['chunk_size'] = self.seq_len
        norm_dict = copy.deepcopy(self.norm_dict)
        norm_dict['preprocess_type'] = self.preprocess_type
        self.diffusion_dict['norm_dict'] = norm_dict
        model = UNet(**self.model_dict)
        diffusion_model = GaussianDiffusion(model, **self.diffusion_dict)

        self.load_model = self._find_checkpoint(checkpoint_idx)
        if self.load_model:
            ds = xr.open_dataset(self.checkpoint, engine='netcdf4')
            if ds.attrs['model_name'] != self.model_name:
                raise IOError(f'Error in model load. Loading different type of model. Current model: {self.model_name}, Saved model: {ds.attrs["model_name"]}')

            last_run_number = ds.attrs['run_number']
            last_state_file = ds.attrs['last_state_file']
            print(f'Loading {last_state_file}\n')
            state = torch.load(last_state_file)
            diffusion_model.load_state_dict( state['model'] )
            self.initial_step = int(ds.attrs['final_step']) + 1
            self.run_number = last_run_number + 1

        return diffusion_model

    def _train(self, batch, step):
        mode = 'train'
        self.diffusion_model.train()

        u, u_obs = batch
        u, u_obs = u.to(self.device), u_obs.to(self.device)

        u     = self._preprocess(u)
        u_obs = self._preprocess(u_obs)

        self.opt.zero_grad()
        rhs = partial(_rhs, F=self.F, normalize=self._rescale, denormalize=self._postprocess)
        #rhs = partial(_rhs, F=self.F, normalize=self._preprocess, denormalize=self._postprocess)
        loss = self.diffusion_model(u, labels=u_obs, phys_func=rhs)

        loss.backward()
        self.opt.step()

        self.losses[f'{mode}'].append(loss.item())
        self.losses[f'{mode}_step'].append(step)

    def _test(self, batch, step, mode):
        self.diffusion_model.eval()

        u, u_obs = batch
        u, u_obs = u.to(self.device), u_obs.to(self.device)

        u     = self._preprocess(u)
        u_obs = self._preprocess(u_obs)

        rhs = partial(_rhs, F=self.F, normalize=self._rescale, denormalize=self._postprocess)
        #rhs = partial(_rhs, F=self.F, normalize=self._preprocess, denormalize=self._postprocess)
        samples = self.diffusion_model.sample(self.batch_size, labels=u_obs, phys_func=rhs)

        _u       = self._postprocess(u)
        _u_obs   = self._postprocess(u_obs)
        _samples = self._postprocess(samples)

        self.saver.save(u=_u, u_obs=_u_obs, samples=_samples, step=step, mode=mode)

    def sampling(self, *args, **kwargs):
        self.diffusion_model.eval()
        obs = kwargs.get('obs')
        u = kwargs.get('u')

        to_tensor = lambda var: torch.tensor(var).float().to(self.device)

        rhs = partial(_rhs, F=self.F, normalize=self._rescale, denormalize=self._postprocess)
        #rhs = partial(_rhs, F=self.F, normalize=self._preprocess, denormalize=self._postprocess)
        if self.use_ddib and u is not None:
            obs = batchfy(obs, self.batch_size)
            obs = to_tensor(obs)
            u = batchfy(u, self.batch_size, remove_seq_dim=True)
            u = to_tensor(u)

            u   = self._preprocess(u)
            obs = self._preprocess(obs)

            samples = self.diffusion_model.style_transfer(x=u, labels=obs, phys_func=rhs)

        else:
            obs = batchfy(obs, self.batch_size)
            obs = to_tensor(obs)

            obs = self._preprocess(obs)

            samples = self.diffusion_model.sample(self.batch_size, labels=obs, phys_func=rhs)

        samples = self._postprocess(samples)

        samples = samples.squeeze(1)
        samples = samples.cpu().numpy()
        return samples

    def _save_model(self, *args, **kwargs):
        state_file_name = kwargs.get('state_file_name')
        state_dict = {'model': self.diffusion_model.state_dict()}
        torch.save(state_dict, state_file_name)
