import pathlib
import numpy as np
import xarray as xr
#from collections import namedtuple
from model.trainers import get_trainer

#Ensemble = namedtuple('Ensemble', ['u', 'idx'])
class Ensemble:
    def __init__(self, Nx, result_dir, idx, model_name, chunk=10):
        self._idx = idx
        self._u = np.zeros(Nx)
        self._x = np.arange(Nx)
        self._chunk = chunk
        self._result_dir = pathlib.Path(result_dir) / f'ens_idx{idx:03}'
        if not self._result_dir.exists():
            self._result_dir.mkdir(parents=True)

        self._diag_it = 0
        self._attrs = {}
        self._attrs['Nx'] = Nx
        self._attrs['model_name'] = model_name

        self.t_all = []
        self.u_all = []

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        self._u = u

    def diag(self, time):
        self.t_all.append(time)
        self.u_all.append(self._u)

        if len(self.t_all) == self._chunk:
            data_vars = {}
            data_vars['u'] = (['time', 'x'], np.asarray(self.u_all))
            data_vars['ensemble_idx'] = self._idx

            coords = {'time': np.asarray(self.t_all),
                      'x': self._x,}

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self._attrs)
            ds.to_netcdf(self._result_dir / f'u{self._diag_it:04d}.nc')

            self._diag_it += 1

class EnsembleGenerator:
    def __init__(self, *args, **kwargs):
        self.nn_settings = kwargs.get('nn_settings')
        self.n_ens = self.nn_settings.get('n_ens')
        self.Nx = self.nn_settings.get('Nx')
        nn_model_type = self.nn_settings.get('nn_model_type', 'Denoising_Diffusion')
        result_dir = self.nn_settings.get('result_dir')
        chunk = self.nn_settings.get('chunk', 10)
        self._obs_interval = 1 # Update based on the training dataset
        self._ensembles = [Ensemble(Nx=self.Nx, result_dir=result_dir, idx=idx, model_name=nn_model_type, chunk=chunk) for idx in range(self.n_ens)]
        _nn_settings = self.nn_settings.copy()
        _nn_settings['batch_size'] = self.n_ens
        _nn_settings['save_intermediate_sampling_imgs'] = True
        _nn_settings['intermediate_sample_dir'] = str( result_dir / 'samples' )

        if nn_model_type == 'nn':
            nn_model_type = 'Denoising_Diffusion'

        self.nn_model = None
        self._obs_interval = 1
        self._seq_len = 1
        if nn_model_type == 'debug':
            pass
        elif nn_model_type in ['Denoising_Diffusion', 'PhysicsInformedDiffusion']:
            dirname = self.nn_settings['model_dir']
            filename = 'settings.json'
            self.nn_model = get_trainer(nn_model_type)(dirname=dirname, filename=filename, **_nn_settings)
            self.nn_model.initialize()
            self._obs_interval = self.nn_model.obs_interval
            self._seq_len = self.nn_model.seq_len
        else:
            raise NotImplementedError(f'Wrong model_type in EnsembleGenerator: {nn_model_type}')

    @property
    def ensembles(self):
        return self._ensembles

    @property
    def obs_interval(self):
        return self._obs_interval

    @property
    def seq_len(self):
        return self._seq_len

    def generate(self, u, observation, time, save=False, zeros=False):
        if zeros:
            for ensemble in self._ensembles:
                ensemble.u = np.zeros(self.Nx)
        else:
            if self.nn_model is not None:
                if len(observation) != self._seq_len:
                    raise ValueError(f'length of observation data is not equal to the seq_len. len(observation): {len(observation)}, seq_len: {self._seq_len}')
                observation = np.asarray(observation)
                samples = self.nn_model.sampling(u=u, obs=observation)
                n_ens, _ = samples.shape
                assert n_ens == self.n_ens, f'batch_size in nn_model is not equal to ensemble size. batch_size: {n_ens}, n_ens: {self.n_ens}'

                for ensemble, sample in zip(self._ensembles, samples):
                    ensemble.u = sample
        
        if save:
            for ensemble in self._ensembles:
                ensemble.diag(time=time)

        return self._ensembles
