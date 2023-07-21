import json
import pathlib
import numpy as np
import xarray as xr
from collections import deque
from ._base_model import _BaseModel
from .lorenz96 import Lorenz96
from .time_integral import get_time_integral_method
from .ensemble_generator import EnsembleGenerator
from .enkf import get_kalman_filter
from .utils import observe_with_zeros, print_details

class Lorenz96_EFDA(_BaseModel):
    """
    Solving Lorenz96 model with EnKF
    du_{j} dt = (u_{j+1} - u_{j-2}) u_{j-1} - u_{j} + F

    Periodic boundary condition
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

        model_name = kwargs.get('model_name')
        self.model_name = f'Lorenz96_{model_name}'
        json_data = kwargs.get('json_data')
        self.result_dir = kwargs.get('result_dir')
        self.result_dir = pathlib.Path(self.result_dir)

        self.default_values = {
            'time_integral_method': 'rkg4th',
            'kalman_filter': 'letkf',
            'n_local': 6,
            'beta': 1.0,
            'sigma': 1.0,
            'da_steps': 1,
            'alpha': 0.5,
            'use_ensemble_mean': True,
        }

        nn_settings = json_data.get('nn_settings')
        self.use_ddib = nn_settings.get('use_ddib', False)
        F = nn_settings.get('F')
        if F is not None:
            json_data['simulation']['F'] = F

        super()._add_json_as_attributes(json_data, self.default_values)
        self._log_dict['model_name'] = self.model_name
        self._log_dict = {**self._log_dict, **self._attrs}

        # Make multiple Lorentz96 solvers
        # 0: using mean (data assimilation by ddib)
        # 1: nudging
        # 2: no-data assimilation
        nb_sims = 3
        self.da = self.__efda_ddib

        self.simulators = [Lorenz96(json_data=json_data, result_dir=f'{self.result_dir}/simulator{idx}', suppress_diag=True)
                           for idx in range(nb_sims)]

        # Initialize the simulator
        self.__spin_up()

        # Load osse data
        in_dir = pathlib.Path(self.out_dir) / self.in_case_name / 'results'
        if not in_dir.exists():
            raise IOError(f'OSSE data does not exist in {in_dir}')
        files = sorted(list(in_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        self.nb_steps = len(ds['time'])
        self.u_obs = ds['u']
        print(ds)
        print(f'--- Loading OSSE data from {in_dir}, {self.nb_steps} steps ---')

        # Initialize Ensemble generator
        self.obs_memory = None
        nn_settings['Nx'] = self.Nx
        nn_settings['result_dir'] = self.result_dir
        nn_settings['chunk'] = self.diag_chunk
        self.ensemble_generator = EnsembleGenerator(nn_settings=nn_settings)
        self.obs_interval = self.ensemble_generator.obs_interval
        self.seq_len = self.ensemble_generator.seq_len
        self.obs_memory = deque([], maxlen=self.seq_len)

        # Noisy observation operator
        self.observe = lambda it: observe_with_zeros(self.u_obs.isel(time=it).values, obs_interval=self.obs_interval, noise_level=self.sigma)

        # Initialize Ensemble Kalman Filter
        self.kf_dict = {
                        'n_ens': self.n_ens,
                        'n_stt': self.Nx,
                        'chunk': self.diag_chunk,
                        'obs_interval': self.obs_interval,
                        'in_dir': str(in_dir),
                        'result_dir': str(self.result_dir),
                        'n_local': self.n_local,
                        'beta': self.beta,
                        'sigma': self.sigma,
                        'nb_steps': self.nb_steps,
                        'da_steps': self.da_steps,
                       }

        self.enkf = get_kalman_filter(self.kalman_filter)(**self.kf_dict)

        # Buffers for diganostics
        self.start_da = False
        self.t_all = []
        self.u_all = []

    def initialize(self, *args, **kwargs):
        mode = kwargs.get('mode')
        self.start_da = mode == 'enable_da'

        self._it = 0
        self._time = 0.

        if self.start_da:
            for sim in self.simulators:
                sim.initialize(mode = 'reset_while_keeping_values')
            print('start efda')
            self.da(it=self._it, time=self._time)

    def __spin_up(self):
        sim0 = self.simulators[0]
        for it in range(self.nbiter):
            sim0.solve()
        u0 = np.copy(sim0.u)

        for sim in self.simulators:
            sim.initialize(u_init=u0)
            for it in range(self.nbiter):
                sim.solve()

            u0 = np.copy(sim.u)

    def finalize(self, *args, **kwargs):
        seconds = kwargs.get('seconds')
        result_dir = kwargs.get('result_dir', str(self.result_dir))
        self.enkf.finalize()

        message = super()._report_elapsed_time(seconds=seconds, n_ens=self.n_ens)
        print(message)
        self._log_dict['elapsed_time'] = seconds
        print_details(log_dict=self._log_dict, dirname=result_dir, filename='log.txt')
        print_details(log_dict=self.kf_dict, dirname=result_dir, filename='log_kf.txt')

    def solve(self, **kwargs):
        """
        Solve Lorenz96 equation for dt
        """

        super()._print_time(message='run')
        for sim_idx, sim in enumerate(self.simulators):
            sim.diag(ens_idx=sim_idx)
            sim.solve()

        # Increment counter after solving one step
        self._it += 1
        self._time += self.dt

        # enkf followed by time integral
        if self._it == self.nb_steps:
            self.start_da = False

        if self._it % self.da_steps == 0:
            self.da(it=self._it, time=self._time)

    def __efda_ddib(self, it, time):
        if self.start_da:
            u_obs = self.observe(it=it)
            u_ref = self.u_obs.isel(time=it).values
            u_sim = None
            if self.use_ddib:
                u_sim = self.simulators[0].u

            # Store the observation data until it reaches the seq len
            self.obs_memory.append(u_obs)
            if len(self.obs_memory) < self.seq_len:
                ensembles = self.ensemble_generator.generate(u=u_sim, observation=self.obs_memory, time=time, save=True, zeros=True)
                self.enkf.save_state(it=it, time=time, ensembles=ensembles, observation=u_obs, reference=u_ref)
                return

            print(f'Apply ensemble free data assimulation method at it: {it}, time: {time:.3f}')

            # Geenerate ensembles by the diffusion model
            ensembles = self.ensemble_generator.generate(u=u_sim, observation=self.obs_memory, time=time, save=True)
            if not self.use_ensemble_mean:
                for ensemble in ensembles:
                    ensemble.u = (1-self.alpha) * self.simulators[0].u + self.alpha * ensemble.u

            ensembles = self.enkf.apply(it=it, ensembles=ensembles, observation=u_obs)

            # helper to compute the difference
            rmse = lambda var, ref: np.sqrt( np.mean( (var - ref) ** 2 ) )
            rmse_self_list = []

            for ensemble in ensembles:
                u = ensemble.u
                u_self = self.simulators[0].u

                rmse_self = rmse(u, u_self)

                rmse_self_list.append(rmse_self)

            rmse_self_list = np.asarray(rmse_self_list)
            self_like_idx = np.argmin(rmse_self_list)

            # Saving
            attrs = {
                     'self_like_idx': self_like_idx,
                     'seq_len': self.seq_len,
                     'obs_interval': self.obs_interval,
                    }
            self.enkf.save_state(it=it, time=time, ensembles=ensembles, observation=u_obs, reference=u_ref, attrs=attrs)

            # Each simulator will be updated in different manners
            if self.use_ensemble_mean:
                # 0: nudging with ensemble mean
                self.simulators[0].u = (1-self.alpha) * self.simulators[0].u + self.alpha * self.enkf.mean
            else:
                # 0: nudging with ensembles then LETKF
                self.simulators[0].u = self.enkf.mean

            # 1: using nuding
            self.simulators[1].u[::self.obs_interval] = (1-self.alpha) * self.simulators[1].u[::self.obs_interval] + self.alpha * u_obs[::self.obs_interval]

            # 2: No data assimilation (Do nothing here)

    def diag(self, **kwargs):
        """
        Diagnostics for every diag_steps (default 1)
        Diagnostics are performed in enkf.save_state method and each simulation
        """
        pass
