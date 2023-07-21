import pathlib
import numpy as np
import xarray as xr
from ._base_model import _BaseModel
from .lorenz96 import Lorenz96
from .time_integral import get_time_integral_method
from .enkf import get_kalman_filter
from .utils import observe, print_details

class Lorenz96_EnKF(_BaseModel):
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
            'kalman_filter': 'enkf',
            'n_local': 6,
            'beta': 1.0,
            'sigma': 1.0,
            'da_steps': 1,
            'random_init': False,
        }

        super()._add_json_as_attributes(json_data, self.default_values)
        self._log_dict['model_name'] = self.model_name
        self._log_dict = {**self._log_dict, **self._attrs}

        # Make n_ens models of Lorentz96
        print(f'--- Preparing {self.n_ens} ensemble simulations ---')
        self.ensembles = [Lorenz96(json_data=json_data, result_dir=f'{self.result_dir}/ens_idx{idx:03}', suppress_diag=True, random_init=self.random_init) for idx in range(self.n_ens)]

        # Initialize each ensembles with different spin-up state
        strategy = 'random initialization' if self.random_init else 'spin up'
        if not self.random_init:
            self.__spin_up()
        print(f'--- {self.n_ens} ensembles are initialized with {strategy} successfully ---')

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

        # Noisy observation
        self.observe = lambda it: observe(self.u_obs.isel(time=it).values, noise_level=self.sigma)

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
                       }
        self.kf_dict.update(self._attrs)

        self.enkf = get_kalman_filter(self.kalman_filter)(**self.kf_dict)

        # Buffers for diagnostics
        self.start_da = False
        self.t_all = []
        self.u_all = []

    def initialize(self, *args, **kwargs):
        mode = kwargs.get('mode')
        self.start_da = mode == 'enable_da'

        self._it = 0
        self._time = 0.

        if self.start_da:
            for ensemble in self.ensembles:
                ensemble.initialize(mode = 'reset_while_keeping_values')
            print('start enkf')
            self.__enkf(it=self._it, time=self._time)

    def __spin_up(self):
        ensemble0 = self.ensembles[0]
        for it in range(self.nbiter):
            ensemble0.solve()
        u0 = np.copy(ensemble0.u)

        for ensemble in self.ensembles:
            ensemble.initialize(u_init=u0)
            for it in range(self.nbiter):
                ensemble.solve()

            u0 = np.copy(ensemble.u)

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

        print(f'it: {self._it}, time: {self._time:.3f}')
        for ens_idx, ensemble in enumerate(self.ensembles):
            ensemble.diag(ens_idx=ens_idx)
            ensemble.solve()

        # Increment counter after solving one step
        self._it += 1
        self._time += self.dt

        # enkf followed by time integral
        if self._it == self.nb_steps:
            self.start_da = False

        self.__enkf(it=self._it, time=self._time)

    def __enkf(self, it, time):
        if self.start_da:
            u_obs = self.observe(it=it)
            u_ref = self.u_obs.isel(time=it).values

            if self._it % self.da_steps == 0:
                print(f'Apply {self.kalman_filter} at it: {it}, time: {time:.3f}')
                self.ensembles = self.enkf.apply(it=it, ensembles=self.ensembles, observation=u_obs)

            self.enkf.save_state(it=it, time=time, ensembles=self.ensembles, observation=u_obs, reference=u_ref)

    def diag(self, **kwargs):
        """
        Diagnostics for every diag_steps (default 1)
        Diagnostics are performed in enkf.save_state method and each simulation
        """
        pass
