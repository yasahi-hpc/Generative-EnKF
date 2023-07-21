import pathlib
import numpy as np
import xarray as xr
from ._base_model import _BaseModel
from .time_integral import get_time_integral_method
from .utils import print_details

class Lorenz96(_BaseModel):
    """
    Solving Lorenz96 model
    du_{j} dt = (u_{j+1} - u_{j-2}) u_{j-1} - u_{j} + F

    Periodic boundary condition
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

        self.model_name = 'Lorenz96'

        self.default_values = {
            'time_integral_method': 'rkg4th',
            'perturbation_amp': 1.e-5,
        }

        json_data = kwargs.get('json_data')
        self.result_dir = kwargs.get('result_dir')
        self.result_dir = pathlib.Path(self.result_dir)
        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True)
        self.suppress_diag = kwargs.get('suppress_diag', False) # For EnKF
        self.random_init = kwargs.get('random_init', False)
        self.verbose = not self.suppress_diag
        self.max_value = 100.

        super()._add_json_as_attributes(json_data, self.default_values)
        self._log_dict['model_name'] = self.model_name
        self._log_dict = {**self._log_dict, **self._attrs}

        # define initial u
        if self.random_init:
            # same as in M. Peyron et al, Latent Space Data Assimilation by using Deep Learning, 2021
            #self.u0 = np.random.randn(self.Nx) * (self.u0_factor-1) + self.F
            self.u0 = np.random.randn(self.Nx) * 0.01 + self.F
            self.u0 += np.random.randn(self.Nx)
        else:
            self.u0 = np.ones(self.Nx) * self.F
            self.u0[self.u0_idx] *= self.u0_factor

        self.u = np.copy(self.u0)

        # Grid
        self.x = np.arange(self.Nx)

        self.time_integral = get_time_integral_method(self.time_integral_method)(h = self.dt)

        # Buffers for diagnostics
        self.t_all = []
        self.u_all = []
        self.u_obs_all = []
        self.action_all = []
        self.u_filtered_all = []

        self._terminated = False
        self.obs_interval = 1

    @property
    def terminated(self):
        return self._terminated

    def initialize(self, *args, **kwargs):
        mode = kwargs.get('mode')
        self._it = 0
        self._time = 0.

        if mode == 'perturbed':
            # Add perturbation and reset timing
            print('Add perturbation')
            self.u += self.perturbation_amp * np.random.rand(*self.u.shape)
        elif mode == 'reset_while_keeping_values':
            self.suppress_diag = False
            self.verbose = False
        elif mode == 'reset':
            self.u = np.copy(self.u0)
            self._terminated = False

        u_init = kwargs.get('u_init')

        if u_init is not None:
            # Initialize u with u_init and start simulation
            if u_init.shape != self.u.shape:
                raise ValueError('self.u.shape and u_init.shape must be identical')
            self.u = np.copy(u_init)

    def finalize(self, *args, **kwargs):
        seconds = kwargs.get('seconds')
        result_dir = kwargs.get('result_dir', str(self.result_dir))

        message = super()._report_elapsed_time(seconds=seconds)
        if self.verbose:
            print(message)

        self._log_dict['elapsed_time'] = seconds
        print_details(log_dict=self._log_dict, dirname=result_dir, filename='log.txt')

    def solve(self, **kwargs):
        """
        Solve Lorenz96 equation for dt
        """

        if self.verbose:
            print(f'it: {self._it}, time: {self._time:.3f}')

        for step in range(self.time_integral.order):
            self.u = self.time_integral.advance(f=self.__RHS,
                                                y=self.u,
                                                step=step)

        # Increment counter after solving one step
        self._it += 1
        self._time += self.dt

        is_nan_included = lambda arr: np.isnan(arr).any()

        if np.max(np.abs(self.u)) > self.max_value or is_nan_included(self.u):
            self._terminated = True

    def __RHS(self, u):
        # Compute du/dt 
        u_p1 = np.roll(u, -1)
        u_m1 = np.roll(u,  1)
        u_m2 = np.roll(u,  2)

        dudt = (u_p1 - u_m2) * u_m1 - u + self.F
        return dudt

    def diag(self, **kwargs):
        """
        Diagnostics for every diag_steps (default 1)
        """
        if self.suppress_diag:
            return

        ens_idx = kwargs.get('ens_idx')
        action = kwargs.get('action')
        u_filtered = kwargs.get('u_filtered')
        u_obs = kwargs.get('u_obs')

        def save():
            data_vars = {}
            data_vars['u'] = (['time', 'x'], np.asarray(self.u_all))

            if ens_idx is not None:
                data_vars['ensemble_idx'] = ens_idx

            if action is not None:
                data_vars['action'] = (['time', 'x'], np.asarray(self.action_all))

            if u_filtered is not None:
                data_vars['u_filtered'] = (['time', 'x'], np.asarray(self.u_filtered_all))
            
            coords = {'time': np.asarray(self.t_all),
                      'x': self.x,}

            if u_obs is not None:
                data_vars['u_obs'] = (['time', 'x_obs'], np.asarray(self.u_obs_all))
                coords['x_obs'] = self.x_obs

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self._attrs)
            ds.to_netcdf(self.result_dir / f'u{self._diag_it:04d}.nc')

        if self._it % self.diag_steps == 0:
            self.t_all.append(self._time)
            self.u_all.append(self.u)

            if action is not None:
                self.action_all.append(action)

            if u_filtered is not None:
                self.u_filtered_all.append(u_filtered)

            if u_obs is not None:
                self.u_obs_all.append(u_obs)
                self.obs_interval = len(self.x) // len(u_obs)
                self.x_obs = self.x[::self.obs_interval]

            if len(self.t_all) == self.diag_chunk:
                save()

                self.t_all = []
                self.u_all = []
                self.action_all = []
                self.u_filtered_all = []
                self.u_obs_all = []

                self._diag_it += 1
