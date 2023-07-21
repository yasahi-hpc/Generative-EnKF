import pathlib
import numpy as np
import xarray as xr
from ._base_model import _BaseModel
from .time_integral import get_time_integral_method
from .utils import observe, print_details

class Lorenz96_Nudging(_BaseModel):
    """
    Solving Lorenz96 model with Nuding
    du_{j} dt = (u_{j+1} - u_{j-2}) u_{j-1} - u_{j} + F

    Periodic boundary condition
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

        self.model_name = 'Lorenz96_Nudging'

        self.default_values = {
            'time_integral_method': 'rkg4th',
            'obs_interval': 1,
            'noise_level': 1,
            'da_steps': 1,
        }

        json_data = kwargs.get('json_data')
        self.result_dir = kwargs.get('result_dir')
        self.result_dir = pathlib.Path(self.result_dir)
        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True)

        super()._add_json_as_attributes(json_data, self.default_values)
        self._log_dict['model_name'] = self.model_name
        self._log_dict = {**self._log_dict, **self._attrs}

        # define initial u
        self.u0 = np.ones(self.Nx) * self.F
        self.u0[self.u0_idx] *= self.u0_factor

        self.u = np.copy(self.u0)

        # Grid
        self.x = np.arange(self.Nx)

        self.time_integral = get_time_integral_method(self.time_integral_method)(h = self.dt)

        # Load osse data 
        in_dir = pathlib.Path(self.out_dir) / self.in_case_name / 'results'
        if not in_dir.exists():
            raise IOError(f'OSSE data does not exist in {in_dir}')
        files = sorted(list(in_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        self.nb_steps = len(ds['time'])
        self.da_u_obs = ds['u']
        print(ds)
        print(f'--- Loading OSSE data from {in_dir}, {self.nb_steps} steps ---')

        self.observe = lambda it: observe(self.da_u_obs.isel(time=it).values, noise_level=self.noise_level)

        # Buffers for diagnostics
        self.t_all = []
        self.u_all = []
        self.u_obs_all = []
        self.u_pre_all = []

    def initialize(self, *args, **kwargs):
        mode = kwargs.get('mode')
        self.start_da = mode == 'enable_da'

        self._it = 0
        self._time = 0.

        if self.start_da:
            print('start nudging')
            self.__nuding(it=self._it)

    def finalize(self, *args, **kwargs):
        if self.t_all:
            self.__save()

        seconds = kwargs.get('seconds')
        result_dir = kwargs.get('result_dir', str(self.result_dir))

        message = f'It took {seconds} [s] to run {self.model_name} for {self.nbiter} iterations'
        self._log_dict['elapsed_time'] = seconds
        print_details(log_dict=self._log_dict, dirname=result_dir, filename='log.txt')

    def solve(self, **kwargs):
        """
        Solve Lorenz96 equation for dt
        """

        print(f'it: {self._it}, time: {self._time:.3f}')

        for step in range(self.time_integral.order):
            self.u = self.time_integral.advance(f=self.__RHS,
                                                y=self.u,
                                                step=step)

        # Increment counter after solving one step
        self._it += 1
        self._time += self.dt

        # nuding followed by time integral
        if self._it == self.nb_steps:
            self.start_da = False
        self.__nuding(it=self._it)

    def __nuding(self, it):
        if self.start_da and self._it % self.da_steps == 0:
            print(f'Nudging at t = {it}')
            self.u_obs = self.observe(it)
            self.u_pre = self.u.copy()
            self.u[::self.obs_interval] = (1-self.alpha) * self.u[::self.obs_interval] + self.alpha * self.u_obs[::self.obs_interval]

    def __RHS(self, u):
        # Compute du/dt 
        u_p1 = np.roll(u, -1)
        u_m1 = np.roll(u,  1)
        u_m2 = np.roll(u,  2)

        dudt = (u_p1 - u_m2) * u_m1 - u + self.F
        return dudt

    def __save(self):
        data_vars = {}
        data_vars['u'] = (['time', 'x'], np.asarray(self.u_all))
        data_vars['u_pre'] = (['time', 'x'], np.asarray(self.u_pre_all))
        data_vars['u_obs'] = (['time', f'x_obs{self.obs_interval}'], np.asarray(self.u_obs_all))

        coords = {'time': np.asarray(self.t_all),
                  'x': self.x,
                  f'x_obs{self.obs_interval}': self.x[::self.obs_interval]}

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self._attrs)
        ds.to_netcdf(self.result_dir / f'u{self._diag_it:04d}.nc')

    def diag(self, **kwargs):
        """
        Diagnostics for every diag_steps (default 1)
        """

        if self._it % self.diag_steps == 0:
            self.t_all.append(self._time)
            self.u_all.append(self.u)
            self.u_pre_all.append(self.u_pre)
            self.u_obs_all.append(self.u_obs)

            if len(self.t_all) == self.diag_chunk:
                self.__save()

                self.t_all = []
                self.u_all = []
                self.u_pre_all = []
                self.u_obs_all = []

                self._diag_it += 1
