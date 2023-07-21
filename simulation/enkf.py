import abc
import numpy as np
import xarray as xr
import pathlib
import itertools
import copy
from einops import rearrange

class _BaseEnKF(abc.ABC):
    """
    Abstract base model class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attrs = kwargs.copy()

        self.n_ens = kwargs.get('n_ens')
        self.chunk = kwargs.get('chunk', 10)
        self.obs_interval = kwargs.get('obs_interval', 1)
        self.n_points = kwargs.get('n_stt')

        in_dir = kwargs.get('in_dir')
        in_dir = pathlib.Path(in_dir)
        if not in_dir.exists():
            raise IOError(f'OSSE data does not exist in {in_dir}')

        result_dir = kwargs.get('result_dir')
        self.result_dir = pathlib.Path(result_dir)

        self.t_all = []
        self.mean_all = []
        self.spread_all = []
        self.obs_all = []
        self.ref_all = []

        self._diag_it = 0
        self._mean = np.zeros(self.n_points)
        self._spread = np.zeros(self.n_points)

    @property
    def nb_steps(self):
        return self._nb_steps

    @property
    def mean(self):
        return self._mean

    @property
    def spread(self):
        return self._spread

    @abc.abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError()

    def save_state(self, *args, **kwargs):
        it = kwargs.get('it')
        time = kwargs.get('time')
        ensembles = kwargs.get('ensembles')
        observation = kwargs.get('observation')
        reference = kwargs.get('reference')

        u_ens = []
        for ensemble in ensembles:
            u_ens.append(ensemble.u)

        # (n_ens, n_points)
        u_ens = np.asarray(u_ens)

        ensemble_average = lambda var: np.mean(var, axis=0)
        ensemble_spread = lambda var: np.var(var, axis=0)

        x_mean = ensemble_average(u_ens)
        x_spread = ensemble_spread(u_ens)

        self._mean = x_mean
        self._spread = x_spread

        self.t_all.append(time)
        self.mean_all.append(x_mean)
        self.spread_all.append(x_spread)
        self.obs_all.append(observation)
        self.ref_all.append(reference)

        if len(self.t_all) == self.chunk:
            self.__to_netcdf(*args, **kwargs)

            self.t_all = []
            self.mean_all = []
            self.spread_all = []
            self.obs_all = []
            self.ref_all = []
            self._diag_it += 1

    def finalize(self, *args, **kwargs):
        if self.t_all:
            self.__to_netcdf(*args, **kwargs)

    def __to_netcdf(self, *args, **kwargs):
        attrs = kwargs.get('attrs')

        data_vars = {}
        data_vars['ensemble_mean'] = (['time', 'x'], np.squeeze(self.mean_all))
        data_vars['ensemble_spread'] = (['time', 'x'], np.squeeze(self.spread_all))
        data_vars['observation'] = (['time', 'x'], np.squeeze(self.obs_all))
        data_vars['reference'] = (['time', 'x'], np.squeeze(self.ref_all))

        coords = {'time': np.asarray(self.t_all),
                  'x': np.arange(self.n_points)}

        _attrs = copy.deepcopy(self.attrs)
        if attrs is not None:
            _attrs.update(attrs)
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=_attrs)
        ds.to_netcdf(self.result_dir / f'enkf_stats{self._diag_it:04d}.nc', engine='netcdf4')

class EnKF(_BaseEnKF):
    """
    Ensemble Transform Karman Filter (Bishop 2001)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'EnKF'
        self.n_stt = kwargs.get('n_stt')
        self.chunk = kwargs.get('chunk', 10)

        # n_stt must be divisible by obs_interval
        if self.n_stt % self.obs_interval != 0:
            raise ValueError(f'n_stt must be divisible by obs_interval. n_stt: {self.n_stt}, obs_interval: {self.obs_interval}')

        self.n_obs = self.n_stt // self.obs_interval

        self.R = kwargs.get('R', 1)

        # Ensembles of states
        self.X = np.zeros((self.n_stt, self.n_ens))

        # Y = H(X)
        self.Y = np.zeros((self.n_obs, self.n_ens))

        # Observations
        self.yo = np.zeros((self.n_obs, 1))

        # (n_obs, n_obs)
        self.R = np.eye(self.n_obs)

        print(f'--- {self.model_name} is initialized successfully ---')

    def apply(self, *args, **kwargs):
        """
        """

        it = kwargs.get('it')
        ensembles = kwargs.get('ensembles')
        observation = kwargs.get('observation')

        for idx, ensemble in enumerate(ensembles):
            u = np.copy(ensemble.u)
            self.X[:,idx] = u
            self.Y[:,idx] = u[::self.obs_interval]

        # Observation
        self.yo = observation[::self.obs_interval]
        self.yo = self.yo[:, np.newaxis]

        # Check data shape
        assert self.yo.shape == (self.n_obs, 1), f'self.yo.shape: {self.yo.shape} should be {(self.n_obs, 1)}'

        # Solve system: X = X + K@(y-H(X))
        self.__solve()

        # Update ensembles
        for idx, ensemble in enumerate(ensembles):
            ensemble.u = self.X[:,idx]

        return ensembles

    def __solve(self):
        ensemble_average = lambda var: np.mean(var, axis=-1, keepdims=True)
        matmat = lambda A, B: np.einsum('i j, j k-> i k', A, B)
        transpose = lambda A: rearrange(A, 'i j -> j i')

        x_mean = ensemble_average(self.X)
        y_mean = ensemble_average(self.Y)

        # dX = X - mean(X)
        dX = self.X - x_mean

        # dY = Y - mean(Y)
        dY = self.Y - y_mean

        # Kalman gain
        # K = 1/(M-1) dX * dYT * ( 1/(M-1) dY * dYT + R )^-1
        dYT = transpose(dY)
        Z = np.linalg.inv( matmat(dY, dYT) / (self.n_ens-1) + self.R )
        K = matmat(dX, matmat(dYT, Z)) / (self.n_ens-1)

        # update 
        self.X = self.X + matmat(K, (self.yo-self.Y))

class LETKF(_BaseEnKF):
    """
    Local Transform Ensemble Karman Filter (Bishop 2001)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'LETKF'

        self.R = kwargs.get('R', 1)
        self.beta = kwargs.get('beta', 1.0)
        self.sigma = kwargs.get('sigma', 1.0)
        self.n_batch = kwargs.get('n_batch', self.n_points)
        self.n_local = kwargs.get('n_local', 6)
        self.n_stt = 1 + 2 * self.n_local # +/- n_local points
        self.n_obs = self.n_stt

        # Ensembles of states
        self.X = np.zeros((self.n_batch, self.n_stt, self.n_ens))

        # Y = H(X)
        self.Y = np.zeros((self.n_batch, self.n_obs, self.n_ens))

        # Observations
        self.yo = np.zeros((self.n_batch, self.n_obs, 1))

        # (n_batch, n_obs, n_obs)
        self.R = np.zeros((self.n_batch, self.n_obs, self.n_obs))
        for i_batch in range(self.n_batch):
            for j_obs, i_obs in itertools.product(range(self.n_obs), range(self.n_obs)):
                idx = (j_obs - self.n_local + self.n_points + i_batch) % self.n_points
                s = self.sigma * int(j_obs==i_obs) if idx % self.obs_interval == 0 else 0
                self.R[i_batch,j_obs,i_obs] = np.float64(s)
        #self.R = np.repeat( np.eye(self.n_obs)[np.newaxis], self.n_batch, axis=0 )

        # (n_batch, n_ens, n_ens)
        self.I = np.repeat( np.eye(self.n_ens)[np.newaxis], self.n_batch, axis=0 )

        print(f'--- {self.model_name} is initialized successfully ---')

    def apply(self, *args, **kwargs):
        it = kwargs.get('it')
        ensembles = kwargs.get('ensembles')
        observation = kwargs.get('observation')

        # Set simulation data
        for i_batch in range(self.n_batch):
            for i_state in range(self.n_stt):
                idx = (i_state - self.n_local + self.n_points + i_batch) % self.n_points

                # Set obervation data
                self.yo[i_batch, i_state] = observation[idx%self.n_points] if idx % self.obs_interval == 0 else 0

                # Set ensemble data
                for i_ens, ensemble in enumerate(ensembles):
                    u = np.copy(ensemble.u)
                    self.X[i_batch, i_state, i_ens] = u[idx%self.n_points]
                    self.Y[i_batch, i_state, i_ens] = u[idx%self.n_points] if idx % self.obs_interval == 0 else 0

        self.__solve()

        # Update ensembles
        for i_ens, ensemble in enumerate(ensembles):
            for i_batch in range(self.n_batch):
                ensemble.u[i_batch%self.n_points] = self.X[i_batch, self.n_local%self.n_points, i_ens]

        return ensembles

    def __solve(self):
        ensemble_average = lambda var: np.mean(var, axis=-1, keepdims=True)
        matmat = lambda A, B: np.einsum('b i j, b j k-> b i k', A, B)
        transpose = lambda A: rearrange(A, 'b i j -> b j i')

        x_mean = ensemble_average(self.X)
        y_mean = ensemble_average(self.Y)

        # dX = X - mean(X)
        dX = self.X - x_mean

        # dY = Y - mean(Y)
        dY = self.Y - y_mean

        # dyo = yo - mean(Y)
        dyo = self.yo - y_mean

        # Q = (Ne-1)I/beta + (dY.T * inv(R) * dY)
        # Q: (n_batch, n_ens, n_ens)
        # dYT: (n_batch, n_ens, n_obs)
        dYT = transpose(dY)
        Q = (self.n_ens-1) * self.I / self.beta + matmat(dYT, matmat(self.R, dY))

        # Eigen value decomposition
        # Q = V*D*V.T
        # d: (n_batch, n_ens), v: (n_batch, n_ens, n_ens)
        # inv_D: (n_batch, n_ens, n_ens)
        d, v = np.linalg.eigh(Q)
        #d, v = np.linalg.eig(Q)
        inv_D = np.zeros_like(self.I)
        for i in range(self.n_batch):
            inv_D[i] = np.eye(self.n_ens) * 1 / d[i]

        # P = V * inv(D) * V.T
        # P: (n_batch, n_ens, n_ens)
        vT = transpose(v)
        P = matmat(v, matmat(inv_D, vT))

        # w = P * (dY.T * inv(R) * dyo)
        tmp = matmat(dYT, matmat(self.R, dyo))
        w = matmat(P, tmp)

        # W = sqrt(Ne-1) * V * inv(sqrt(D)) * V.T
        inv_sqrt_D = np.zeros_like(self.I)
        for i in range(self.n_batch):
            inv_sqrt_D[i] = np.eye(self.n_ens) * np.sqrt(1 / d[i])
        W = np.sqrt(self.n_ens-1) * matmat(v, matmat(inv_sqrt_D, vT)) 

        # Update
        W = W + w
        self.X = x_mean + matmat(dX, W)

class NoneKF(_BaseEnKF):
    """
    Do not apply any data assimilation, used for debuging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'NoneDA'

        print(f'--- {self.model_name} is initialized successfully ---')

    def apply(self, *args, **kwargs):
        """
        """
        ensembles = kwargs.get('ensembles')

        return ensembles

def get_kalman_filter(method_name):
    METHODS = {
               'enkf': EnKF,
               'letkf': LETKF,
               'none_kf': NoneKF,
              }
    
    for n, m in METHODS.items():
        if n.lower() == method_name.lower():
            return m
    
    raise ValueError(f'method {method_name} is not defined')
