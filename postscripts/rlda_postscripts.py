import pathlib
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
from ._base_postscripts import _BasePostscripts
from simulation.utils import str2num, observe

class RLDA_Postscripts(_BasePostscripts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'Lorenz96-RLDA'
        self.img_sub_dirs = ['contour', 'series'] 
        self.umax = 20.
        self.error_max = 1.5
        self.action_max = 1

    def initialize(self, *args, **kwargs):
        self._result_dir, self._img_dir = super()._prepare_dirs(out_dir = self.settings['out_dir'],
                                                                case_name = self.settings['case_name'])

        self._sim_result_dir = self._result_dir / 'simulator'
        self._rl_result_dir = self._result_dir / 'checkpoint'
        self._series_img_dir = self._img_dir / 'series'
        if not self._series_img_dir.exists():
            self._series_img_dir.mkdir(parents=True)

        episode_dirs = sorted(list(self._sim_result_dir.glob('episode*')))
        self.episodes = [str(episode_dir).split('/')[-1] for episode_dir in episode_dirs]

        # the reference information
        settings = self.json_data.get('settings')
        self.obs_interval = self.json_data['simulation'].get('obs_interval', 1)
        out_dir = settings['out_dir']
        in_case_name = settings['in_case_name']
        self.reference_result_dir = pathlib.Path(out_dir) / in_case_name / 'results'

    def run(self, *args, **kwargs):
        self._visualize_spatial_structures_grid(episodes=self.episodes)
        self._visualize_spatial_errors_grid(episodes=self.episodes)
        self._visualize_spatial_actions_grid(episodes=self.episodes)

        # Comparison over episodes
        self._visualize_rewards()
        self._visualize_rmse(episodes=self.episodes)
        #self._visualize_da(episodes=self.episodes)

    def finalize(self, *args, **kwargs):
        pass

    def _visualize_spatial_structures_grid(self, episodes, n_cols=4):
        nb_cases = len(episodes) + 1 # inlcuding reference

        if nb_cases < n_cols:
            n_cols = nb_cases
        n_rows = int( (nb_cases - 1) / n_cols ) + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(32,32/n_cols*n_rows), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        if type(axes) is not np.ndarray:
            axes = np.asarray(axes)

        for idx, ax in np.ndenumerate(axes.ravel()):
            idx = idx[0]
            if idx >= nb_cases:
                ax.set_visible(False)
                continue

            # Load results
            if idx == 0:
                case_name = 'reference'
                sub_result_dir = self.reference_result_dir
            else:
                case_name = episodes[idx-1]
                sub_result_dir = self._sim_result_dir / case_name
            p = self._visualize_spatial_structures(sub_result_dir, ax, case_name)
        cbar = fig.colorbar(p, ax=axes.ravel().tolist())
        fig.savefig(self._img_dir / f'spatio_temporal_{self.model_name}.png', bbox_inches='tight')
        plt.close('all')

    def _visualize_spatial_structures(self, result_dir, ax, case_name):
        # Predictions
        files = sorted(list(result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        if 'u_filtered' in ds:
            u = ds['u_filtered'].values
        else:
            u = ds['u'].values

        ny, nx = u.shape
        aspect = nx / ny

        im = ax.imshow(u,  origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')
        ax.set_title(case_name, **self.title_font)
        ax.set_xlabel(r'$x$', **self.axis_font)
        ax.set_ylabel(r'$t$', **self.axis_font)

        return im

    def _visualize_spatial_errors_grid(self, episodes, n_cols=4):
        nb_cases = len(episodes)

        if nb_cases < n_cols:
            n_cols = nb_cases
        n_rows = int( (nb_cases - 1) / n_cols ) + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(32,32/n_cols*n_rows), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        if type(axes) is not np.ndarray:
            axes = np.asarray(axes)

        for idx, ax in np.ndenumerate(axes.ravel()):
            idx = idx[0]
            if idx >= nb_cases:
                ax.set_visible(False)
                continue

            # Load results
            case_name = episodes[idx]
            sub_result_dir = self._sim_result_dir / case_name
            p = self._visualize_spatial_errors(sub_result_dir, ax, case_name)
        cbar = fig.colorbar(p, ax=axes.ravel().tolist())
        fig.savefig(self._img_dir / f'spatio_temporal_errors_{self.model_name}.png', bbox_inches='tight')
        plt.close('all')

    def _visualize_spatial_errors(self, result_dir, ax, case_name):
        # Predictions
        files = sorted(list(result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        u = ds['u_filtered'].values

        # Reference
        files = sorted(list(self.reference_result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        u_ref = ds['u'].values

        ny, nx = u.shape
        aspect = nx / ny

        im = ax.imshow(np.abs(u - u_ref),  origin='lower', cmap='jet', aspect=aspect, vmin=0, vmax=self.error_max, interpolation='none')
        ax.set_title(case_name, **self.title_font)
        ax.set_xlabel(r'$x$', **self.axis_font)
        ax.set_ylabel(r'$t$', **self.axis_font)

        return im

    def _visualize_spatial_actions_grid(self, episodes, n_cols=4):
        nb_cases = len(episodes)

        if nb_cases < n_cols:
            n_cols = nb_cases
        n_rows = int( (nb_cases - 1) / n_cols ) + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(32,32/n_cols*n_rows), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        if type(axes) is not np.ndarray:
            axes = np.asarray(axes)

        for idx, ax in np.ndenumerate(axes.ravel()):
            idx = idx[0]
            if idx >= nb_cases:
                ax.set_visible(False)
                continue

            # Load results
            case_name = episodes[idx]
            sub_result_dir = self._sim_result_dir / case_name
            p = self._visualize_spatial_actions(sub_result_dir, ax, case_name)
        cbar = fig.colorbar(p, ax=axes.ravel().tolist())
        fig.savefig(self._img_dir / f'spatio_temporal_actions_{self.model_name}.png', bbox_inches='tight')
        plt.close('all')

    def _visualize_spatial_actions(self, result_dir, ax, case_name):
        # Predictions
        files = sorted(list(result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        action = ds['action'].values
        action = (action * 2) - 1.

        ny, nx = action.shape
        aspect = nx / ny

        im = ax.imshow(action,  origin='lower', cmap='seismic', aspect=aspect, vmin=-self.action_max, vmax=self.action_max, interpolation='none')
        ax.set_title(case_name, **self.title_font)
        ax.set_xlabel(r'$x$', **self.axis_font)
        ax.set_ylabel(r'$t$', **self.axis_font)

        return im

    def _visualize_rewards(self):
        # first checkpoint includes 0 to n_episodes data size different
        files = sorted(list(self._rl_result_dir.glob('checkpoint*.nc')))
        ds = xr.open_mfdataset(files, concat_dim='episodes', compat='no_conflicts', combine='nested')

        episodes = ds['episodes']
        rewards = ds['train_rewards']

        fig, ax = plt.subplots(figsize=(12,12))
        ax.plot(episodes, rewards, ls='-', lw=3, label='reward')
        ax.set_xlabel('episodes', **self.axis_font)
        ax.set_ylabel('rewards', **self.axis_font)
        ax.legend(prop={'size': self.fontsize})
        ax.grid(ls='dashed', lw=1)
        fig.savefig(self._img_dir / f'rewards_{self.model_name}.png', bbox_inches='tight')
        plt.close('all')

    def _visualize_rmse(self, episodes, start=200):
        # Reference
        files = sorted(list(self.reference_result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        u_ref = ds['u'].isel(time=slice(start, None)).values

        rmse_list = []
        rmse_obs_list = []
        episode_list = []

        rmse = lambda var, obs_interval: np.sqrt( np.mean( ( var[:, ::obs_interval] - u_ref[:, ::obs_interval] )**2) )

        for episode in episodes:
            sub_result_dir = self._sim_result_dir / episode
            files = sorted(list(sub_result_dir.glob('u*.nc')))
            ds = xr.open_mfdataset(files, engine='netcdf4')

            u_model = ds['u_filtered'].isel(time=slice(start, None)).values
            u_obs = ds['u_obs'].isel(time=slice(start, None)).values

            rmse_tmp = rmse(u_model, 1)
            rmse_obs_tmp = rmse(u_obs, self.obs_interval)

            rmse_obs_list.append(rmse_obs_tmp)
            rmse_list.append(rmse_tmp)
            i_episode = episode.replace('episode', '')
            episode_list.append( str2num(i_episode) )

        rmse_list = np.asarray(rmse_list)
        rmse_obs_list = np.asarray(rmse_obs_list)
        episode_list = np.asarray(episode_list)

        fig, ax = plt.subplots(figsize=(12,12))
        ax.plot(episode_list, rmse_list, 'r-*', lw=3, markersize=18, label='RL-DA')
        ax.plot(episode_list, rmse_obs_list, '--k', lw=3, label='observation')
        ax.set_xlabel('episodes', **self.axis_font)
        ax.set_ylabel('RMSE', **self.axis_font)
        ax.set_ylim(ymin=0)
        ax.legend(prop={'size': self.fontsize})
        ax.grid(ls='dashed', lw=1)
        fig.savefig(self._img_dir / f'rmse_{self.model_name}.png', bbox_inches='tight')
        plt.close('all')

    def _visualize_da(self, episodes):
        # Reference
        files = sorted(list(self.reference_result_dir.glob('u*.nc')))
        ds_ref = xr.open_mfdataset(files, engine='netcdf4')
        x = ds_ref['x'].values
        nbiter = ds_ref.attrs['nbiter']

        for episode in episodes:
            sub_result_dir = self._sim_result_dir / episode
            files = sorted(list(sub_result_dir.glob('u*.nc')))
            ds = xr.open_mfdataset(files, engine='netcdf4')
            x_obs = ds['x_obs'].values

            diff_sum = 0
            for it in range(nbiter):
                ds_sub = ds.isel(time=it)

                u_ref = ds_ref['u'].isel(time=it).values
                u_model = ds_sub['u_filtered'].values
                u_prev = ds_sub['u'].values
                u_obs = ds_sub['u_obs'].values
                action = ds_sub['action'].values

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.plot(x, u_prev, 'r-', lw=3, label='prev')
                ax.plot(x, u_model, 'm--', lw=3, label='filtered')
                ax.plot(x_obs, u_obs, 'g--', lw=3, label='obs')
                ax.plot(x, u_ref, 'b-.', lw=3, label='ref')

                # Plot actions
                for prev_x, prev_y, next_y in zip(x, u_prev, u_model):
                    dx = 0
                    dy = next_y - prev_y
                    ax.arrow(x=prev_x, y=prev_y, dx=0, dy=dy, color='k', length_includes_head=True, width=0.1)

                ax.legend(loc='upper left', ncol=2, prop={'size': self.fontsize*0.7})
                ax.grid(which='both', ls='-.')
                ax.axhline(y=0, color='k')
                ax.set_xlabel(r'$x$', **self.axis_font)
                ax.set_ylabel(r'$u$', **self.axis_font)
                ax.set_title(r'$t = {:03}$'.format(it), **self.title_font)
                ax.set_xlim([0, 40])
                ax.set_ylim([-self.umax, self.umax])

                sub_img_dir = self._img_dir / f'series/{episode}'
                if not sub_img_dir.exists():
                    sub_img_dir.mkdir(parents=True)
                fig.savefig(sub_img_dir / f'{self.model_name}_da_ep{episode}_it{it:05}.png', bbox_inches='tight')
                plt.close('all')
