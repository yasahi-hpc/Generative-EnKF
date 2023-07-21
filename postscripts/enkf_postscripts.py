import pathlib
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
from numpy import ma
from ._base_postscripts import _BasePostscripts

class EnKF_Postscripts(_BasePostscripts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'Lorenz96-EnKF'
        self.img_sub_dirs = ['contour', 'series'] 
        self.umax = 15.
        self.error_max = 1.5

    def initialize(self, *args, **kwargs):
        self._result_dir, self._img_dir = super()._prepare_dirs(out_dir = self.settings['out_dir'],
                                                                case_name = self.settings['case_name'])

        settings = self.json_data.get('settings')
        nb_runs = settings.get('nb_runs', 1)
        # Only plotting shot_idx = 0
        shot_idx = 0
        self._result_dir = self.result_dir / f'shot{shot_idx:03}' if nb_runs > 1 else self._result_dir

        self.model_name = f'{self.model_name}'
        self._series_img_dir = self._img_dir / 'series'
        if not self._series_img_dir.exists():
            self._series_img_dir.mkdir(parents=True)

    def run(self, *args, **kwargs):
        self._visualize_spatial_structures_compare(name=self.model_name, result_dir=self._result_dir, img_dir=self._img_dir)
        self._visualize_ensembles(name=self.model_name, result_dir=self._result_dir, img_dir=self._img_dir)
        self._visualize_rmse_series(name=self.model_name, result_dir=self._result_dir, img_dir=self._img_dir)
        if self.plot_series:
            self._visualize_da(name=self.model_name, result_dir=self._result_dir, img_dir=self._series_img_dir)

    def finalize(self, *args, **kwargs):
        pass

    def _visualize_spatial_structures_compare(self, name, result_dir, img_dir):
        # Reference
        files = sorted(list(result_dir.glob('enkf*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        obs_interval = ds.attrs.get('obs_interval', 1)
        da_steps = ds.attrs.get('da_steps', 1)

        u_ref = ds['reference'].values
        u_obs = ds['observation'].values
        u_mean = ds['ensemble_mean'].values
        u_spread = ds['ensemble_spread'].values

        u_mask = np.ones_like(u_obs) * -1
        u_mask[::da_steps, ::obs_interval] = 1.

        ny, nx = u_ref.shape
        aspect = nx / ny

        fig, axes = plt.subplots(2, 4, figsize=(40, 20))
        axes[1, 0].set_visible(False)
        axes[1, 3].set_visible(False)

        im = axes[0,0].imshow(u_ref,  origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')
        im = axes[0,1].imshow(ma.masked_where(u_mask <=0, u_obs),  origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')
        im = axes[0,2].imshow(u_mean, origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')
        im = axes[0,3].imshow(u_spread, origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')

        im2 = axes[1,1].imshow(ma.masked_where(u_mask <=0, np.abs(u_obs-u_ref)), origin='lower', cmap='jet', aspect=aspect, vmin=0, vmax=self.error_max, interpolation='none')
        im2 = axes[1,2].imshow(np.abs(u_mean-u_ref), origin='lower', cmap='jet', aspect=aspect, vmin=0, vmax=self.error_max, interpolation='none')

        axes[0,0].set_title('Reference', **self.title_font)
        axes[0,1].set_title('Observation', **self.title_font)
        axes[0,2].set_title(f'{name} (mean)', **self.title_font)
        axes[0,3].set_title(f'{name} (spread)', **self.title_font)
        axes[1,1].set_title(f'Error (Observation)', **self.title_font)
        axes[1,2].set_title(f'Error ({name})', **self.title_font)

        for ax in axes.ravel():
            ax.set_xlabel(r'$x$', **self.axis_font)
            ax.set_ylabel(r'$t$', **self.axis_font)

        cbar  = fig.colorbar(im,  ax=axes[0])
        cbar2 = fig.colorbar(im2, ax=axes[1])

        filename = f'{name}_compare.png'
        fig.savefig(img_dir / filename, bbox_inches='tight')
        plt.close('all')

    def _visualize_ensembles(self, name, result_dir, img_dir, n_cols=8):
        nb_ensembles = len(list(result_dir.glob('ens_idx*')))

        if nb_ensembles < n_cols:
            n_cols = nb_ensembles

        n_rows = nb_ensembles // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24,24/n_cols*n_rows), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))

        for idx, ax in np.ndenumerate(axes.ravel()):
            # Load ensemble data
            path = result_dir / f'ens_idx{int(idx[0]):03}'
            files = sorted(list(path.glob('u*.nc')))
            ds = xr.open_mfdataset(files, engine='netcdf4')

            u = ds['u'].values
            ny, nx = u.shape
            aspect = nx / ny
            im = ax.imshow(u, origin='lower', interpolation='none', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax)

        filename = f'{name}_ensembles.png'
        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.savefig(img_dir / filename, bbox_inches='tight')
        plt.close('all')

    def _visualize_da(self, name, result_dir, img_dir):
        """
        Ensemble data before DA are stored in ens_idx*/u{it}.nc
        observation and Ensemble mean are stored in enkf_stats{it}.nc
        """
        # Reference
        files = sorted(list(result_dir.glob('enkf*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        nbiter = ds.attrs['nbiter']

        nb_ensembles = len(list(result_dir.glob('ens_idx*')))

        # Ensembles
        ds_ens = []
        for idx in range(nb_ensembles):
            path = result_dir / f'ens_idx{idx:03}'
            files = sorted(list(path.glob('u*.nc')))
            _ds = xr.open_mfdataset(files, engine='netcdf4')
            ensemble_idx = np.array( [ _ds['ensemble_idx'].values[0] ] )
            _ds = _ds.assign_coords(ensemble=ensemble_idx)
            _ds = _ds.drop('ensemble_idx')
            ds_ens.append(_ds)
        ds_ens = xr.combine_by_coords(ds_ens)

        for it in range(nbiter):
            filename = f'{name}_da_ens_it{it:05}.png'
            filename2 = f'{name}_da_it{it:05}.png'
            ds_it = ds.isel(time=it).compute()
            ds_ens_it = ds_ens.isel(time=it).compute()

            x = ds_it['x'].values

            mean = ds_it['ensemble_mean'].values
            spread = ds_it['ensemble_spread'].values
            obs = ds_it['observation'].values
            ref = ds_it['reference'].values

            fig, ax = plt.subplots(figsize=(8, 8))
            fig2, ax2 = plt.subplots(figsize=(8, 8))

            ax.plot(x, mean, 'r-', lw=3, label='mean')
            ax.plot(x, obs, 'g--', lw=3, label='obs')
            ax.plot(x, ref, 'b-.', lw=3, label='ref')

            # Ensembles
            for idx in range(nb_ensembles):
                u_tmp = ds_ens_it['u'].isel(ensemble=idx).values
                ax.plot(x, u_tmp, lw=1, alpha=0.3)

            ax.legend(loc='upper left', prop={'size': self.fontsize*0.7})
            ax.grid(which='both', ls='-.')
            ax.axhline(y=0, color='k')
            ax.set_xlabel(r'$x$', **self.axis_font)
            ax.set_ylabel(r'$u$', **self.axis_font)
            ax.set_title(r'$t = {:03}$'.format(it), **self.axis_font)
            ax.set_ylim([-self.umax, self.umax])

            fig.savefig(img_dir / filename, bbox_inches='tight')
            plt.close('all')

            # Averaged values 
            ax2.plot(x, mean, 'r-', lw=3, label='mean')
            ax2.fill_between(x, mean-spread, mean+spread, color='r', alpha=0.2)
            #ax2.errorbar(x, mean, yerr=spread, c='r', ls='-', lw=3, label='mean')
            ax2.plot(x, obs, 'g--', lw=3, label='obs')
            ax2.plot(x, ref, 'b-.', lw=3, label='ref')

            ax2.legend(loc='upper left', prop={'size': self.fontsize*0.7})
            ax2.grid(which='both', ls='-.')
            ax2.axhline(y=0, color='k')
            ax2.set_xlabel(r'$x$', **self.axis_font)
            ax2.set_ylabel(r'$u$', **self.axis_font)
            ax2.set_title(r'$t = {:03}$'.format(it), **self.axis_font)
            ax2.set_ylim([-self.umax, self.umax])

            fig2.savefig(img_dir / filename2, bbox_inches='tight')
            plt.close('all')

    def _visualize_rmse_series(self, name, result_dir, img_dir, start=200):
        # Reference
        files = sorted(list(result_dir.glob('enkf*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        ds = ds.isel(time=slice(start, None))
        #obs_interval = ds.attrs['obs_interval']

        u_ref = ds['reference'].values
        u_obs = ds['observation'].values
        u_mean = ds['ensemble_mean'].values

        time = ds['time']
        rmse = lambda var, obs_interval: np.sqrt( np.mean( ( var[:, ::obs_interval] - u_ref[:, ::obs_interval] )**2, axis=1 ) )

        fig, ax = plt.subplots(figsize=(12,12))
        ax.plot(time, rmse(u_mean, 1), '-r', lw=3, label=f'{self.model_name}')
        ax.plot(time, rmse(u_obs, 1), '-k', lw=3, label='observation')

        ax.set_xlabel('Time', **self.axis_font)
        ax.set_ylabel('RMSE', **self.axis_font)
        ax.set_ylim(ymin=0, ymax=5)
        ax.legend(prop={'size': self.fontsize})
        ax.grid(ls='dashed', lw=1)
        fig.savefig(img_dir / f'rmse_series_{self.model_name}.png', bbox_inches='tight')
        plt.close('all')
