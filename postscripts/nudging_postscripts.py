import pathlib
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
from ._base_postscripts import _BasePostscripts

class Nudging_Postscripts(_BasePostscripts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'Lorenz96-Nudging'
        self.img_sub_dirs = ['contour'] 

    def initialize(self, *args, **kwargs):
        self._result_dir, self._img_dir = super()._prepare_dirs(out_dir = self.settings['out_dir'],
                                                                case_name = self.settings['case_name'])

        settings = self.json_data.get('settings')
        nb_runs = settings.get('nb_runs', 1)
        # Only plotting shot_idx = 0
        shot_idx = 0
        self._result_dir = self.result_dir / f'shot{shot_idx:03}' if nb_runs > 1 else self._result_dir

        self.model_name = f'{self.model_name}'

    def run(self, *args, **kwargs):
        super()._visualize_spatial_structures(name=self.model_name, result_dir=self._result_dir, img_dir=self._img_dir)
        if self.in_dir is not None:
            in_case_name = self.settings['in_case_name']
            names = [in_case_name, self.model_name]
            result_dirs = [self.in_dir, self._result_dir]
            self._visualize_spatial_structures_compare(names=names, result_dirs=result_dirs, img_dir=self._img_dir)
            self._visualize_rmse_series(result_dirs=result_dirs, img_dir=self._img_dir)

    def finalize(self, *args, **kwargs):
        pass

    def _visualize_spatial_structures_compare(self, names, result_dirs, img_dir):
        ref_name, sim_name = names
        ref_result_dir, sim_result_dir = result_dirs

        # Reference
        files = sorted(list(ref_result_dir.glob('u*.nc')))
        ds_ref = xr.open_mfdataset(files, engine='netcdf4')

        u_ref = ds_ref['u'].values

        # Simulation
        files = sorted(list(sim_result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        u     = ds['u']
        u_obs = ds['u_obs']

        ny, nx = u_ref.shape
        aspect = nx / ny

        fig, axes = plt.subplots(2, 3, figsize=(30, 20))
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

        im = axes[0,0].imshow(u_ref, origin='lower', cmap='viridis', aspect=aspect)
        im = axes[0,1].imshow(u_obs, origin='lower', cmap='viridis', aspect=aspect)
        im = axes[0,2].imshow(u,     origin='lower', cmap='viridis', aspect=aspect)

        im2 = axes[1,2].imshow(np.abs(u-u_ref), origin='lower', cmap='jet', aspect=aspect)

        axes[0,0].set_title('Reference')
        axes[0,1].set_title('Observation')
        axes[0,2].set_title(f'{sim_name}')
        axes[1,2].set_title(f'Error ({sim_name})')

        for ax in axes.ravel():
            ax.set_xlabel(r'$x$', **self.axis_font)
            ax.set_ylabel(r'$t$', **self.axis_font)

        cbar  = fig.colorbar(im,  ax=axes[0])
        cbar2 = fig.colorbar(im2, ax=axes[1])

        filename = f'{sim_name}_compare.png'
        fig.savefig(img_dir / filename, bbox_inches='tight')
        plt.close('all')

    def _visualize_rmse_series(self, result_dirs, img_dir):
        ref_result_dir, sim_result_dir = result_dirs

        # Reference
        files = sorted(list(ref_result_dir.glob('u*.nc')))
        ds_ref = xr.open_mfdataset(files, engine='netcdf4')

        u_ref = ds_ref['u'].values

        # Simulation
        files = sorted(list(sim_result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        u     = ds['u'].values
        u_obs = ds['u_obs'].values

        time = ds['time']
        rmse = lambda var: np.sqrt( np.mean( ( var - u_ref )**2, axis=1 ) )

        fig, ax = plt.subplots(figsize=(12,12))
        ax.plot(time, rmse(u), '-r', lw=3, label=f'{self.model_name}')
        ax.plot(time, rmse(u_obs), '-k', lw=3, label='observation')

        ax.set_xlabel('Time', **self.axis_font)
        ax.set_ylabel('RMSE', **self.axis_font)
        ax.set_ylim(ymin=0, ymax=5)
        ax.legend(prop={'size': self.fontsize})
        ax.grid(ls='dashed', lw=1)
        fig.savefig(img_dir / f'rmse_series_{self.model_name}.png', bbox_inches='tight')
        plt.close('all')
