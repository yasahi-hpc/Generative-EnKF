import pathlib
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style
from numpy import ma
from ._base_postscripts import _BasePostscripts
from simulation.utils import observe_with_zeros

class EFDA_Postscripts(_BasePostscripts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'Lorenz96-EFDA'
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

        simulation_dirs = sorted(self._result_dir.glob('simulator*'))
        self.simulators = [str(simulation_dir).split('/')[-1] for simulation_dir in simulation_dirs]

    def run(self, *args, **kwargs):
        self._visualize_ensembles(name=self.model_name, result_dir=self._result_dir, img_dir=self._img_dir)

        if self.plot_series:
            self._visualize_da(name=self.model_name, result_dir=self._result_dir, img_dir=self._series_img_dir, simulators=self.simulators)

        for simulator in self.simulators:
            self._visualize_spatial_structures_compare(name=self.model_name, result_dir=self._result_dir, ref_dir=self.in_dir, img_dir=self._img_dir, simulator=simulator)
            self._visualize_rmse_series(result_dir=self._result_dir, ref_dir=self.in_dir, img_dir=self._img_dir, simulator=simulator)

    def finalize(self, *args, **kwargs):
        pass

    def _visualize_spatial_structures_compare(self, name, result_dir, ref_dir, img_dir, simulator):
        filename = img_dir / f'{name}_{simulator}_compare.png'
        if filename.exists():
            return

        # Reference
        files = sorted(list(ref_dir.glob('u*.nc')))
        ds_ref = xr.open_mfdataset(files, engine='netcdf4')
        u_ref = ds_ref['u'].values

        # Observation
        files = sorted(list(result_dir.glob('enkf*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        obs_interval = ds.attrs.get('obs_interval', 1)
        da_steps = ds.attrs.get('da_steps', 1)
        u_obs = ds['observation'].values

        nt = u_ref.shape[0]
        nt_obs = u_obs.shape[0]

        assert nt == nt_obs * da_steps, f"nt in ref and nt in obs should satisfy nt_obs = nt / da_step: nt {nt}, nt_obs {nt_obs}, da_steps {da_steps}"

        # Simulation
        _sim_result_dir = result_dir / simulator
        files = sorted(list(_sim_result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        u = ds['u'].values

        u_obs_reshaped = np.zeros_like(u_ref)
        u_obs_reshaped[::da_steps] = u_obs

        u_mask = np.ones_like(u_ref) * -1
        u_mask[::da_steps, ::obs_interval] = 1.

        ny, nx = u_ref.shape
        aspect = nx / ny

        fig, axes = plt.subplots(2, 3, figsize=(30, 20))
        axes[1, 0].set_visible(False)

        im = axes[0,0].imshow(u_ref,  origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')
        im = axes[0,1].imshow(ma.masked_where(u_mask <=0, u_obs_reshaped),  origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')
        im = axes[0,2].imshow(u, origin='lower', cmap='viridis', aspect=aspect, vmin=-self.umax, vmax=self.umax, interpolation='none')

        im2 = axes[1,1].imshow(ma.masked_where(u_mask <=0, np.abs(u_obs_reshaped-u_ref)), origin='lower', cmap='jet', aspect=aspect, vmin=0, vmax=self.error_max, interpolation='none')
        im2 = axes[1,2].imshow(np.abs(u-u_ref), origin='lower', cmap='jet', aspect=aspect, vmin=0, vmax=self.error_max, interpolation='none')

        axes[0,0].set_title('Reference', **self.title_font)
        axes[0,1].set_title('Observation', **self.title_font)
        axes[0,2].set_title(f'{name}', **self.title_font)
        axes[1,1].set_title(f'Error (Observation)', **self.title_font)
        axes[1,2].set_title(f'Error ({name})', **self.title_font)

        for ax in axes.ravel():
            ax.set_xlabel(r'$x$', **self.axis_font)
            ax.set_ylabel(r'$t$', **self.axis_font)

        cbar  = fig.colorbar(im,  ax=axes[0])
        cbar2 = fig.colorbar(im2, ax=axes[1])

        fig.savefig(filename, bbox_inches='tight')
        plt.close('all')

    def _visualize_ensembles(self, name, result_dir, img_dir, n_cols=8):
        filename = img_dir / f'{name}_ensembles.png'
        if filename.exists():
            return

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

        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.savefig(filename, bbox_inches='tight')
        plt.close('all')

    def _visualize_da(self, name, result_dir, img_dir, simulators):
        """
        Ensemble data before DA are stored in ens_idx*/u{it}.nc
        observation and Ensemble mean are stored in enkf_stats{it}.nc
        """
        # Lazy import
        from multiprocessing import Pool

        # Reference
        files = sorted(list(result_dir.glob('enkf*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        obs_interval = ds.attrs['obs_interval']
        nbiter = len(ds['time'])

        x = ds['x'].values
        x_obs = ds['x'].values[::obs_interval]

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

        for simulator in simulators:
            _sim_result_dir = result_dir / simulator
            files = sorted(list(_sim_result_dir.glob('u*.nc')))
            ds_sim = xr.open_mfdataset(files, engine='netcdf4')

            def plot_task(it):
                filename  = img_dir / f'{name}_{simulator}_da_ens_it{it:05}.png'
                filename2 = img_dir / f'{name}_{simulator}_da_it{it:05}.png'

                if filename.exists():
                    return
                    
                ds_it = ds.isel(time=it).compute()
                ds_ens_it = ds_ens.isel(time=it).compute()
                ds_sim_it = ds_sim.isel(time=it).compute()

                mean = ds_it['ensemble_mean'].values
                spread = ds_it['ensemble_spread'].values
                obs = ds_it['observation'].values[::obs_interval]
                ref = ds_it['reference'].values
                u = ds_sim_it['u'].values

                fig, ax = plt.subplots(figsize=(8, 8))
                fig2, ax2 = plt.subplots(figsize=(8, 8))

                ax.plot(x, mean, 'r-', lw=3, label='mean')
                ax.plot(x_obs, obs, 'g*', markersize=18, label='obs')
                ax.plot(x, ref, 'b-.', lw=3, label='ref')
                ax.plot(x, u, 'm-.', lw=3, label='pred')

                # Ensembles
                for idx in range(nb_ensembles):
                    u_tmp = ds_ens_it['u'].isel(ensemble=idx).values
                    ax.plot(x, u_tmp, lw=1, alpha=0.3)

                ax.legend(loc='upper left', ncols=2, prop={'size': self.fontsize*0.7})
                ax.grid(which='both', ls='-.')
                ax.axhline(y=0, color='k')
                ax.set_xlabel(r'$x$', **self.axis_font)
                ax.set_ylabel(r'$u$', **self.axis_font)
                ax.set_title(r'$t = {:03}$'.format(it), **self.title_font)
                ax.set_ylim([-self.umax, self.umax])

                fig.savefig(filename, bbox_inches='tight')
                plt.close('all')

                # Averaged values 
                ax2.plot(x, mean, 'r-', lw=3, label='mean')
                ax2.fill_between(x, mean-spread, mean+spread, color='r', alpha=0.2)
                ax2.plot(x_obs, obs, 'g*', markersize=18, label='obs')
                ax2.plot(x, ref, 'b-.', lw=3, label='ref')
                ax2.plot(x, u, 'm-.', lw=3, label='pred')

                ax2.legend(loc='upper left', ncols=2, prop={'size': self.fontsize*0.7})
                ax2.grid(which='both', ls='-.')
                ax2.axhline(y=0, color='k')
                ax2.set_xlabel(r'$x$', **self.axis_font)
                ax2.set_ylabel(r'$u$', **self.axis_font)
                ax2.set_title(r'$t = {:03}$'.format(it), **self.title_font)
                ax2.set_ylim([-self.umax, self.umax])

                fig2.savefig(filename2, bbox_inches='tight')
                plt.close('all')

            for it in range(nbiter):
                plot_task(it)

            #p = Pool(36)
            #for it in range(nbiter):
            #    p.apply_async(plot_task, args=(it,))

            #p.close()
            #p.join()
            #print('All subprocesses done.')

    def _visualize_rmse_series(self, result_dir, ref_dir, img_dir, simulator, start=200):
        filename = img_dir / f'rmse_series_{self.model_name}_{simulator}.png'
        if filename.exists():
            return

        # Reference
        files = sorted(list(ref_dir.glob('u*.nc')))
        ds_ref = xr.open_mfdataset(files, engine='netcdf4')
        ds_ref = ds_ref.isel(time=slice(start, None))
        u_ref = ds_ref['u'].values
        u_obs = observe_with_zeros(u_ref)

        # Simulation
        _sim_result_dir = result_dir / simulator
        files = sorted(list(_sim_result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        ds = ds.isel(time=slice(start, None))
        u = ds['u'].values
        time = ds['time']

        rmse = lambda var, obs_interval: np.sqrt( np.mean( ( var[:, ::obs_interval] - u_ref[:, ::obs_interval] )**2, axis=1 ) )

        fig, ax = plt.subplots(figsize=(12,12))
        ax.plot(time, rmse(u, 1), '-r', lw=3, label=f'{self.model_name}')
        ax.plot(time, rmse(u_obs, 1), '-k', lw=3, label='observation')

        ax.set_xlabel('Time', **self.axis_font)
        ax.set_ylabel('RMSE', **self.axis_font)
        ax.set_ylim(ymin=0, ymax=5)
        ax.legend(prop={'size': self.fontsize})
        ax.grid(ls='dashed', lw=1)
        fig.savefig(filename, bbox_inches='tight')
        plt.close('all')
