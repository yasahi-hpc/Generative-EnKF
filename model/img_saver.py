import pathlib
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style
from ._base_saver import _BaseSaver

def save_loss(loss_data_dir, img_dir, run_number, vmin=0, vmax=0.1):
    mpl.style.use('classic')

    fontsize = 36
    fontname = 'Times New Roman'
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('font', family=fontname)

    title_font = {'fontname': fontname, 
                  'size': fontsize,
                  'color': 'black',
                  'verticalalignment': 'bottom'
                 }

    axis_font = {'fontname': fontname, 
                 'size': fontsize,
                }

    # Load results
    result_files = sorted(list(loss_data_dir.glob('checkpoint*.nc')))
    ds = xr.open_mfdataset(result_files, concat_dim='steps', compat='no_conflicts', combine='nested')

    loss_type = ds.attrs['loss_type']
    steps = ds['steps'].values
    losses = ds['train_losses'].values

    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(steps, losses, 'r-', lw=3, label='train')

    ax.set_xlabel(r'${\rm steps}$', **axis_font)
    loss_label = r'${\rm MSE}$ ${\rm loss}$' if loss_type == 'MSE' else r'${\rm MAE}$ ${\rm loss}$'
    ax.set_ylabel(loss_label, **axis_font)
    ax.set_ylim(ymin=vmin)
    ax.legend(prop={'size': fontsize*1.3})
    ax.grid(ls='dashed', lw=1)
    fig.tight_layout()
    fig.savefig(img_dir / f'loss_{run_number}.png')
    plt.close('all')

class ImageSaver(_BaseSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_cols = kwargs.get('n_cols', 4)
        self.coords = kwargs.get('coords')
        self.save_freq = 10

        # settings for figures
        mpl.style.use('classic')

        self.fontsize = 36
        self.fontname = 'Times New Roman'
        plt.rc('xtick', labelsize=self.fontsize)
        plt.rc('ytick', labelsize=self.fontsize)
        plt.rc('font', family=self.fontname)

        self.title_font = {'fontname': self.fontname, 
                           'size': self.fontsize,
                           'color': 'black',
                           'verticalalignment': 'bottom'
                          }

        self.axis_font = {'fontname': self.fontname, 
                          'size': self.fontsize,
                         }

    def save(self, *args, **kwargs):
        u = kwargs.get('u')
        samples = kwargs.get('samples')
        u_obs = kwargs.get('u_obs')
        step = kwargs.get('step')
        mode = kwargs.get('mode')

        to_numpy = lambda var: np.squeeze(var.numpy(), axis=-2) if var.device == 'cpu' else np.squeeze(var.cpu().numpy(), axis=-2)
        u = to_numpy(u)
        samples = to_numpy(samples)
        u_obs = to_numpy(u_obs)
        
        x, x_obs = self.coords['x'].values, self.coords['x_obs'].values
        obs_interval = len(x) // len(x_obs)

        u_obs = u_obs[:, -1, ::obs_interval] # Just showing the last observation

        n_samples = len(u)
        n_cols = self.n_cols if n_samples >= self.n_cols else n_samples
        n_rows = n_samples // n_cols

        def save_images(name, data):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24,24), subplot_kw={'xticks': [], 'yticks': []},
                                     gridspec_kw=dict(hspace=0.1, wspace=0.1))

            if type(axes) is not np.ndarray:
                axes = np.asarray(axes)

            for i, ax in np.ndenumerate(axes.ravel()):
                _data, obs = data[i], u_obs[i]

                ax.plot(x, _data, 'r-', lw=3)
                ax.plot(x_obs, obs, 'b*', lw=3, markersize=18)
                ax.grid(which='both', ls='-.')
                ax.axhline(y=0, linewidth=1, color='k')

                ax.set_xlim([0, 40])
                ax.set_ylim([-25, 25])

            # Set title and filename
            title = f'step = {step:06}'
            figname = f'{mode}_{name}_{step:06}.png'
            fig.suptitle(title, **self.title_font, y = 0.9)
            fig.savefig(self.out_dir / mode / figname, bbox_inches = 'tight')
            plt.close('all')

        # Save original and generated images
        data_dict = {'u_pred': samples,
                     'u_ref': u,}

        for name, data in data_dict.items():
            save_images(name=name, data=data)
