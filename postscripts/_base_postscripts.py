import abc
import pathlib
import json
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
from scipy import signal
from matplotlib.colors import LogNorm
from simulation.utils import split_dir_and_file

class _BasePostscripts(abc.ABC):
    """
    Base class for postscript
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._result_dir = None
        self._img_dir = None

        self.in_dir = None
        self.model_name = kwargs.get('model_name')
        self.plot_series = kwargs.get('plot_series')
        dirname = kwargs.get('dirname')
        filename = kwargs.get('filename')

        # Update dirname and filename (filename can be given as fullpath)
        dirname, filename = split_dir_and_file(full_path=filename, default_dirname=dirname)
        json_file = pathlib.Path(dirname) / filename

        if not json_file:
            raise FileNotFoundError(f'input file {filename} not found at {dirname}')

        with open(json_file, 'r') as f:
            self.json_data = json.load(f)

        self.settings = self.json_data['settings']
        if 'in_case_name' in self.settings:
            in_case_name = self.settings['in_case_name']
            self.json_data = self._merge_json_data(json_data=self.json_data, in_case_name=in_case_name)
            self.in_dir = pathlib.Path(self.settings['out_dir']) / in_case_name / 'results'

        # Matplotlib settings
        mpl.style.use('classic')
        fontsize = 28
        self.fontsize = fontsize
        fontname = 'Times New Roman'
        plt.rc('xtick', labelsize=fontsize)
        plt.rc('ytick', labelsize=fontsize)
        plt.rc('font', family=fontname)
                                                                        
        self.title_font = {'fontname':fontname, 'size':fontsize, 'color':'black',
                           'verticalalignment':'bottom'}
        self.axis_font = {'fontname':fontname, 'size':fontsize}

    @property
    def img_dir(self):
        return self._img_dir

    @property
    def result_dir(self):
        return self._result_dir

    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def finalize(self, *args, **kwargs):
        raise NotImplementedError()

    def _merge_json_data(self, json_data, in_case_name):
        base_out_dir = self.settings['out_dir']
        case_name = self.settings['case_name']

        prev_dir_sim = pathlib.Path(base_out_dir) / in_case_name
        prev_dir_model = pathlib.Path(base_out_dir) / case_name / in_case_name

        potentially_prev_json_files = []
        potentially_prev_json_files.append(prev_dir_sim / 'settings.json')
        potentially_prev_json_files.append(prev_dir_model / 'nn_settings.json')

        prev_json_file = None
        for potentially_prev_json_file in potentially_prev_json_files:
            if potentially_prev_json_file.exists():
                prev_json_file = potentially_prev_json_file

        if prev_json_file is None:
            raise IOError(f'Previous simulation result does not exist in {prev_dir_sim} or {prev_dir_model}')

        with open(prev_json_file, 'r') as f:
            prev_json_data = json.load(f)

        # Merge jsons (priotizing the new json)
        merged_json_data = {**prev_json_data, **json_data}

        return merged_json_data

    def _prepare_dirs(self, out_dir, case_name, result_dir_name='results', img_dir_name='imgs'):
        """
        Check the existence of required directories
        """
        out_dir = pathlib.Path(out_dir) / case_name
        result_dir = out_dir / result_dir_name
                                                 
        required_dirs = [result_dir, out_dir]
        for required_dir in required_dirs:
            if not required_dir.exists():
                raise IOError(f'required directory {required_dir} does not exist')
                                                                                                               
        img_dir = out_dir / img_dir_name
        if not img_dir.exists():
            img_dir.mkdir(parents=True)
        
        return result_dir, img_dir

    def _visualize_spatial_structures(self, name, result_dir, img_dir):
        files = sorted(list(result_dir.glob('u*.nc')))
        ds = xr.open_mfdataset(files, engine='netcdf4')

        u = ds['u'].values
        ny, nx = u.shape
        aspect = nx / ny
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(u, origin='lower', interpolation='none', cmap='viridis', aspect=aspect)
        ax.set_xlabel(r'$x$', **self.axis_font)
        ax.set_ylabel(r'$t$', **self.axis_font)

        filename = f'spatio_temporal_{name}.png'
        fig.colorbar(im, ax=ax)
        fig.savefig(img_dir / filename, bbox_inches='tight')
        plt.close('all')
