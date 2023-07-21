import torch
import pathlib
import abc
import json
import os
import copy
import numpy as np
import xarray as xr
from collections import defaultdict
from torch.utils.data import DataLoader
from .utils import cycle, Timer
from simulation.utils import save_meta, save_json, split_dir_and_file
from .lorentz96_dataset import Lorenz96Dataset
from .img_saver import ImageSaver, save_loss

class _BaseTrainer(abc.ABC):
    """
    Base class for training
    """

    def __init__(self, *args, **kwargs):
        self.losses = defaultdict(list)
        self.epoch_dict = defaultdict(list)
        self.elapsed_times = defaultdict(list)
        self.memory_consumption = {}

        self.device = kwargs.get('device', 'cuda')
        if self.device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dirname = kwargs.get('dirname')
        filename = kwargs.get('filename')

        # Update dirname and filename
        dirname, filename = split_dir_and_file(full_path=filename, default_dirname=dirname)

        self.out_dir = kwargs.get('out_dir', './')
        self.inference_mode = kwargs.get('inference_mode', False)
        self.mode_name = 'inference' if self.inference_mode else 'train'
        self.denoising_diffusion = False
        json_file = pathlib.Path(dirname) / filename
        if not json_file:
            raise FileNotFoundError(f'input file {filename} not found at {dirname}')

        with open(json_file, 'r') as f:
            self.json_data = json.load(f)

        # Update parameters for hybrid runs
        if self.inference_mode:
            maybe_found_kwargs = {
                'batch_size': 1,
                'sampling_timesteps': None,
                'ddim_sampling_eta': 0.,
                'use_ddib': False,
                'save_intermediate_sampling_imgs': False,
                'intermediate_sample_dir': None,
            }

            for key, default in maybe_found_kwargs.items():
                self.json_data['nn_model'][key] = kwargs.get(key, default)

        # Some default values
        self.default_values = {
            'lr': 0.0001,
            'beta_1': 0.9,
            'beta_2': 0.99,
            'loss_type': 'l1',
            'batch_size': 16,
            'checkpoint_idx': -1,
            'train_num_steps': 100,
            'n_freq_checkpoint': 10,
            'n_freq_eval': 10,
            'n_freq_sample_checkpoint': 1000,
            'run_number': 0,
            'opt_type': 'Adam',
            'device': self.device,
            'obs_interval': 1,
            'sigma': 1.0,
            'sampling_timesteps': None,
            'ddim_sampling_eta': 0.,
            'use_ddib': False,
            'preprocess_type': 'normalization',
            'obs_noise_runtime': False,
            'save_intermediate_sampling_imgs': False,
            'intermediate_sample_dir': None,
        }

        self._add_dict_as_attributes(self.json_data, self.default_values)

        self.timer = Timer(device=self.device)
        self.initial_step = 0
        self.seq_len = 1

    @abc.abstractmethod
    def _initialize(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _train(self, batch, step):
        raise NotImplementedError()

    @abc.abstractmethod
    def _test(self, batch, step, mode):
        raise NotImplementedError()

    @abc.abstractmethod
    def sampling(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _save_model(self, *args, **kwargs):
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        # Common initialization for all models
        if self.inference_mode:
            self.__initialize_inference(*args, **kwargs)
        else:
            self.__initialize_train(*args, **kwargs)

        # Model specific initialization
        self._initialize(*args, **kwargs)

    def __initialize_train(self, *args, **kwargs):
        # Prepare directories
        self.result_dir = self._prepare_dirs()

        # Define dataloaders
        self.train_loader, self.val_loader, self.test_loader = self._get_dataloaders()

        # Saving only images
        saver_dict = {'out_dir': self.fig_dir,
                      'coords': self.coords,}
        self.saver = ImageSaver(**saver_dict)

    def __initialize_inference(self, *args, **kwargs):
        # Prepare directories
        self.result_dir = self._prepare_dirs()

        # Define dataloaders
        self.train_loader, self.val_loader, self.test_loader = self._get_dataloaders()

    def run(self, *args, **kwargs):
        self.timer.start()
        for step in range(self.initial_step, self.train_num_steps+1):
            train_batch = next(self.train_loader)

            # Training
            with torch.enable_grad():
                self._train(batch=train_batch, step=step)

            # Validation and test
            if step % self.n_freq_eval == 0:
                val_batch = next(self.val_loader)
                test_batch = next(self.test_loader)
                #with torch.no_grad():
                self._test(batch=val_batch, step=step, mode='val')
                self._test(batch=val_batch, step=step, mode='test')

            if step % self.n_freq_sample_checkpoint == 0 or step == self.train_num_steps:
                self._check_point(step=step)

    def finalize(self, *args, **kwargs):
        seconds = kwargs.get('seconds')

        if self.inference_mode:
            message = f'It took {seconds} [s] to infer'
        else:
            save_loss(loss_data_dir = self.checkpoint_dir,
                      img_dir = self.fig_dir,
                      run_number = self.run_number, 
                      vmin = 0,
                      vmax = 0.1)
            message = f'It took {seconds} [s] to train {self.model_name} model for {self.train_num_steps} steps'

        print(message)
        log_filename = pathlib.Path(self.result_dir) / f'log_{self.mode_name}_{self.run_number:03}.txt'
        with open(log_filename, 'w') as f:
            print(message, file=f)
            checkpoint_found = self._find_checkpoint(self.checkpoint_idx)
            if checkpoint_found:
                ds = xr.open_dataset(self.checkpoint, engine='netcdf4')
                print(ds, file=f)

    def _check_point(self, step):
        self.timer.stop()
        elapsed_seconds = self.timer.elapsed_seconds
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint*.nc'))

        idx = len(checkpoint_files)
        # Set file names
        next_checkpoint_file_name = self.checkpoint_dir / f'checkpoint{idx:03}.nc'
        current_state_file_name = self.checkpoint_dir / f'model_checkpoint{idx:03}.pt'

        attrs = {}
        if idx > 0:
            previous_checkpoint_file_name  = self.checkpoint_dir / f'checkpoint{idx-1:03}.nc'
            if not previous_checkpoint_file_name.is_file():
                raise FileNotFoundError(f'{prev_result_filename} does not exist')

            ds = xr.open_dataset(previous_checkpoint_file_name, engine='netcdf4')
            attrs = copy.deepcopy(ds.attrs)
            final_step = attrs['final_step']
            attrs['last_state_file'] = str(current_state_file_name)
            attrs['initial_step'] = final_step + 1
            attrs['final_step'] = step
            attrs['elapsed_time'] = elapsed_seconds
            attrs['run_number'] = self.run_number
        else:
            # Then first checkpoint
            attrs = copy.deepcopy(self.default_values)
            attrs = {**attrs, **self.norm_dict}
            def add_recursive(dict_like):
                for key, value in dict_like.items():
                    if type(value) is dict:
                        add_recursive(value)
                    else:
                        attrs[key] = value

                        if value is None:
                            value = 0
                        if type(value) is bool:
                            value = int(value)
                        attrs[key] = value

            def convert_to_netcdf_type(dict_like):
                for key, value in dict_like.items():
                    if value is None:
                        value = 0
                    if type(value) is bool:
                        value = int(value)
                    dict_like[key] = value

                return dict_like

            add_recursive(self.json_data)
            attrs = convert_to_netcdf_type(attrs)
            attrs['initial_step'] = 0
            attrs['final_step'] = step
            attrs['last_state_file'] = str(current_state_file_name)
            attrs['elapsed_time'] = elapsed_seconds
            attrs['model_name'] = self.model_name

        data_vars = {}
        data_vars['train_losses'] = (['steps'], self.losses['train'])

        n_epochs = attrs['final_step'] - attrs['initial_step'] + 1
        coords = {}
        coords['steps'] = np.arange(attrs['initial_step'], attrs['final_step']+1)

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Save meta data
        ds.to_netcdf(next_checkpoint_file_name, engine='netcdf4')

        # Save model
        self._save_model(state_file_name=current_state_file_name)

        # Initialize loss dict after saving
        self.losses = defaultdict(list)

        # Start timer again
        self.timer.start()

    def _find_checkpoint(self, checkpoint_idx=-1):
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint*.nc'))

        if checkpoint_idx==-1:
            if not checkpoint_files:
                self.checkpoint = None
                return False
        else:
            # Then inference mode with specified checkpoint file
            if not (checkpoint_idx < len(checkpoint_files)):
                raise ValueError(f'specified checkpoint idx {checkpoint_idx} is out of range')

        checkpoint_files = sorted(checkpoint_files)
        self.checkpoint = checkpoint_files[checkpoint_idx]
        return True

    def _add_dict_as_attributes(self, dict_data, defalut_values): 
        """
        Add all elements in a dict_data as attributes
        Default values are overwritten by dict_data
        """

        for key, value in defalut_values.items():
            setattr(self, key, value)

        def add_recursive(dict_like):
            for key, value in dict_like.items():
                if type(value) is dict:
                    add_recursive(value)
                else:
                    setattr(self, key, value)

        add_recursive(dict_data)

    def _get_dataloaders(self):
        modes = ['train', 'val', 'test']
        train_dir, val_dir, test_dir = [pathlib.Path(self.in_dir) / mode for mode in modes]

        dataset_dict = {
            'inference_mode': self.inference_mode,
            'obs_interval': self.obs_interval,
            'obs_noise_runtime': self.obs_noise_runtime,
            'sigma': self.sigma,
        }

        train_dataset = Lorenz96Dataset(path=train_dir, **dataset_dict)
        val_dataset   = Lorenz96Dataset(path=val_dir, **dataset_dict)
        test_dataset  = Lorenz96Dataset(path=test_dir, **dataset_dict)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False)

        self.default_values['nb_train'] = len(train_loader.dataset)
        self.default_values['nb_val']   = len(val_loader.dataset)
        self.default_values['nb_test']  = len(test_loader.dataset)

        train_loader = cycle(train_loader)
        val_loader   = cycle(val_loader)
        test_loader  = cycle(test_loader)

        self.norm_dict = copy.deepcopy(train_dataset.norm_dict)

        self.coords = train_dataset.coords
        if 'time' in self.coords:
            self.seq_len = len(self.coords['time'])

        return train_loader, val_loader, test_loader

    def _prepare_dirs(self):
        base_out_dir = self.out_dir
        self.in_dir = pathlib.Path(self.in_dir)
        case_name = self.case_name

        # Create <out_dir>/<case_name> and <out_dir>/<case_name>/checkpoint
        out_dir = pathlib.Path(base_out_dir) / case_name
        result_dir = out_dir
        print(f'--- Start preparing directories at {out_dir} ---')
        if not result_dir.exists():
            result_dir.mkdir(parents=True)

        self.checkpoint_dir = result_dir / 'checkpoint'

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

        if self.inference_mode:
            self.inference_dir = result_dir / 'inference'
            if not self.inference_dir.exists():
                self.inference_dir.mkdir(parents=True)

        else:
            self.fig_dir = result_dir / 'imgs'
            if not self.fig_dir.exists():
                self.fig_dir.mkdir(parents=True)

        print(f'--- directories are ready at {out_dir} ---')
        symdir = case_name

        if not os.path.exists(symdir):
            os.symlink(out_dir, symdir)
            print(f'--- symbolic_link to {out_dir}: {symdir} ---')

        # Save meta data
        save_meta(dirname=result_dir, filename='meta.txt')

        # Save json file
        save_json(dirname=result_dir, json_data=self.json_data, filename='settings.json')

        return result_dir
