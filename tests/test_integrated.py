import pytest
import numpy as np
import copy
import pathlib
import shutil
from simulation.solvers import get_solver
from simulation.utils import save_json
from postscripts.postscripts import get_postscripts

class TestNature:
    model_name: str = 'DNS'
    json_file_name: str = 'perturbed.json'
    needed_files: list = ['spatio_temporal_Lorenz96-DNS.png']

    @pytest.fixture(autouse=True)
    def _mock_input(self, mock_input_perturbed, tmp_dir) -> None:
        save_json(dirname=tmp_dir, json_data=mock_input_perturbed, filename=self.json_file_name)

    @pytest.fixture(autouse=True)
    def _solver(self, tmp_dir) -> None:
        solver_dict = {'model_name': self.model_name,
                       'dirname': tmp_dir,
                       'filename': self.json_file_name,
                      }
        self.solver = get_solver(self.model_name)(**solver_dict)

    @pytest.fixture(autouse=True)
    def _postscripts(self, mock_input_perturbed, tmp_dir) -> None:
        postscript_dict = {
            'model_name': self.model_name,
            'dirname': tmp_dir,
            'filename': self.json_file_name,
        }
        self.post_script = get_postscripts(self.model_name)(**postscript_dict)

    def test_run(self, seconds=5) -> None:
        self.solver.initialize()
        self.solver.run()
        self.solver.finalize(seconds=seconds)

    def test_post(self, seconds=3) -> None:
        self.post_script.initialize()
        self.post_script.run()
        self.post_script.finalize(seconds=seconds)

        img_dir_name = str(self.post_script.img_dir) + '/'
        plotted_files = [str(file).replace(img_dir_name, '') for file in self.post_script.img_dir.glob('*.png')]

        assert sorted(plotted_files) == sorted(self.needed_files)

@pytest.mark.parametrize('da_steps, nb_runs',
    [(1, 1),
     (10, 1),
     (1, 2),
    ])
class TestNudging:
    """
    This test should be executed after TestNature
    """
    model_name: str = 'Nudging'
    json_file_name: str = 'nudging.json'
    needed_files: list = ['spatio_temporal_Lorenz96-Nudging.png',
                          'Lorenz96-Nudging_compare.png',
                          'rmse_series_Lorenz96-Nudging.png']

    @pytest.fixture(autouse=True)
    def _mock_input(self, mock_input_perturbed, tmp_dir, da_steps, nb_runs) -> None:
        mock_input_nudging = copy.deepcopy(mock_input_perturbed)

        settings = mock_input_nudging['settings']
        settings['case_name'] = self.model_name
        settings['in_case_name'] = 'Perturbed'
        settings['nb_runs'] = nb_runs

        simulation = mock_input_nudging['simulation']
        simulation['alpha'] = 0.5
        simulation['noise_level'] = 1
        simulation['obs_interval'] = 1
        simulation['da_steps'] = da_steps

        mock_input_nudging['settings'] = settings
        mock_input_nudging['simulation'] = simulation
        self.json_data = mock_input_nudging

        save_json(dirname=tmp_dir, json_data=mock_input_nudging, filename=self.json_file_name)

    @pytest.fixture(autouse=True)
    def _solver(self, tmp_dir) -> None:

        solver_dict = {'model_name': self.model_name,
                       'dirname': tmp_dir,
                       'filename': self.json_file_name,
                      }
        self.solver = get_solver(self.model_name)(**solver_dict)

    @pytest.fixture(autouse=True)
    def _postscripts(self, mock_input_perturbed, tmp_dir) -> None:
        postscript_dict = {
            'model_name': self.model_name,
            'dirname': tmp_dir,
            'filename': self.json_file_name,
        }
        self.post_script = get_postscripts(self.model_name)(**postscript_dict)

    def teardown_method(self, method):
        out_dir = self.json_data['settings']['out_dir']
        case_name = self.json_data['settings']['case_name']
        sym = pathlib.Path(case_name)
        path = sym.resolve() / 'results'
        pwd = pathlib.Path.cwd()

        # First remove the symbolic link
        sym.unlink()

        # Remove the directory and contents if current directory is not under path
        if str(path) in str(pwd):
            print(f'current directory {pwd} is under {path}. Do not remove results.')
        else:
            shutil.rmtree(path)

    def test_run(self, seconds=5, da_steps=None, nb_runs=None) -> None:
        self.solver.initialize()
        self.solver.run()
        self.solver.finalize(seconds=seconds)

        self.post_script.initialize()
        self.post_script.run()
        self.post_script.finalize(seconds=seconds)

        img_dir_name = str(self.post_script.img_dir) + '/'
        plotted_files = [str(file).replace(img_dir_name, '') for file in self.post_script.img_dir.glob('*.png')]

        assert sorted(plotted_files) == sorted(self.needed_files)

@pytest.mark.parametrize('kalman_filter, da_steps',
    [('none_kf', 1),
     ('none_kf', 10),
     ('letkf', 1),
    ])
class TestLETKF:
    """
    This test should be executed after TestNature
    """
    model_name: str = 'LETKF'
    json_file_name: str = 'letkf.json'
    needed_files: list = ['Lorenz96-EnKF_compare.png',
                          'Lorenz96-EnKF_ensembles.png',
                          'rmse_series_Lorenz96-EnKF.png']

    @pytest.fixture(autouse=True)
    def _mock_input(self, mock_input_perturbed, tmp_dir, kalman_filter, da_steps) -> None:
        mock_input_letkf = copy.deepcopy(mock_input_perturbed)

        settings = mock_input_letkf['settings']
        settings['case_name'] = self.model_name
        settings['in_case_name'] = 'Perturbed'

        simulation = mock_input_letkf['simulation']
        simulation['beta'] = 1.0
        simulation['sigma'] = 1.0
        simulation['n_ens'] = 4
        simulation['obs_interval'] = 1
        simulation['kalman_filter'] = kalman_filter
        simulation['da_steps'] = da_steps

        mock_input_letkf['settings'] = settings
        mock_input_letkf['simulation'] = simulation

        self.json_data = mock_input_letkf

        save_json(dirname=tmp_dir, json_data=mock_input_letkf, filename=self.json_file_name)

    @pytest.fixture(autouse=True)
    def _solver(self, tmp_dir) -> None:
        solver_dict = {'model_name': self.model_name,
                       'dirname': tmp_dir,
                       'filename': self.json_file_name,
                      }
        self.solver = get_solver(self.model_name)(**solver_dict)

    @pytest.fixture(autouse=True)
    def _postscripts(self, mock_input_perturbed, tmp_dir) -> None:
        postscript_dict = {
            'model_name': self.model_name,
            'dirname': tmp_dir,
            'filename': self.json_file_name,
        }
        self.post_script = get_postscripts(self.model_name)(**postscript_dict)

    def teardown_method(self, method):
        out_dir = self.json_data['settings']['out_dir']
        case_name = self.json_data['settings']['case_name']
        sym = pathlib.Path(case_name)
        path = sym.resolve() / 'results'
        pwd = pathlib.Path.cwd()

        # First remove the symbolic link
        sym.unlink()

        # Remove the directory and contents if current directory is not under path
        if str(path) in str(pwd):
            print(f'current directory {pwd} is under {path}. Do not remove results.')
        else:
            shutil.rmtree(path)

    def test_run(self, seconds=5, kalman_filter=None, da_steps=None) -> None:
        self.solver.initialize()
        self.solver.run()
        self.solver.finalize(seconds=seconds)

        self.post_script.initialize()
        self.post_script.run()
        self.post_script.finalize(seconds=seconds)

        img_dir_name = str(self.post_script.img_dir) + '/'
        plotted_files = [str(file).replace(img_dir_name, '') for file in self.post_script.img_dir.glob('*.png')]

        assert sorted(plotted_files) == sorted(self.needed_files)

@pytest.mark.parametrize('kalman_filter, ddim_sampling_eta, da_steps',
    [('none_kf', 0., 1),
     ('none_kf', 0., 10),
     ('none_kf', 0.1, 1),
    ])
class TestEFDA:
    """
    This test should be executed after TestNature
    """
    model_name: str = 'EFDA'
    json_file_name: str = 'efda.json'
    nb_sims: int = 3

    @pytest.fixture(autouse=True)
    def _plotted_files(self) -> None:
        self.needed_files = [f'Lorenz96-EFDA_ensembles.png']
        for idx in range(self.nb_sims):
            self.needed_files.append(f'Lorenz96-EFDA_simulator{idx}_compare.png')
            self.needed_files.append(f'rmse_series_Lorenz96-EFDA_simulator{idx}.png')

    @pytest.fixture(autouse=True)
    def _mock_input(self, mock_input_efda, tmp_dir, kalman_filter, ddim_sampling_eta, da_steps) -> None:
        nn_settings = mock_input_efda['nn_settings']
        nn_settings['kalman_filter'] = kalman_filter
        nn_settings['ddim_sampling_eta'] = ddim_sampling_eta
        nn_settings['use_ddib'] = ddim_sampling_eta > 0
        nn_settings['da_steps'] = da_steps

        mock_input_efda['nn_settings'] = nn_settings

        self.json_data = mock_input_efda
        save_json(dirname=tmp_dir, json_data=mock_input_efda, filename=self.json_file_name)

    @pytest.fixture(autouse=True)
    def _solver(self, tmp_dir) -> None:
        solver_dict = {'model_name': self.model_name,
                       'dirname': tmp_dir,
                       'filename': self.json_file_name,
                      }
        self.solver = get_solver(self.model_name)(**solver_dict)

    @pytest.fixture(autouse=True)
    def _postscripts(self, mock_input_perturbed, tmp_dir) -> None:
        postscript_dict = {
            'model_name': self.model_name,
            'dirname': tmp_dir,
            'filename': self.json_file_name,
        }
        self.post_script = get_postscripts(self.model_name)(**postscript_dict)

    def teardown_method(self, method):
        out_dir = self.json_data['settings']['out_dir']
        case_name = self.json_data['settings']['case_name']
        sym = pathlib.Path(case_name)
        path = sym.resolve() / 'results'
        pwd = pathlib.Path.cwd()

        # First remove the symbolic link
        sym.unlink()

        # Remove the directory and contents if current directory is not under path
        if str(path) in str(pwd):
            print(f'current directory {pwd} is under {path}. Do not remove results.')
        else:
            shutil.rmtree(path)

    def test_run(self, seconds=5, kalman_filter=None, ddim_sampling_eta=None, da_steps=None) -> None:
        self.solver.initialize()
        self.solver.run()
        self.solver.finalize(seconds=seconds)

        self.post_script.initialize()
        self.post_script.run()
        self.post_script.finalize(seconds=seconds)

        img_dir_name = str(self.post_script.img_dir) + '/'
        plotted_files = [str(file).replace(img_dir_name, '') for file in self.post_script.img_dir.glob('*.png')]

        assert sorted(plotted_files) == sorted(self.needed_files)
