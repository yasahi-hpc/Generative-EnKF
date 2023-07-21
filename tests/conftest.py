import pytest
from simulation.lorenz96 import Lorenz96

@pytest.fixture(scope='session')
def tmp_dir(tmp_path_factory):
    """
    mock directory
    """
    _dir = tmp_path_factory.mktemp('tmp_dir')
    return _dir

@pytest.fixture(scope='session')
def mock_input(tmp_dir):
    """
    mock input data
    """

    data_vars = {}
    settings = {
        'out_dir': str(tmp_dir),
        'case_name': 'LETKF',
        'in_case_name': 'Perturbed'
    }

    grids = {
        'Nx': 40,
    }

    simulation = {
        'model': 'DNS',
        'F': 8,
        'diag_chunk':10,
        'diag_steps':1,
        'dt':0.05,
        'mode':'default',
        'perturbation_amp':0.00001,
        'nbiter':300,
        'u0_factor':1.001,
        'u0_idx':19,
        'n_ens':4,
        'obs_interval':1,
        'kalman_filter':'letkf',
        'n_local':6,
        'beta':1.0,
        'sigma':1.0
    }

    data_vars['settings'] = settings
    data_vars['grids'] = grids
    data_vars['simulation'] = simulation

    return data_vars

@pytest.fixture(scope='session')
def mock_input_perturbed(tmp_dir):
    """
    mock input data
    """

    data_vars = {}
    settings = {
        'out_dir': str(tmp_dir),
        'case_name': 'Perturbed'
    }

    grids = {
        'Nx': 40,
    }

    simulation = {
        'model': 'DNS',
        'F': 8,
        'diag_chunk':10,
        'diag_steps':1,
        'dt':0.05,
        'mode':'default',
        'perturbation_amp':0.00001,
        'nbiter':300,
        'u0_factor':1.001,
        'u0_idx':19,
    }

    data_vars['settings'] = settings
    data_vars['grids'] = grids
    data_vars['simulation'] = simulation

    return data_vars

@pytest.fixture(scope='session')
def mock_input_efda(tmp_dir):
    """
    mock input data for efda
    """

    data_vars = {}
    settings = {
        'out_dir': str(tmp_dir),
        'case_name': 'EFDA',
        'in_case_name': 'Perturbed'
    }

    nn_settings = {
        "nn_model_type": "debug",
        "batch_size": 1,
        "inference_mode": True,
        "model_dir": "diffusion_test",
        "da_steps": 1,
        "kalman_filter":"none_kf",
        "n_ens": 4,
        "F": 10,
        "sampling_timesteps": 5,
    }

    data_vars['settings'] = settings
    data_vars['nn_settings'] = nn_settings

    return data_vars

@pytest.fixture(scope='session')
def mock_simulator(mock_input, tmp_dir):
    """
    mock simulator (vanilla Lorenz96)
    """
    model = Lorenz96(json_data=mock_input, result_dir=tmp_dir, suppress_diag=True)
    return model
