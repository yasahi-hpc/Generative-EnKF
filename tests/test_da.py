import pytest
import numpy as np
from simulation.enkf import get_kalman_filter
from simulation.utils import observe

@pytest.mark.parametrize('obs_interval', [1, 2, 4])
def test_observation(obs_interval):
    nx = 40
    u = np.random.rand(nx)
    
    u_obs = observe(data=u, obs_interval=obs_interval)

    assert u_obs.shape == (nx//obs_interval,)

@pytest.mark.parametrize('filter_type', ['enkf', 'letkf', 'none_kf'])
@pytest.mark.parametrize('obs_interval', [1, 2, 4])
def test_kalman_filter(filter_type, obs_interval, mock_simulator, tmp_dir):
    kf_dict = {
               'n_ens': 32,
               'n_stt': 40,
               'obs_interval': obs_interval,
               'in_dir': str(tmp_dir),
               'result_dir': str(tmp_dir),
               'n_local': 6,
               'beta': 1.0,
               'sigma': 10.,
               'nb_steps': 10,
              }

    kalman_filter = get_kalman_filter(filter_type)(**kf_dict)

    # Test with a single ensemble
    ensembles = [mock_simulator]

    # mock observation
    nx = 40
    u_obs = np.random.rand(nx)

    # This may be a bad design
    kalman_filter.apply(it=0, ensembles=ensembles, observation=u_obs)
