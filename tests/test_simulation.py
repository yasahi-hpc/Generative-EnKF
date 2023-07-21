import pytest
import numpy as np
from simulation.time_integral import get_time_integral_method

@pytest.mark.parametrize('time_integral_method', ['rk4th', 'rkg4th'])
def test_time_integral(time_integral_method, N=100) -> None:
    """
    harmonic oscilation
    """

    M = 1.0
    k = 1.0
    t0, t1 = 0.0, 10.0
    dt = (t1-t0) / N

    time = np.arange(t0, t1, dt)
    analytical = np.sin(time)

    # Numerical solution
    # harmonic oscilation
    def f(y):
        x, v = y
        dydt = np.zeros_like(y)

        # dx/dt
        dydt[0] = v

        # dv/dt 
        dydt[1] = -k * x / M

        return dydt

    time_integral = get_time_integral_method(time_integral_method)(h=dt)

    # Initial condition
    x0, v0 = 0.0, 1.0
    y = np.array([x0, v0])
    numerical = []

    for _ in time:
        # Keeping position only
        numerical.append(y[0])
        for step in range(time_integral.order):
            y = time_integral.advance(
                f=f,
                y=y,
                step=step)

    numerical = np.asarray(numerical)

    assert np.allclose(analytical, numerical, rtol=1.e-3, atol=1.e-3)

@pytest.mark.parametrize('mode', ['nature', 'perturbed'])
def test_lorenz96(mode, mock_simulator, nbiter=10):
    def run(model):
        for it in range(nbiter):
            model.diag()
            model.solve()

    if mode == 'perturbed':
        mock_simulator.initialize()

        # Spinup (no diag)
        run(model=mock_simulator)

        # Add perturbation and then perform simulation
        mock_simulator.initialize(mode = 'perturbed')

        run(model=mock_simulator)
    else:
        mock_simulator.initialize()
        run(model=mock_simulator)
