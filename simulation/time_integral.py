import abc
import numpy as np

class _TimeIntegral(abc.ABC):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.order = None

    @abc.abstractmethod
    def advance(self, f, y, step):
        raise NotImplementedError()

class RungeKutta4th(_TimeIntegral):
    """
    Solving dy/dt = f(t,y) by 4th-order Runge-Kutta-Method
    y^{n+1} = y^{n} + (k1 + 2*k2 + 2*k3 + k4)/6
    t^{n+1} = t^{n} + h
    where h is a time step and
    k1 = f(t^{n}      , y^{n}     ) * h
    k2 = f(t^{n} + h/2, y^{n}+k1/2) * h
    k3 = f(t^{n} + h/2, y^{n}+k2/2) * h
    k4 = f(t^{n} + h  , y^{n}+k3  ) * h

    Attributes
    ----------
    h : float
        Time step size
    y : ndarray
        1-dimensional array of starting values
    k1, k2, k3, k4 : ndarray
        1-dimensional array containing coefficietns
    y : ndarray
        1-dimensional array of starting values
    step : int
        0, 1, 2, or 3
    order : int
        4
    Methods
    -------
    advance(f, y, step)
    """

    def __init__(self, h):
        super().__init__(h)
        self.y  = None
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.k4 = None

        self.order = 4

    def advance(self, f, y, step):
        """
        Parameters
        ----------
        f : functor
            returing 1-dimensional array of f(y)
        y : ndarray
            1-dimensional array of starting values
        step : int
            0, 1, 2, or 3
        Returns
        -------
        y : ndarray
            1-dimensional array of values after h
        """

        y = np.asanyarray(y)

        if step==0:
            self.y = np.copy(y)
            self.k1 = f(y) * self.h
            y = self.y + self.k1/2
        elif step==1:
            self.k2 = f(y) * self.h
            y = self.y + self.k2/2
        elif step==2:
            self.k3 = f(y) * self.h
            y = self.y + self.k3
        elif step==3:
            self.k4 = f(y) * self.h
            y = self.y + (self.k1 + 2*self.k2 + 2*self.k3 + self.k4) / 6
        else:
            raise ValueError('step should be 0, 1, 2, or 3')

        return y

class RungeKuttaGill4th(_TimeIntegral):
    """
    Solving dy/dt = f(t,y) by 4th-order Runge-Kutta-Gill-Method
    Step 1:
    k1 = f(t^{n}, y^{n}) * h
    r1 = k1 / 2
    q1 = k1
    y1 = yn + r1

    Step 2:
    k2 = f(t^{n} + h/2, y1) * h
    r2 = (1 - sqrt(0.5)) * (k2 - q1)
    q2 = (1 - 3 * (1 - sqrt(0.5))) * q1
          + 2 * (1 - sqrt(0.5)) * k2
    y2 = y1 + r2

    Step 3:
    k3 = f(t^{n} + h/2, y2) * h
    r3 = (1 + sqrt(0.5)) * (k3 - q2)
    q3 = (1 - 3 * (1 + sqrt(0.5))) * q2
          + 2 * (1 + sqrt(0.5)) * k3
    y3 = y2 + r3

    Step 4:
    k4 = f(t^{n} + h, y3) * h
    r4 = (k4 - 2*q3) / 6
    q4 = 0
    y4 = y3 + r4

    y^{n+1} = y4
    t^{n+1} = t^{n} + h

    Attributes
    ----------
    h : float
        Time step size
    k, q : ndarray
        coefficietns k and q
    step : int
        0, 1, 2, or 3
    order : int
        4
    Returns
    -------
    y : ndarray
        1-dimensional array of values after h
    """

    def __init__(self, h):
        super().__init__(h)
        self.k = None
        self.q = None

        self.order = 4

    def advance(self, f, y, step):
        """
        Parameters
        ----------
        f : functor
            returing 1-dimensional array of f(y)
        y : ndarray
            1-dimensional array of starting values
        step : int
            0, 1, 2, or 3
        Returns
        -------
        y : ndarray
            1-dimensional array of values after h
        """

        y = np.asanyarray(y)

        if step==0:
            self.k = f(y) * self.h
            r      = self.k / 2
            self.q = np.copy(self.k)
            y = y + r

        elif step==1:
            self.k = f(y) * self.h
            r      = (1. - np.sqrt(0.5)) * (self.k - self.q)
            self.q = (1. - 3. * (1. - np.sqrt(0.5))) * self.q + 2. * (1. - np.sqrt(0.5)) * self.k
            y = y + r

        elif step==2:
            self.k = f(y) * self.h
            r      = (1. + np.sqrt(0.5)) * (self.k - self.q)
            self.q = (1. - 3. * (1. + np.sqrt(0.5))) * self.q + 2. * (1. + np.sqrt(0.5)) * self.k
            y = y + r
        elif step==3:
            self.k = f(y) * self.h
            r      = 1./6. * (self.k - 2. * self.q)
            self.q = 0.
            y = y + r
        else:
            raise ValueError('step should be 0, 1, 2, or 3')

        return y

def get_time_integral_method(method_name):
    METHODS = {
        'rk4th': RungeKutta4th,
        'rkg4th': RungeKuttaGill4th,
    }

    for n, m in METHODS.items():
        if n.lower() == method_name.lower():
            return m

    raise ValueError(f'method {method_name} is not defined')
