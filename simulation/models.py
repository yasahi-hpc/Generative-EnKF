from .lorenz96 import Lorenz96
from .lorenz96_nudging import Lorenz96_Nudging
from .lorenz96_enkf import Lorenz96_EnKF
from .lorenz96_efda import Lorenz96_EFDA

def get_model(model_name):
    MODELS = {
              'DNS': Lorenz96,
              'Nudging': Lorenz96_Nudging,
              'EnKF': Lorenz96_EnKF,
              'LETKF': Lorenz96_EnKF,
              'NoDA': Lorenz96_EnKF,
              'DatasetFactory': Lorenz96_EnKF,
              'EFDA': Lorenz96_EFDA,
             }

    for n, m in MODELS.items():
        if n.lower() == model_name.lower():
            return m

    raise ValueError(f'model {model_name} is not defined')
