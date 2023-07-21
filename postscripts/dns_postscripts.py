import pathlib
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
from ._base_postscripts import _BasePostscripts

class DNS_Postscripts(_BasePostscripts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'Lorenz96-DNS'
        self.img_sub_dirs = ['contour'] 

    def initialize(self, *args, **kwargs):
        self._result_dir, self._img_dir = super()._prepare_dirs(out_dir = self.settings['out_dir'],
                                                                case_name = self.settings['case_name'])

    def run(self, *args, **kwargs):
        super()._visualize_spatial_structures(name=self.model_name, result_dir=self._result_dir, img_dir=self._img_dir)

    def finalize(self, *args, **kwargs):
        pass
