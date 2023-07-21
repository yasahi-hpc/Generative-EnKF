"""
Abstract model class for Lorenz-96 model
"""
import abc
from .utils import sec_to_hh_mm_ss

class _BaseModel(abc.ABC):
    """
    Abstract base model class.

    Attributes
    ----------
    _it : int
        Current iteration number

    Methods
    -------
    solve(**kwargs)
        Solve the given equation
    diag(**kwargs)
        Diagnose some physical quantities
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._time = 0.
        self._it   = 0
        self._diag_it = 0
        self._attrs = {}
        self._default_attrs = {}
        self._log_dict = {}

    @property
    def attrs(self):
        return self._attrs

    @property
    def default_attrs(self):
        return self._default_attrs

    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def diag(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def finalize(self, *args, **kwargs):
        raise NotImplementedError()

    def _print_time(self, message='diag'):
        print(f'{message}: it = {self._it}, time = {self._time:.3f}')

    def _report_elapsed_time(self, seconds, n_ens=1):
        hh, mm, ss = sec_to_hh_mm_ss(seconds)
        ss = int(ss)
        message = f'{self.model_name} simulation finished. It took {hh:02}h{mm:02}m{ss:02}s ({seconds:.2f}s) with {n_ens} ensembles for {self.nbiter} iterations.'
        return message

    def _add_json_as_attributes(self, json_data, default_values=None):
        """
        Default values are overwritten by json_data
        """

        # In case nothing is given in dict
        if (type(json_data) is not dict) and (type(default_values) is not dict):
            return

        attrs = {}
        if type(default_values) is dict:
            for key, value in default_values.items():
                setattr(self, key, value)
                attrs[key] = value

        def add_recursive(dict_like):
            for key, value in dict_like.items():
                if type(value) is dict:
                    add_recursive(value)
                else:
                    setattr(self, key, value)
                    attrs[key] = value

        if type(json_data) is dict:
            add_recursive(json_data)

        for key, value in attrs.items():
            if isinstance(value, bool):
                attrs[key] = int(value)
        self._attrs = attrs
