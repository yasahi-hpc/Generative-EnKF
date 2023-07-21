import pathlib
from .models import get_model
from ._base_solver import _BaseSolver

class DNS_Solver(_BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create <out_dir>/<case_name> and the symbolic link <case_name>
        self.result_dir = self._prepare_dirs()

        self.acceptable_modes = ['default', 'perturbed']
        if not self.mode in self.acceptable_modes:
            raise ValueError(f'unacceptable mode: {self.mode}')

    def run(self, *args, **kwargs):
        self.model = get_model(self.model_name)(model_name=self.model_name, json_data=self.json_data, result_dir=self.result_dir)
        self.model.initialize(*args, **kwargs)

        if self.mode == 'perturbed':
            self.__run_perturbed(*args, **kwargs)
        else:
            self.__run_nature(*args, **kwargs)

    def __run_nature(self, *args, **kwargs):
        for it in range(self.nbiter):
            self.model.diag()
            self.model.solve()

    def __run_perturbed(self, *args, **kwargs):
        # Spinup (do not diag)
        for it in range(self.nbiter):
            self.model.solve()

        # Add perturbation and then perform simulation
        self.model.initialize(mode = 'perturbed')

        for it in range(self.nbiter):
            self.model.diag()
            self.model.solve()
