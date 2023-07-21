import pathlib
import copy
from .models import get_model
from ._base_solver import _BaseSolver

class DA_Solver(_BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create <out_dir>/<case_name> and the symbolic link <case_name>
        self.result_dir = self._prepare_dirs()

        self.acceptable_modes = ['default', 'perturbed']
        if not self.mode in self.acceptable_modes:
            raise ValueError(f'unacceptable mode: {self.mode}')

    def run(self, *args, **kwargs):
        for shot_idx in range(self.nb_runs):
            result_dir = self.result_dir / f'shot{shot_idx:03}' if self.nb_runs > 1 else self.result_dir
            json_data = copy.deepcopy(self.json_data)
            self.model = get_model(self.model_name)(model_name=self.model_name, json_data=json_data, result_dir=str(result_dir))
            self.model.initialize(*args, **kwargs)

            # Spinup for 1+1 year (do not diag)
            for it in range(self.nbiter):
                self.model.solve()

            # Data assimulation if needed
            self.model.initialize(mode='enable_da')

            # Simulation with data assimilation
            for it in range(self.nbiter):
                self.model.diag()
                self.model.solve()
