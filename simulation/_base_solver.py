import abc
import os
import json
import pathlib
from .utils import save_meta, split_dir_and_file, save_json
from .models import get_model

class _BaseSolver(abc.ABC):
    """
    Base class for solver
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model_name = kwargs.get('model_name')
        dirname = kwargs.get('dirname')
        filename = kwargs.get('filename')

        # Update dirname and filename
        dirname, filename = split_dir_and_file(full_path=filename, default_dirname=dirname)
        json_file = pathlib.Path(dirname) / filename
        
        if not json_file:
            raise FileNotFoundError(f'input file {filename} not found at {dirname}')
         
        with open(json_file, 'r') as f:
            self.json_data = json.load(f)
        
        self.settings   = self.json_data['settings']
        if 'in_case_name' in self.settings:
            in_case_name = self.settings['in_case_name']
            self.json_data = self._merge_json_data(json_data=self.json_data, in_case_name=in_case_name)

        self.grids      = self.json_data['grids']
        self.simulation = self.json_data['simulation']

        self.nbiter  = self.simulation['nbiter']
        self.mode    = self.simulation['mode']
        self.nb_runs = self.settings.get('nb_runs', 1)

        self.acceptable_modes = []

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        self.model.finalize(*args, **kwargs, result_dir=self.result_dir)

    def _merge_json_data(self, json_data, in_case_name):
        base_out_dir = self.settings['out_dir']
        prev_dir = pathlib.Path(base_out_dir) / in_case_name
        prev_json_file = prev_dir / 'settings.json'
        if not prev_json_file.exists():
            raise IOError(f'Previous simulation result does not exist in {prev_dir}')

        with open(prev_json_file, 'r') as f:
            prev_json_data = json.load(f)

        # Merge jsons (priotizing the new json)
        merged_json_data = {**prev_json_data, **json_data}
        
        return merged_json_data

    def _prepare_dirs(self):
       base_out_dir = self.settings['out_dir'] 
       case_name = self.settings['case_name'] 

       # Create <out_dir>/<case_name> and <out_dir>/<case_name>/results
       out_dir = pathlib.Path(base_out_dir) / case_name
       result_dir = out_dir / 'results'
       print(f'--- Start preparing directories at {out_dir} ---')
       if not out_dir.exists():
           out_dir.mkdir(parents=True)

       if not result_dir.exists():
           result_dir.mkdir(parents=True)

       print(f'--- directories are ready at {out_dir} ---')
       symdir = case_name
       if not os.path.exists(symdir):
           os.symlink(out_dir, symdir)
           print(f'--- symbolic_link to {out_dir}: {symdir} ---')

       # Save Meta data
       save_meta(dirname=out_dir, filename='meta.txt')

       # Save json file
       save_json(dirname=out_dir, json_data=self.json_data, filename='settings.json')

       return result_dir
