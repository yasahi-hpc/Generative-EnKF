from .dns_solver import DNS_Solver
from .da_solver import DA_Solver

def get_solver(solver_name):
    SOLVERS = {
               'DNS': DNS_Solver,
               'Nudging': DA_Solver,
               'EnKF': DA_Solver,
               'LETKF': DA_Solver,
               'NoDA': DA_Solver,
               'DatasetFactory': DA_Solver,
               'EFDA': DA_Solver,
              }

    for n, s in SOLVERS.items():
        if n.lower() == solver_name.lower():
            return s

    raise ValueError(f'solver {solver_name} is not defined')
