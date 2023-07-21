"""
Usage
python run.py --mode {default, kf}
"""

import time
import argparse
from simulation.solvers import get_solver

def parse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-dirname', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='cases/simulation', \
                        type=str, \
                        choices=None, \
                        help='directory of inputfile', \
                        metavar=None
                       )

    parser.add_argument('--filename', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='dns.json', \
                        type=str, \
                        choices=None, \
                        help='input file name', \
                        metavar=None
                       )

    parser.add_argument('--model_name', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='DNS', \
                        type=str, \
                        choices=None, \
                        help='Model name (DNS, Nudging, EnKF, LETKF, NoDA, DatasetFactory, EFDA)', \
                        metavar=None
                       )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name

    solver = get_solver(model_name)(**vars(args))

    solver.initialize()

    # Start simulation
    start = time.time()
    solver.run()
    seconds = time.time() - start

    solver.finalize(seconds=seconds)
