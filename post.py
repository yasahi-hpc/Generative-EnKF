"""
Usage
python post.py --filename 
"""

import time
import argparse
from postscripts.postscripts import get_postscripts

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
                        help='Model name (DNS, Nudging, EnKF, LETKF, NoDA, DatasetFactory, EFDA, SoftActorCritic)', \
                        metavar=None
                       )

    parser.add_argument('--plot_series', \
                        action='store_true', \
                        default=False, \
                        help='plot time series data or not'
                       )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name

    post_script = get_postscripts(model_name)(**vars(args))
    post_script.initialize()

    # Convert data or save figures
    start = time.time()
    post_script.run()
    seconds = time.time() - start

    post_script.finalize(seconds=seconds)
