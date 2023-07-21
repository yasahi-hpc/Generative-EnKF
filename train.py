"""
Usage
python train.py --model_name <model_name>
"""

import time
import argparse
from model.trainers import get_trainer

def parse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-dirname', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='cases/model', \
                        type=str, \
                        choices=None, \
                        help='directory of inputfile', \
                        metavar=None
                       )

    parser.add_argument('--filename', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='diffusion.json', \
                        type=str, \
                        choices=None, \
                        help='input file name', \
                        metavar=None
                       )

    parser.add_argument('--model_name', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='Denoising_Diffusion', \
                        type=str, \
                        choices=None, \
                        help='Model name (default: Denoising_Diffusion)', \
                        metavar=None
                       )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name

    trainer = get_trainer(model_name)(**vars(args))
    trainer.initialize()

    # Training
    total_start = time.time()
    trainer.run()
    seconds = time.time() - total_start

    trainer.finalize(seconds=seconds)
