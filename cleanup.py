"""
Usage:
    python cleanup.py <symbolic_link_to_result_directory>
"""

import argparse
import shutil
import pathlib

__author__     = 'Yuuichi ASAHI'
__date__       = '2022/04/11'
__version__    = '1.0'
__maintainer__ = 'Yuuichi ASAHI'
__email__      = 'asahi.yuichi@jaea.go.jp'
__status__     = 'Production'

def parse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('symdir', \
                        action='store', \
                        nargs=None, \
                        const=None, \
                        default=None, \
                        type=str, \
                        choices=None, \
                        help='Symbolic link to the directory you may want to delete', \
                        metavar=None
                       )

    parser.add_argument('--verbose', \
                        action='store_false', \
                        default=True, \
                        help='ask if it is OK to remove or not'
                       )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    symdir = args.symdir
    verbose = args.verbose

    pwd = pathlib.Path.cwd()
    sym = pathlib.Path(symdir)
    path = sym.resolve()

    remove_ok = True
    if verbose:
        remove_ok = input(f'Are you sure to remove the directory "{symdir} ({path})" [default=n]: ')
        remove_ok = False if not remove_ok else remove_ok in ['y', 'yes', 'Y', 'Yes']

    # Remove the directory and contents if current directory is not under path
    if str(path) in str(pwd):
        print(f'current directory {pwd} is under {path}. Do not remove results.')
        remove_ok = False

    if remove_ok:
        print(f"removing {symdir}")
        # First remove the symbolic link
        sym.unlink()

        # Remove the directory and contents
        shutil.rmtree(path)
    else:
        print(f"Do not remove {symdir}")
