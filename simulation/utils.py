import argparse
import subprocess
import json
import numpy as np
import pathlib
from distutils.util import strtobool
from datetime import timedelta

def str2num(datum):
    """ Convert string to integer or float"""

    try:
        return int(datum)
    except:
        try:
            return float(datum)
        except:
            try:
                return strtobool(datum)
            except:
                return

def sec_to_hh_mm_ss(seconds):
    # create timedelta and convert it into string
    td_str = str(timedelta(seconds=seconds))

    hh, mm, ss = td_str.split(':')
    hh, mm, ss = str2num(hh), str2num(mm), str2num(ss)
    return hh, mm, ss

def create_scan_parameters(scan_parameters):
    if type(scan_parameters) is not dict:
        raise ValueError('scan_parameters must be dictionary')

    case_names = None
    case_values = None
    for scan_key, scan_values in scan_parameters.items():
        if case_names is None:
            _case_names = [f'{scan_key}{scan_value}' for scan_value in scan_values]
            _case_values = [{scan_key: scan_value} for scan_value in scan_values]
        else:
            _case_names = []
            _case_values = []
            for case_name in case_names:
                _case_names.extend([f'{case_name}_{scan_key}{scan_value}' for scan_value in scan_values])

            for case_value in case_values:
                _case_values.extend([{**case_value, **{scan_key: scan_value}} for scan_value in scan_values])

        case_names = _case_names.copy()
        case_values = _case_values.copy()

    return case_names, case_values

def split_dir_and_file(full_path, default_dirname='./'):
    """
    get dirname and filename from full_path
    """
    def remove_first_and_last_slash(s):
        if s.endswith('/'):
            s = s[:-1]
        if s.startswith('/'):
            s = s[1:]
                        
        return s
    
    full_path = remove_first_and_last_slash(full_path)
    
    if '/' in full_path:
        splitted = full_path.split('/')
        dirname = '/'.join(splitted[:-1])
        filename = splitted[-1]
    else:
        # Then filename
        dirname = default_dirname
        filename = full_path
        
    return dirname, filename

def observe_with_zeros(data, obs_interval=1, noise_level=1):
    """
    adding gaussian noise for osse
    In LETKF, noise_level == leads, about 10% to the signal level
    """

    if data.ndim == 2:
        assert data.shape[1] % obs_interval == 0

        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        masks = zeros
        masks[:, ::obs_interval] = ones[:, ::obs_interval]

        noisy_data = masks * (data + np.random.normal(loc=0, scale=1, size=masks.shape) * noise_level)
        return noisy_data
    elif data.ndim == 1:
        assert data.shape[0] % obs_interval == 0

        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        masks = zeros
        masks[::obs_interval] = ones[::obs_interval]

        noisy_data = masks * (data + np.random.normal(loc=0, scale=1, size=masks.shape) * noise_level)
        return noisy_data
    else:
        raise ValueError(f'data should be 1D or 1D time series data (2D). data shape is {data.shape}')

def observe(data, obs_interval=1, noise_level=1):
    """
    adding gaussian noise for osse
    In LETKF, noise_level == leads, about 10% to the signal level
    """
    assert data.shape[0] % obs_interval == 0

    obs_data = data[::obs_interval]
    noisy_data = obs_data + np.random.normal(loc=0, scale=1, size=obs_data.shape) * noise_level
    return noisy_data

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def save_meta(dirname, filename='meta.txt'):
    """
    Save git version, date, user, and hostname
    """

    def get_meta():
        meta = ""
                                                                                             
        args_dict = {
            'Git version       : ': ['git', 'rev-parse', 'main'],
            'Date              : ': ['date'],
            'User              : ': ['whoami'],
            'Host name         : ': ['hostname'],
            'Current directory : ': ['pwd'],
        }

        for meta_name, args in args_dict.items():
            stdout = subprocess.run(args=args, encoding='utf-8', stdout=subprocess.PIPE).stdout
            meta += (meta_name + stdout)
        
        meta = meta[:-1] # removing the final '\n'
        return meta

    meta = get_meta()
    if not dirname.exists():
        dirname.mkdir(parents=True)

    meta_file = dirname / filename
    with open(meta_file, 'w') as f:
        f.write(meta)

def save_json(dirname, json_data, filename='settings.json'):
    if type(json_data) is not dict:
        raise IOError('json_data should be dictionary')

    json_file = dirname / filename
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4, separators=(',', ': '))

def make_divisors(n):
    lower_divisors, upper_divisors = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            lower_divisors.append(i)
            if i != n // i:
                upper_divisors.append(n//i)
        i += 1
    return lower_divisors + upper_divisors[::-1]

def dict2str(dict, separators=(', ', ': '), add_new_line_in_the_end=False):
    tmp_str = ''
    sep_value, sep_key = separators
    last_element = list(dict.keys())[-1]
    for key, values in dict.items():
        if type(values) is not list:
            values = [values]
        sep_values = [sep_value for _ in values]
        sep_values[-1] = ''

        tmp_str += f'{key}{sep_key}'

        for value, tmp_sep_value in zip(values, sep_values):
            tmp_str += f'{value}{tmp_sep_value}'

        # New line
        if add_new_line_in_the_end or (key is not last_element):
            tmp_str += '\n'

    return tmp_str

def print_details(log_dict, dirname, filename='log.txt'):
    """log_dict can be empty if nn_model is not relying on dl model
    """
    if type(log_dict) is not dict:
        return

    log_filename = pathlib.Path(dirname) / filename
    with open(log_filename, 'w') as f:
        log_str = dict2str(log_dict)
        print(log_str, file=f)
