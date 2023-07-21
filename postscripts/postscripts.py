from .dns_postscripts import DNS_Postscripts
from .nudging_postscripts import Nudging_Postscripts
from .enkf_postscripts import EnKF_Postscripts
from .efda_postscripts import EFDA_Postscripts
from .rlda_postscripts import RLDA_Postscripts

def get_postscripts(name):
    POST_SCRIPTS = {
        'DNS': DNS_Postscripts,
        'Nudging': Nudging_Postscripts,
        'EnKF': EnKF_Postscripts,
        'LETKF': EnKF_Postscripts,
        'NoDA': EnKF_Postscripts,
        'EFDA': EFDA_Postscripts,
        'SoftActorCritic': RLDA_Postscripts,
    }

    for n, p in POST_SCRIPTS.items():
        if n.lower() == name.lower():
            return p

    raise ValueError(f'post_script {name} is not defined')
