#--------------------------#
#  Statistics              #
#--------------------------#

from scipy.special import gamma
from scipy import stats
import numpy as np


beta = stats.beta.pdf

# stats.beta.pdf(x, a, b)
#def beta(x, a, b):
#    return gamma(a+b)/(gamma(a)*gamma(b)) * x**(a-1) * (1-x)**(b-1)

def dbeta(x, a, b):
    return gamma(a+b)/(gamma(a)*gamma(b)) * \
           ((a-1) * x**(a-2) * (1-x)**(b-1) - x**(a-1) * (b-1) * (1-x)**(b-2))

assert dbeta(0.5, 2, 2) == 0
assert dbeta(0.6, 2, 2) != 0



#--------------------------#
#  General                 #
#--------------------------#


def sample_dist(d):
    """Sample a single element from the distribution.
    d           --> {'type': '', 'params': []}
    d['type']   --> 'normal_pos' || 'beta'
    NOTE: 'type'=='normal_pos' may not terminate"""

    t = d['type']
    params = d['params']

    if t == 'normal_pos':
        # mean = params[0]
        # stdev = params[1]
        v = -1
        while v <= 0:
            v = np.random.normal(*params)
        return v
    elif t == 'beta':
        return np.random.beta(*params)
    elif t == 'lognorm':
        return stats.lognorm.rvs(*params) 
    else:
        raise Exception('Undefined parameter distribution')
