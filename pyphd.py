import numpy as np
import random
import os
from collections import Mapping
from operator import itemgetter
import itertools

#--- statistics
def entropy_log2(p):
    # hack for singleton p
    if p == 1 or p == 0:
        return 0

    prob = np.array(p)
    return -prob*np.log2(prob) - (1-prob)*np.log2(1-prob)


#--- file utils
def dict_val_or(d, k, v):
    try:
        return d[k]
    except KeyError:
        return v

def ensure_dir(d):
    try:
        os.mkdir(d)
    except OSError:
        pass # dir already exists

#--- random
def argextreme(d, f, rand=True):
    """ Returns (key,value) for some function (like min/max)
        if d is a list, returns (index,value)
    """
    if isinstance(d, Mapping):
        v = f(d.itervalues())
        keys = (k for k in d if d[k] == v)
    else:
        v = f(d)
        keys = (i for i,k in enumerate(d) if k == v)

    if rand:
        k = random.choice(list(keys))
    else:
        k = keys.next()
    
    return k,v

def argmin(d, rand=True):
    return argextreme(d, min, rand)

def argmax(d, rand=True):
    return argextreme(d, max, rand)


#--- strings
def leadingzeros(n,max_n):
    """
    >>> leadingzeros(5,500)
    '005'
    >>> leadingzeros(5,5)
    '5'
    """
    return str(n).zfill(len(str(max_n)))

#--- matrices
def is_numpy_array(obj):
    return isinstance(obj, np.ndarray) and obj.ndim == 1

def dict_to_matrix(d):
    """converts d[row,col] = 2 to numpy array([[],...,[]])
    """
    m = max(k[0] for k in d)
    n = max(k[1] for k in d)
    mat = np.zeros((m+1,n+1))
    for k in d:
        mat[k] = d[k]
    return mat

def geometric_series(a, r, n):
    """
    >>> import numpy as np
    >>> geometric_series(2, .99, 3)
    array([ 2.    ,  1.98  ,  1.9602])
    """
    return a * np.logspace(0, n, num=n, base=r, endpoint=False)

#--- plotting
def linecycler():
    """
    >>> linecycler().next()
    '-'
    """
    lines = ['-','--','-.',':']
    return itertools.cycle(lines)
