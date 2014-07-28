# version 0.2 (newer)

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
    
def array_to_dict(arr):
    """Convert numpy array to matrix with (row, col) keys
    >>> import numpy as np
    >>> arr = np.array([[1,2],[3,4]])
    >>> sol = array_to_dict(arr)
    >>> [(x,sol[x]) for x in sorted(sol)]
    [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]
    """
    d = dict()
    for i,row in enumerate(arr):
        for j,v in enumerate(row):
            d[i,j] = v
    return d

def geometric_series(a, r, n):
    """
    >>> import numpy as np
    >>> geometric_series(2, .99, 3)
    array([ 2.    ,  1.98  ,  1.9602])
    """
    return a * np.logspace(0, n, num=n, base=r, endpoint=False)
    
    
def digitize_right_closed(x, bins):
    """Version that maps [0,0.5,1] into 2 bins (inclusive on right side)
    >>> import numpy as np
    >>> digitize_right_closed(x=[0,0.25,0.5,0.75,1], bins=[0,0.5,1])
    array([0, 0, 1, 1, 1])
    """
    bins_out = np.digitize(x, bins)
    final_right = bins[-1]
    for i,v in enumerate(x):
        if v == final_right:
            bins_out[i] -= 2
        else:
            bins_out[i] -= 1
    return bins_out

def bin_midpoints(bins):
    """Return midpoints of bin boundaries
    >>> import numpy as np
    >>> b = np.linspace(0,1,11)
    >>> bin_midpoints(b)
    array([ 0.05,  0.15,  0.25,  0.35,  0.45,  0.55,  0.65,  0.75,  0.85,  0.95])
    """
    return 0.5*(bins[1:]+bins[:-1])

#--- plotting
def linecycler():
    """
    >>> linecycler().next()
    '-'
    """
    lines = ['-','--','-.',':']
    return itertools.cycle(lines)
