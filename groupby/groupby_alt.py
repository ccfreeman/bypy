import numpy as np
from numba import jit

import groupby.numba_functions as nbf
from collections import defaultdict


# DEFINE A FEW FUNCTIONS USED BY Group CLASS
@jit
def _agg_grps(ar, f, gr_idx, gr_n, col_n):
    """
        Iterate through groups and apply a specified function to each column
    """
    res = np.empty(shape=(gr_n, col_n), dtype=np.float32)
    for ri in range(gr_idx.shape[0] - 1):
        grar = ar[gr_idx[ri]:gr_idx[ri+1]]
        res[ri] = np.array([_agg_col(ar=grar[:, ci], fi=f[ci]) for ci in range(ar.shape[1])])
    return res


@jit
def _agg_col(ar, fi):
    """
    """
    if fi == 0:
        return ar[0]
    if fi == 1:
        return nbf.size(ar)
    if fi == 2:
        return nbf.mean(ar)
    if fi == 3:
        return nbf.median(ar)
    if fi == 4:
        return nbf.std(ar)
    if fi == 5:
        return nbf.var(ar)
    if fi == 6:
        return np.nan
    if fi == 7:
        return nbf.sum(ar)

class Group:
    def __init__(self, shape, shape_vals, dtype):
        self.arr = np.empty(shape, dtype=dtype)
        self.n = 0

    def add(self, row, row_vals, i):
        self.arr[self.n] = row
        self.n += 1

class GroupByAlt:  
    fmap = {
        'index':0,
        'count':1,
        'size':1,
        'mean':2,
        'median':3,
        'std':4,
        'var':5,
        'nunique':6,
        'sum':7,
#         'max':7,
#         'min':8,
    }
    
    def __init__(self, ar, by):
        """ Initialize the groupby object with a 2d array """
        self.by = by
        
        aux = np.full(ar.shape[1], True)
        aux[by] = False
        self.val_idx = np.nonzero(aux)[0] # A list of the column indices for values (group indices removed)
        self.vals = ar[:, self.val_idx]
        
        self.key_cols = ar[:, by]
        key_tuples = [tuple(vals) for vals in self.key_cols]
        
        # Use hashing to store each group. Each key corresponds to a Group object
        self.groups = defaultdict(lambda: Group(shape=ar.shape, shape_vals=self.vals.shape, dtype=ar.dtype))
        for i in range(ar.shape[0]):
            self.groups[key_tuples[i]].add(row=ar[i], row_vals=self.vals[i], i=i)
        
        # Create the array holding the groups, and store the start/stop index of each group
        self.arr = np.empty(ar.shape, dtype=ar.dtype)
        self.gr_idx = np.empty(len(self.groups)+1, dtype=np.int32)
        self.gr_idx[0] = 0
        i, ii = 0, 1
        for gr in self.groups.values():
            self.arr[i:i+gr.n] = gr.arr[:gr.n]
            i += gr.n
            self.gr_idx[ii] = i
            ii += 1
        
    def agg(self, fncs):
        """
        Apply a different specified function to each column of a group. The index columns will be included 
        in the result as the left-hand-side of the return array. 
        
        Parameters
        ----------
        fnc : list of strings
            Each item in the list should be one of ['count', 'size', 'mean', 'median', 'std', 'var', 'nunique', 
            'sum']. These represent the functions to be applied to each non-index columns of the data.
        
        Returns
        -------
        arr_transformed : 2d array
            An array with number of rows matching the number of groups, with transformed values of the
            non-index columns.
        
        """
        for i in self.by:
            fncs[i] = 'index'
        f = np.array([self.fmap[fncs[i]] for i in range(self.arr.shape[1])])

        return _agg_grps(ar=self.arr, f=f, gr_idx=self.gr_idx, gr_n=len(self.groups), col_n=self.arr.shape[1])