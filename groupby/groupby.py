import numpy as np
from numba import jit

import groupby.numba_functions as nbf
from .unique import unique


# DEFINE A FEW FUNCTIONS USED BY GroupBy CLASS
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



class GroupBy:    
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
        self.arr, self.gr_idx, self.keys = unique(ar, by)
        
        self.by = by
        
        # Separate the matrix into two groups: indices and values
        aux = np.full(self.arr.shape[1], True)
        aux[by] = False
        self.val_cols = np.nonzero(aux)
        
        self.idx = self.arr[:, self.by]
        self.vals = self.arr[:, self.val_cols]
        
        # Initialize a list of group values. This may be computed once if needed for aggregation functions
        # self.gr_vals_list = None
    

    def get_group(self, idx):
        """
            Return a list of all groups (idx, vals). Can also only return values in group
            
            Parameters
            ----------
            only_vals : optional, False
                Only return groups of values instead of (idx, vals). 
            
            Returns
            -------
            groups : list
                A list of 2d arrays representing each group
        """
        if only_vals:
            if self.gr_vals_list is None:
                self.gr_vals_list = [
                    self.vals[self.gr_idx[i]:self.gr_idx[i+1]] for i in range(self.gr_idx.shape[0] - 1)
                ]
                return self.gr_vals_list
            else:
                return self.gr_vals_list
            
        return [self.arr[self.gr_idx[i]:self.gr_idx[i+1]] for i in range(self.gr_idx.shape[0] - 1)]

    
    # def apply(self, fnc):
    #     """
    #     Apply a function to each column of a group. The index columns will be included in the result as the left-
    #     hand-side of the return array. 
        
    #     Parameters
    #     ----------
    #     fnc : string
    #         Should be one of ['count', 'size', 'mean', 'median', 'std', 'var', 'nunique', 'sum']. 
    #         The function to be applied to all non-index columns of the data.
        
    #     Returns
    #     -------
    #     arr_transformed : 2d array
    #         An array with number of rows matching the number of groups, with transformed values of the
    #         non-index columns.
        
    #     """
    #     f = self.fncs_apply[self.fnc_dict[fnc]]
        
    #     if self.gr_vals_list is None:
    #         # We can store some values here to save ourselves from recomputing again later on
    #         self.get_groups(only_vals=True)
            
    #     res = np.array([f(gr) for gr in self.gr_vals_list])
        
    #     return np.concatenate((self.keys, res), axis=1)
    
        
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

        return _agg_grps(ar=self.arr, f=f, gr_idx=self.gr_idx, gr_n=self.keys.shape[0], col_n=self.arr.shape[1])

        # res = np.empty(shape=(self.keys.shape[0], self.arr.shape[1]), dtype=np.float32)
    
        # for ri in range(self.gr_idx.shape[0] - 1):
        #     grar = self.arr[self.gr_idx[ri]:self.gr_idx[ri+1]]
        #     res[ri] = np.array([self._agg_col(grar[:, ci], f[ci]) for ci in range(self.arr.shape[1])])
        # return res
