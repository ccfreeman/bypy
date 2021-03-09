from numba import jit
import groupby.numba_functions as nbf
import numpy as np

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