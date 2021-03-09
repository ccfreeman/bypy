import numpy as np
from numba import jit

@jit
def _inner_merge_jit(idx, l_vals, r_vals, n_intersect, 
                      l_gr_idx, r_gr_idx, l_idx, r_idx):
    """ Helper jitted function for inner merge """
    # Make the results object of the proper proportions
    # get counts per group
    l_counts = np.diff(l_gr_idx)[l_idx] 
    r_counts = np.diff(r_gr_idx)[r_idx]

    # Multiplying these counts elementwise gives us the number of elements for each group
    group_n = l_counts * r_counts
    res_group_i = np.concatenate((np.array([0]), np.cumsum(group_n)))

    n = res_group_i[-1] # number of total rows is the dot product of the each merge-group's lengths
    m = idx.shape[1] + l_vals.shape[1] + r_vals.shape[1]

    # Make an empty array to be filled in piece by piece
    res = np.empty((n,m), dtype=idx.dtype)

    idx_end = idx.shape[1]
    l_end = idx_end + l_vals.shape[1]

    for i in range(n_intersect):

        index = idx[l_gr_idx[l_idx[i]]] # a single row representation of the index columns (for left position)

        l_arr = l_vals[l_gr_idx[l_idx[i]]:l_gr_idx[l_idx[i]+1]]
        r_arr = r_vals[r_gr_idx[r_idx[i]]:r_gr_idx[r_idx[i]+1]] 

        res_gr = res[res_group_i[i]:res_group_i[i+1]] # Get a view of the rows to be changed at this iteration

        res_gr[:, :idx_end] = index

        l_group_i = np.arange(l_counts[i]+1) * r_counts[i] # positions of each new row of l_arr in res_gr

        for j in range(l_counts[i]):
            res_gr[l_group_i[j]:l_group_i[j+1], idx_end:l_end] = l_arr[j]
            res_gr[l_group_i[j]:l_group_i[j+1], l_end:] = r_arr

    return res


def _inner_merge(idx, l_vals, r_vals, n_intersect, 
                      l_gr_idx, r_gr_idx, l_idx, r_idx):
    """ Helper function for inner merge """
    # Make the results object of the proper proportions
    l_counts = np.diff(l_gr_idx)[l_idx] # get counts per group
    r_counts = np.diff(r_gr_idx)[r_idx]

    group_n = l_counts * r_counts
    res_group_i = np.concatenate((np.array([0]), np.cumsum(group_n)))

    n = res_group_i[-1] # number of total rows is the dot product of the each merge-group's lengths
    m = idx.shape[1] + l_vals.shape[1] + r_vals.shape[1]

    # Make an empty array to be filled in piece by piece
    res = np.empty((n,m), dtype=idx.dtype)

    idx_end = idx.shape[1]
    l_end = idx_end + l_vals.shape[1]

    for i in range(n_intersect):

        index = idx[l_gr_idx[l_idx[i]]] # a single row representation of the index columns (for left position)

        l_arr = l_vals[l_gr_idx[l_idx[i]]:l_gr_idx[l_idx[i]+1]]
        r_arr = r_vals[r_gr_idx[r_idx[i]]:r_gr_idx[r_idx[i]+1]] 

        res_gr = res[res_group_i[i]:res_group_i[i+1]]

        res_gr[:, :idx_end] = index

        l_group_i = np.arange(l_counts[i]+1) * r_counts[i] # positions of each new row of l_arr in res_gr

        for j in range(l_counts[i]):
            res_gr[l_group_i[j]:l_group_i[j+1], idx_end:l_end] = l_arr[j]
            res_gr[l_group_i[j]:l_group_i[j+1], l_end:] = r_arr

    return res