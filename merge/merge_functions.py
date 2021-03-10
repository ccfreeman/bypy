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


@jit
def _lr_merge_constr_res(idx, l_vals, r_vals, n_intersect, 
                         l_gr_idx, r_gr_idx, l_idx, r_idx, 
                         null_val, how=0):
    """ Helper jitted function for inner merge """
    # Make the results object of the proper proportions
    # get counts per group
    l_counts = np.diff(l_gr_idx)
    r_counts_match = np.diff(r_gr_idx)[r_idx]
    
    r_counts = np.ones(l_counts.shape[0], np.int32)
    r_counts[l_idx] = r_counts_match

    # Multiplying these counts elementwise gives us the number of elements for each group
    group_n = l_counts * r_counts
    res_group_i = np.concatenate((np.array([0]), np.cumsum(group_n)))

    n = res_group_i[-1] # number of total rows is the dot product of the each merge-group's lengths
    m = idx.shape[1] + l_vals.shape[1] + r_vals.shape[1]

    # Make an empty array to be filled in piece by piece
    res = np.empty((n,m), dtype=idx.dtype)

    # Generate the column-index start/stop combinations for (1) indices, (2) left values, and (3) right values
    idx_end = idx.shape[1]
    if how == 0: # left merge
        l_begin, l_end = idx_end, idx_end + l_vals.shape[1]
        r_begin, r_end = l_end, l_end + r_vals.shape[1]
    elif how == 1: # right merge
        r_begin, r_end = idx_end, idx_end + r_vals.shape[1]
        l_begin, l_end = r_end, r_end + r_vals.shape[1]
    
    # We want to know when the group on the left has a match in the right
    idx_isin_r = np.full(l_counts.shape[0], False)
    idx_isin_r[l_idx] = True
    # We also need to know where the right group matches start and stop
    l2r_gr_idx = np.empty(l_gr_idx.shape[0], np.int32)
    l2r_gr_idx[l_idx] = r_idx

    # We iterate through each group in the left data object
    for i in range(l_counts.shape[0]):

        index = idx[l_gr_idx[i]] # a single row representation of the index columns (for left position)

        l_arr = l_vals[l_gr_idx[i]:l_gr_idx[i+1]]
        if idx_isin_r[i]:
            # If the index-key at this iteration is shared, get the right-hand values
            r_arr = r_vals[r_gr_idx[l2r_gr_idx[i]]:r_gr_idx[l2r_gr_idx[i]+1]]
        else:
            # If the index-key is only in the data being joined on, generate a null array
            r_arr = np.full(r_vals.shape[1], null_val)

        res_gr = res[res_group_i[i]:res_group_i[i+1]] # Get a view of the rows to be changed at this iteration

        res_gr[:, :idx_end] = index

        l_group_i = np.arange(l_counts[i]+1) * r_counts[i] # positions of each new row of l_arr in res_gr

        for j in range(l_counts[i]):
            res_gr[l_group_i[j]:l_group_i[j+1], l_begin:l_end] = l_arr[j]
            res_gr[l_group_i[j]:l_group_i[j+1], r_begin:r_end] = r_arr

    return res