import numpy as np

from groupby import GroupBy
from .merge_functions import _inner_merge_jit, _inner_merge

class Merge:
    def __init__(self, l, r, l_on, r_on):
        """
            The Merge object implements different types of merges between two arrays. It utilizes
            the sorting an group-finding characterstics of the GroupBy object to facilitate the
            merges, as well as functions from NumPy's setops suite.
        """
        self.l_on = l_on
        self.r_on = r_on
        
        # Merge supports l and r parameters to be either arrays, or GroupBy objects. 
        # If they are arrays, we need to perform GroupBy initialization to get keys
        if isinstance(l, GroupBy) and (l.by == l_on):
            self.l_gb = l
        else:
            self.l_gb = GroupBy(l, l_on)
            
        if isinstance(r, GroupBy) and (r.by == r_on):
            self.r_gb = r
        else:
            self.r_gb = GroupBy(r, r_on)
        
    
    def inner(self, jitted=True):
        """ inner join on the specified columns of the Merge object """
        
        # We'll need contiguous arrays to get the proper view of our keys
        l_keyc, r_keyc = np.ascontiguousarray(self.l_gb.keys), np.ascontiguousarray(self.r_gb.keys)
        dtype = [(f'{i}', l_keyc.dtype) for i in range(l_keyc.shape[1])] # l_gb.on and r_gb.on should be of same length
        # Get a view of the keys (the one stored in the GroupBy objects may have differently named fields)
        l_keyv, r_keyv = l_keyc.view(dtype)[:, 0], r_keyc.view(dtype)[:, 0]
        
        # Find the intersection between the two key-sets and the indices in each set for those intersections
        intersect, l_idx, r_idx = np.intersect1d(l_keyv, r_keyv, 
                                                 assume_unique=True, return_indices=True)

        # The creation of the return array is spedup with numba no-python mode
        if jitted:
            res = _inner_merge_jit(idx=self.l_gb.idx, l_vals=self.l_gb.vals, r_vals=self.r_gb.vals, 
                                    n_intersect=intersect.shape[0], l_gr_idx=self.l_gb.gr_idx,
                                    r_gr_idx=self.r_gb.gr_idx, l_idx=l_idx, r_idx=r_idx)
        else:
            res = _inner_merge(idx=self.l_gb.idx, l_vals=self.l_gb.vals, r_vals=self.r_gb.vals, 
                               n_intersect=intersect.shape[0], l_gr_idx=self.l_gb.gr_idx,
                               r_gr_idx=self.r_gb.gr_idx, l_idx=l_idx, r_idx=r_idx)
        return res