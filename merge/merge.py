import numpy as np

from groupby import GroupBy
from .merge_functions import _inner_merge_jit, _inner_merge, _lr_merge_constr_res


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

        if isinstance(self.l_gb.idx.dtype, np.floating) and isinstance(self.r_gb.idx.dtype, np.floating):
            self.null_val = np.nan
        else:
            self.null_val = -99999
      
      
    def _merge_prefix(self, l_gb_keys, r_gb_keys):
        """ A shared function between several merge types """
        # We'll need contiguous arrays to get the proper view of our keys
        l_keyc, r_keyc = np.ascontiguousarray(l_gb_keys), np.ascontiguousarray(r_gb_keys)
        dtype = [(f'{i}', l_keyc.dtype) for i in range(l_keyc.shape[1])] # l_gb.on and r_gb.on should be of same length
        # Get a view of the keys (the one stored in the GroupBy objects may have differently named fields)
        l_keyv, r_keyv = l_keyc.view(dtype)[:, 0], r_keyc.view(dtype)[:, 0]
        
        # Find the intersection between the two key-sets and the indices in each set for those intersections
        intersect, l_idx, r_idx = np.intersect1d(l_keyv, r_keyv, assume_unique=True, 
                                                 return_indices=True)
        return intersect, l_idx, r_idx
    
    
    def inner(self, jitted=True):
        """ inner join on the specified columns of the Merge object """
        
        intersect, l_idx, r_idx = self._merge_prefix(self.l_gb.keys, self.r_gb.keys)

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
    
    
    def left(self, null_val=None):
        """ left outer join on specified columns """
        if null_val is None:
            null_val = self.null_val
        return self._lr_merge(l_gb=self.l_gb, r_gb=self.r_gb, how='left', null_val=null_val)
    
    
    def right(self, null_val=None):
        """ right outer join on specified columns """
        if null_val is None:
            null_val = self.null_val
        return self._lr_merge(l_gb=self.r_gb, r_gb=self.l_gb, how='right', null_val=null_val)
    
    
    def _lr_merge(self, l_gb, r_gb, how, null_val):
        """
            As left and right outer joins are symmetrical, we can reuse code, and just appropriately set the 
            left and right data objects to return the desired result. The default is left merge.
            
            The numpy null value is a floating point, so we need to designate what will signify null values 
            for other datatypes, such as integers. Default is -99,999.
        """
        if how == 'left':
            how = 0
        elif how == 'right':
            how = 1
            
        intersect, l_idx, r_idx = self._merge_prefix(l_gb.keys, r_gb.keys)
        
        return _lr_merge_constr_res(idx=l_gb.idx, l_vals=l_gb.vals, r_vals=r_gb.vals, 
                                    n_intersect=intersect.shape[0], l_gr_idx=l_gb.gr_idx,
                                    r_gr_idx=r_gb.gr_idx, l_idx=l_idx, r_idx=r_idx, 
                                    null_val=null_val, how=how)
        