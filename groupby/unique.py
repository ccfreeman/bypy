import numpy as np

def unique(ar, keycols):
    """
    """
    # Put the array in contiguous position in memory (changing dtype to different size requires C-contiguous)
    arc = np.ascontiguousarray(ar)
    # Make a view of the data as a flattened, structured array
    dtype = [(f'{i}', arc.dtype) for i in range(arc.shape[1])]
    arv = arc.view(dtype)[:, 0]
    
    # Now slice out just the key columns. Sorting by them will still affect arv
    arvkey = arv[[f'{i}' for i in keycols]]    
    arvkey.sort()
    
    # Now make a mask to find unique values in arvkey
    mask = np.empty(arv.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = arvkey[1:] != arvkey[:-1]
    
    # Compute the index start positions for each group in the array
    idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
    aret = arv.view(ar.dtype).reshape(ar.shape)
    keys = aret[mask][:, keycols]
    keyv = arvkey[mask]
    
    return aret, idx, keys, keyv

