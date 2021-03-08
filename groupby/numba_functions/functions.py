from numba import jit
import numpy as np

@jit
def size(arr):
    return arr.shape[0]

@jit
def size_ax0(arr):
    return np.array([arr[i].shape[0] for i in range(arr.shape[0])])

@jit
def size_ax1(arr):
    return np.array([arr[:, i].shape[0] for i in range(arr.shape[1])])

##########################

@jit
def mean(arr):
    return np.mean(arr)

@jit
def mean_ax0(arr):
    return np.array([np.mean(arr[i]) for i in range(arr.shape[0])])

@jit
def mean_ax1(arr):
    return np.array([np.mean(arr[:, i]) for i in range(arr.shape[1])])

##########################

@jit
def median(arr):
    return np.median(arr)

@jit
def median_ax0(arr):
    return np.array([np.median(arr[i]) for i in range(arr.shape[0])])

@jit
def median_ax1(arr):
    return np.array([np.median(arr[:, i]) for i in range(arr.shape[1])])

##########################

@jit
def std(arr):
    return np.std(arr)

@jit
def std_ax0(arr):
    return np.array([np.std(arr[i]) for i in range(arr.shape[0])])

@jit
def std_ax1(arr):
    return np.array([np.std(arr[:, i]) for i in range(arr.shape[1])])

##########################

@jit
def var(arr):
    return np.var(arr)

@jit
def var_ax0(arr):
    return np.array([np.std(arr[i]) for i in range(arr.shape[0])])

@jit
def var_ax1(arr):
    return np.array([np.var(arr[:, i]) for i in range(arr.shape[1])])

##########################

@jit
def sum(arr):
    return np.sum(arr)
    
@jit
def sum_ax0(arr):
    return np.array([np.sum(arr[i]) for i in range(arr.shape[0])])

@jit
def sum_ax1(arr):
    return np.array([np.sum(arr[:, i]) for i in range(arr.shape[1])])

##########################