from groupby import GroupBy
import pandas as pd
import numpy as np

from tqdm import trange
import timeit

n, m = 10000, 4
ar = np.random.randint(10, size=n*m).reshape((n,m))
df = pd.DataFrame(ar)


if __name__ == '__main__':

    by_idx = [[0], [0,1], [0,1,2], [0,1,2,3]]
    n_reps = 1000

    print('**************************')
    print("*** GROUPBY COMPARISON ***")
    print('**************************')
    for ii in range(len(by_idx)):

        print("ByPy groupby, {0}x{1}, by {2}, {3} repetitions...".format(n, m, by_idx[ii], n_reps))
        for i in trange(n_reps):
            GroupBy(ar, by=by_idx[ii]) #[GroupBy(ar, by=[0,1]) for i in trange(1000)]

        print("Pandas groupby...")
        for i in trange(n_reps):
            pd.DataFrame(ar).groupby(by_idx[ii])

    print('**************************')
    print("***** AGG COMPARISON *****")
    print('**************************')
    for ii in range(len(by_idx) - 1):
        gb = GroupBy(ar, by=by_idx[ii]) 
        df_gb = pd.DataFrame(ar).groupby(by_idx[ii])
        agg_keys = [{1:'count', 2:'mean', 3:'sum'}, {2:'mean', 3:'sum'}, {3:'var'}]
        # Make sure numba functions are compiled
        gb.agg(agg_keys[ii])

        print("ByPy agg, {0}x{1}, by {2}, {3} repetitions...".format(n, m, by_idx[ii], n_reps))
        for i in trange(n_reps):
            gb.agg(agg_keys[ii])

        print("Pandas agg...")
        for i in by_idx[ii]:
            if i in agg_keys[ii]:
                del agg_keys[ii][i]
        for i in trange(n_reps):
            df_gb.agg(agg_keys[ii])
