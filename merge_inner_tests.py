from merge import Merge
from groupby import GroupBy
import numpy as np
import pandas as pd
from tqdm import trange

def make_data(n,m,i):
    arr = np.random.randint(i, size=n*m).reshape((n,m))
    df = pd.DataFrame(arr)
    return arr, df

ar1, df1 = make_data(1000, 7, 3)
ar2, df2 = make_data(10000, 10, 3)


if __name__ == '__main__':

    l_on = [[0], [0], [0,1], [0,3], [0,1,2,3,4,5,6]]
    r_on = [[0], [9], [0,1], [6,8], [3,4,5,6,7,8,9]]
    n_reps = 100

    print('**************************')
    print("**** MERGE COMPARISON ****")
    print('**************************')
    for ii in range(len(l_on)):

        # gb1 = GroupBy(ar1, l_on[ii])
        # gb2 = GroupBy(ar2, r_on[ii])

        print("ByPy inner merge, jitted, left on: {0}, right on: {1}, {2} repetitions...".format(l_on[ii], r_on[ii], n_reps))
        for i in trange(n_reps):
            Merge(ar1, ar2, l_on[ii], r_on[ii]).inner(jitted=True)
        
        print("ByPy inner merge, jitted, left on: {0}, right on: {1}, {2} repetitions...".format(l_on[ii], r_on[ii], n_reps))
        for i in trange(n_reps):
            Merge(ar1, ar2, l_on[ii], r_on[ii]).inner(jitted=False)

        print("Pandas inner merge...")
        for i in trange(n_reps):
            df1.merge(df2, how='inner', left_on=l_on[ii], right_on=r_on[ii])
