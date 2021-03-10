ByPy
====

A collection of classes and functions to perform groupby and merge operations on NumPy arrays. For times when you don't want to use Pandas DataFrames, these classes will help you achieve some of the functionality of grouping data by columns, and merging two datasets by different columns and methods (left, right, inner, outer). 

We have tried as much as possible to be as performative as Pandas DataFrame methods. We utilize NumPy's setops suite of functions and Numba jitted functions to largely achieve this goal.