Frequency Asked Questions
=========================

NumExpr has a max 32 arguments
------------------------------

This is a limitation due to a `#define` in NumPy.  It can be removed by downloading
the NumPy git repo, and changing `NPY_MAXARGS`, currently (as of NumPy 1.13) 
found in the file `numpy/numpy/core/include/numpy/ndarraytypes.h`:

https://github.com/numpy/numpy/blob/dc27edb92ec70b5c0ade8ecd1ed78884a0a0a5dc/numpy/core/include/numpy/ndarraytypes.h

Then build and install NumPy with the setup script as follows:

    git clone https://github.com/numpy/numpy.git
    cd numpy
    nano numpy/core/include/numpy/ndarraytypes.h
    
<do your changes>

    python setup.py build install

The current limit in NumExpr for the maximum number of arguments is 254.
