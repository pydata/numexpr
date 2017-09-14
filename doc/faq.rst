Frequency Asked Questions
=========================

Is NumExpr3 really fast?
------------------------

Yes.

(There are benchmarks)

NumExpr3 writes loops in a manner such that :code:`gcc` and other compilers can
auto-vectorize them with SIMD instructions. If your CPU supports AVX2 or AVX512
you should see a significant speed-up.

NumExpr3 also has an efficient thread pool. It optimizes the expression you 
pass it to minimize the number of temporary variables. Temporaries are kept 
to a small size so that the entire program expression is calculated on blocks 
that fit into the L1 cache of your individual CPU cores. NumExpr3 now manages 
its own memory for temporaries and other data containers in an arena 
walled-off from Python's memory manager.

The main limitation at present is the relatively slow math functions in the 
standard C++ math library.  In the future we plan to add a third-party math 
library, such as *Yepp!!!* or *Intel MKL*.  NumExpr3 has an internal 
complex-math library which is very speedy but sometimes trades speed for 
precision, as it is branchless.

Should I compile NumExpr3 myself or install a pre-built wheel?
--------------------------------------------------------------

If you have a new-ish compiler you should use that. Wheels are often 
compiled with old compilers and old C-libraries, that won't feature all the 
performance optimizations you might otherwise see.

Is NumExpr3 really fast on Windows?
-----------------------------------

The default compiler MSVC does not feature as aggressive loop auto-vectorization
with SIMD instructions as :code:`gcc` does, so it's performance lags. In general 
it can vectorize basic operations like :code:`'a*b'` but it does not vectorize 
most special cases, including strided array operations, casts, etc. At some 
point we may try building NumExpr3 with LLVM on Windows.

That said the complex math performance of NumExpr3 on Windows destroys NumPy.

Can I do conditional operations with NumExpr?
---------------------------------------------

Within the limitations of the :code:`where(cond,if_true,if_false)` function, 
which is identical tothe Numpy :code:`where()`. I.e. you can only do whole-array 
conditional operations, like array masking.  If you want a 'stop-execution on 
this element' condition you should look at the :code:`numba` module. 

NumExpr gives an error with more than 32 arguments?
---------------------------------------------------

This is a limitation due to a macro in NumPy.  It can be removed by downloading
the NumPy git repo, and changing :code:`#define NPY_MAXARGS`, currently (as of NumPy 1.13) 
found in the file ndarraytypes.h_. Then build and install NumPy with the setup script as follows:

.. _ndarraytypes.h: https://github.com/numpy/numpy/blob/dc27edb92ec70b5c0ade8ecd1ed78884a0a0a5dc/numpy/core/include/numpy/ndarraytypes.h

..code: bash

    git clone https://github.com/numpy/numpy.git
    cd numpy
    nano numpy/core/include/numpy/ndarraytypes.h
    
<do your changes>:

..code: bash

    python setup.py build install

The current limit in NumExpr for the maximum number of arguments is 254. The 
value :code:`numexpr3.MAX_ARGS` reports the maximum number of arguments at 
compile-time.
