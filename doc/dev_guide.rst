Developer Guide
===============

This guide is intended for people who want to contribute to the development 
of NumExpr, typically by adding new functions or operations.

Abstract Syntax Tree Parsing
----------------------------

Introduction to the Virtual Machine
-----------------------------------

Code Generation
---------------

If you want to hand-edit a :code:`*_GENERATED` source file you can build/install
with generation disabled::

    python setup.py install --nogen
    

Adding New Functions
--------------------


Optimization Tips
-----------------

Auto-vectorization
^^^^^^^^^^^^^^^^^

One of the big speed-ups 3.0 gained over 2.6 was to re-write the task loops so
they use array indexing instead of pointer math. This enables modern compilers 
to auto-vectorize the loops with SIMD instructions. A good introductory guide 
is provided by Lockless:

* http://locklessinc.com/articles/vectorize/

Only certain patterns are recognized by certain compilers. Read up on the 
patterns:

* GCC: http://gcc.gnu.org/projects/tree-ssa/vectorization.html
* MSVC: https://msdn.microsoft.com/en-us/library/hh872235.aspx
* Intel: https://software.intel.com/en-us/articles/a-guide-to-auto-vectorization-with-intel-c-compilers
* LLVM: http://llvm.org/docs/Vectorizers.html

Based on my experience, MSVC fails to perform many auto-vectorizations that 
GCC employs. It can manage the major operations, like multiply, but it seems 
to fail to generate many edge cases, doesn't cover cast operations, etc. One 
possible line of work that could improve performance on Windows would be to use 
LLVM or ICC instead.

1. One should not do in-line operations, where the output is an input::

       x[i] = x[i]*y[i]
 
   Therefore the allocation of temporaries by the program assembler is important. 
   The number of temporaries should not be minimized but rather alternate so that
   the compiler doesn't resort to scalar loops.
  
2. Gather operations will enable some NumPy strided arrays but generally only 
   for powers of 2. However it's fairly unlikely the compiler will understand
   that the stride is a power of two, since stride for both arrays is a 
   variable.
3. 

Thread Pool
^^^^^^^^^^^

[We use a custom barrier implementation]

[Windows sometimes fails to launch a thread, that results in an overall slow-down]

[Perhaps we could reduce the number threads for small arrays by having a decrementing
counter?]

Benchmarking
------------

Benchmarking is enabled for the virtual machine by calling setup with the 
:code:`--bench` flag, e.g.::

    python setup.py install --bench




Memory Checking
---------------

We use :code:`valgrind` on Linux to test the virtual machine for memory leaks,
invalid reads and writes, and other troublesome C bugs.  Typical usage::

    valgrind --tool=memcheck --leak-check=full -v --suppressions=valgrind-python.supp \
                                          python -E -tt numexpr3/tests/simple_ne3.py

There will be a handful of results from CPython standard libaries.  Search 
for occurances of 'numexpr' to see if any of the virtual machine is leaking.
The suppression file is valid for Python <= 3.5.  With Python 3.6 :code:`valgrind` 
will generate about 100,000 false positives as the memory management was changed.
