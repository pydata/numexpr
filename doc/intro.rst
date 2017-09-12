How it works
============

The string passed to :code:`evaluate` is compiled into an object representing the 
expression and types of the arrays used by the function :code:`numexpr`.

The expression is first compiled using Python's :code:`compile` function (this means 
that the expressions have to be valid Python expressions). From this, the 
variable names can be taken. The expression is then evaluated using instances 
of a special object that keep track of what is being done to them, and which 
builds up the parse tree of the expression.

This parse tree is then compiled to a bytecode program, which describes how to 
perform the operation element-wise. The virtual machine uses "vector registers": 
each register is many elements wide (by default 4096 elements). The key to 
NumExpr's speed is handling chunks of elements at a time.

There are two extremes to evaluating an expression elementwise. You can do each 
operation as arrays, returning temporary arrays. This is what you do when you 
use NumPy: :code:`2*a+3*b` uses three temporary arrays as large as :code:`a` or 
:code:`b`. This strategy wastes memory (a problem if your arrays are large), 
and also is not a good use of cache memory: for large arrays, the results of 
:code:`2*a` and :code:`3*b` won't be in cache when you do the add.

The other extreme is to loop over each element, as in::

    for i in xrange(len(a)):
        c[i] = 2*a[i] + 3*b[i]

This doesn't consume extra memory, and is good for the cache, but, if the 
expression is not compiled to machine code, you will have a big case statement 
(or a bunch of if's) inside the loop, which adds a large overhead for each 
element, and will hurt the branch-prediction used on the CPU.

:code:`numexpr` uses a in-between approach. Arrays are handled as chunks (of 
4096 elements) at a time, using a register machine. As Python code, 
it looks something like this::

    for i in xrange(0, len(a), 256):
       r0 = a[i:i+128]
       r1 = b[i:i+128]
       multiply(r0, 2, r2)
       multiply(r1, 3, r3)
       add(r2, r3, r2)
       c[i:i+128] = r2

(remember that the 3-arg form stores the result in the third argument, 
instead of allocating a new array). This achieves a good balance between 
cache and branch-prediction. And the virtual machine is written entirely in 
C, which makes it faster than the Python above.  Furthermore the virtual machine 
is also multi-threaded, which allows for efficient parallelization of NumPy 
operations.

There is some more information and history at:

http://www.bitsofbits.com/2014/09/21/numpy-micro-optimization-and-numexpr/

Expected performance
====================

The range of speed-ups for NumExpr respect to NumPy can vary from 0.95x and 20x, 
being 2x, 3x or 4x typical values, depending on the complexity of the 
expression and the internal optimization of the operators used. The strided and 
unaligned case has been optimized too, so if the expression contains such 
arrays, the speed-up can increase significantly. Of course, you will need to 
operate with large arrays (typically larger than the cache size of your CPU) 
to see these improvements in performance.

Here there are some real timings. For the contiguous case::

    In [1]: import numpy as np
    In [2]: import numexpr as ne
    In [3]: a = np.random.rand(1e6)
    In [4]: b = np.random.rand(1e6)
    In [5]: timeit 2*a + 3*b
    10 loops, best of 3: 18.9 ms per loop
    In [6]: timeit ne.evaluate("2*a + 3*b")
    100 loops, best of 3: 5.83 ms per loop   # 3.2x: medium speed-up (simple expr)
    In [7]: timeit 2*a + b**10
    10 loops, best of 3: 158 ms per loop
    In [8]: timeit ne.evaluate("2*a + b**10")
    100 loops, best of 3: 7.59 ms per loop   # 20x: large speed-up due to optimised pow()

For unaligned arrays, the speed-ups can be even larger::

    In [9]: a = np.empty(1e6, dtype="b1,f8")['f1']
    In [10]: b = np.empty(1e6, dtype="b1,f8")['f1']
    In [11]: a.flags.aligned, b.flags.aligned
    Out[11]: (False, False)
    In [12]: a[:] = np.random.rand(len(a))
    In [13]: b[:] = np.random.rand(len(b))
    In [14]: timeit 2*a + 3*b
    10 loops, best of 3: 29.5 ms per loop
    In [15]: timeit ne.evaluate("2*a + 3*b")
    100 loops, best of 3: 7.46 ms per loop   # ~ 4x speed-up
