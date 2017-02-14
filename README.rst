What it is Numexpr3?
===================

Numexpr3 is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.

In addition, its multi-threaded capabilities can make use of all your
cores, which may accelerate computations, most specially if they are
not memory-bounded (e.g. those using transcendental functions).

Compared to NumExpr 2.6, functions have been re-written in a fashion
such that `gcc` can auto-vectorize them with SIMD instruction sets 
such as SSE2 or AVX2, if your processor supports them. Use of a newer
version of `gcc` such as 5.4 is strongly recommended.


Examples of use
===============

::

  >>> import numpy as np
  >>> import numexpr3 as ne
  # Choose large arrays for greater speed-ups compared to NumPy
  >>> a = np.arange(2**20)   
  >>> b = np.arange(2**20)

  >>> ne.evaluate("out = a + 1")   # a simple expression
  out = array([  1.00000000e+00,   2.00000000e+00,   3.00000000e+00, ...,
           9.99998000e+05,   9.99999000e+05,   1.00000000e+06])

  >>> ne.evaluate('out2 = a*b-4.1*a > 2.5*b')   # a more complex one
  out2 = array([False, False, False, ...,  True,  True,  True], dtype=bool)

  >>> ne.evaluate("out3 = sin(a) + arcsinh(a/b)")   # you can also use functions
  out3 = array([        NaN,  1.72284457,  1.79067101, ...,  1.09567006,
          0.17523598, -0.09597844])

Multi-line commands are also supported, with named intermediate temporaries.
The convention is that if an intermediate assignment target exists in the 
calling namespace, it is assumed to be a pre-allocated output and acts as a 
second return. Otherwise it is a named temporary array that may be re-referenced
within later lines::

  >>> neObj = NumExp( '''a2 = a*a; b2 = b*b
out_magic = exp( -sin(2*a2) - cos(2*b2) - 2*a2*b2''' ) 

Note: The last assignment target is 'magic'. So-called 'magic' outputs are 
promoted to the calling frame if they do not already exist. 



Datatypes supported internally
==============================

Numexpr operates internally with the following types::

    * 8-bit boolean (bool)
    * 8/16/32/64-bit unsigned integer (uint)
    * 8/16/32/64-bit signed integer (int)
    * 32/64-bit floating point numbers (float32, float64)
    * 64/128-bit, complex number (complex64, complex128)

Unicode/bytes string support is planned for the future.

Casting rules
=============

Casting rules in Numexpr follow closely those of NumPy 'safe' mode. 


Supported operators
===================

Numexpr supports the following set of operators::

    * Logical operators: and, or
    * Comparison operators: <, <=, ==, !=, >=, >
    * Unary arithmetic operators: -
    * Binary arithmetic operators: +, -, *, /, **, %, <<, >>
    * Bitwise operators: &, |, ^


Supported functions
===================

The following are the currently supported set::

  * where(bool, all1, all2): all
      all1 if the bool condition is true, allr2 otherwise.
      I.e. the ternary function from C.
  * {sin,cos,tan}(float|complex): float|complex
      Trigonometric sine, cosine or tangent.
  * {arcsin,arccos,arctan}(float|complex): float|complex
      Trigonometric inverse sine, cosine or tangent.
  * arctan2(float1, float2): float
      Trigonometric inverse tangent of float1/float2.
  * {sinh,cosh,tanh}(float|complex): float|complex
      Hyperbolic sine, cosine or tangent.
  * {arcsinh,arccosh,arctanh}(float|complex): float|complex
      Hyperbolic inverse sine, cosine or tangent.
  * {log,log10,log1p}(float|complex): float|complex
      Natural, base-10 and log(1+x) logarithms.
  * {log2,logb, exp2}(float):float
  * {exp,expm1}(float|complex): float|complex
      Exponential and exponential minus one.
  * ones_like(all): all
  * sqrt(float|complex): float|complex
      Square root.
  * abs(signed int|float|complex): float|complex
      Absolute value.
  * {fabs,fmod,fmin,fmax}(float): float
  * {ceil,floor,trunc, rint}(float): float
  * round(float): int32
  * {erf,erfc,cbrt,lgamma,tgamma}(float):float
  * ilogb(float): int32
  * {lrint,lround,nearbyint}(float): int64
  * (fdim,hypot}(float,float): float
  * {isinf,isnan,isnormal,signbit}(float): bool
  * fp_classify(float): int32
  * scalebln(float,float): int64
  * fma(float,float,float):float
     Fused-multiply add
  * conj(complex): complex
      Conjugate value.
  * {real,imag}(complex): float
      Real or imaginary part of complex.
  * complex(float, float): complex
      Complex from real and imaginary parts.


More functions can be added if you need them.  There is
space for 64k operators at present.


General routines
================

::

  * evaluate(expression, local_dict=None,
             out=None, order='K', casting='safe', **kwargs):
    Evaluate a simple array expression element-wise.  See docstrings
    for more info on parameters.  Also, see examples above.

  * NumExpr(): an object oriented version of evaluate().  
    * run( check_arrays=True, **kwargs):
       kwargs should have references to the same names called
    * disassemble(): See the program as generated by the compiler.

  * test():  Run all the tests in the test suite.

  * OPTABLE: look-up dict with all of the available functions.

  * print_versions():  Print the versions of software that numexpr
    relies on.

  * set_num_threads(nthreads): Sets a number of threads to be used in
    operations.  Returns the previous setting for the number of
    threads.  During initialization time Numexpr sets this number to
    the number of detected cores in the system (see
    `detect_number_of_cores()`).

  * detect_number_of_cores(): Detects the number of virtual cores in 
    the system.



How Numexpr can achieve such a high performance?
================================================

The main reason why Numexpr achieves better performance than NumPy (or
even than plain C code) is that it avoids the creation of whole
temporaries for keeping intermediate results, so saving memory
bandwidth (the main bottleneck in many computations in nowadays
computers). Due to this, it works best with arrays that are large
enough (typically larger than processor caches).

Briefly, it works as follows. Numexpr parses the expression into its
own op-codes, that will be used by the integrated computing virtual
machine. Then, the array operands are split in small blocks (that
easily fit in the cache of the CPU) and passed to the virtual
machine. Then, the computational phase starts, and the virtual machine
applies the op-code operations for each block, saving the outcome in
the resulting array. It is worth noting that all the temporaries and
constants in the expression are kept in the same small block sizes
than the operand ones, avoiding additional memory (and most specially,
memory bandwidth) waste.

Numexpr will perform better (in comparison with NumPy) with
larger matrices, i.e. typically those that does not fit in the cache
of your CPU. The break-even point for NumExpr3 with NumPy is generally 
with an array of 64k elements for simple operations such as 'a*b', on a
machine with a 32 kB L1 cache. For example, on 1M float-64 arrays, 
with 8 threads NE3 runs 'a*b+c' 600 % faster than NumPy and on
1M complex-64 arrays NE3 runs > 800 % faster than NumPy.

Also more complicated expressions that would require NumPy to make
full-size temporaries and re-acquire the GIL multiple times, such as 
'a*b-4.1*a > 2.5*b', can see additional large speed-ups.  NumExpr3
generally scales well in the 2-8 thread range. Scaling with threads 
in NE3 is slightly better than in NE2 (~ 10 %).

The speed-up from NE2 to NE3 due to vectorization on x64 processors 
depends heavily on the functions used. On AVX2 chipsets often 200 % 
speed-ups will be observed for operations that have been vectorized 
(e.g. +,-,*,/). The additional data types can also be used to 
accelerate computation. For example calculations with complex-64 can 
be 500 % faster than complex-128 calculations in NumExpr2.

See more info about how Numexpr works in:

https://github.com/pydata/numexpr/wiki


Authors
=======

See AUTHORS.txt


License
=======

Numexpr3 is distributed under the BSD license (see LICENSE.txt file).



.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 70
.. End:
