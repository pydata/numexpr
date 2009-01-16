What it is Numexpr?
===================

Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.


Examples of use
===============

>>> import numpy as np
>>> import numexpr as ne

>>> a = np.arange(1e6)   # Choose large arrays
>>> b = np.arange(1e6)

>>> ne.evaluate("a + 1")   # a simple expression
array([  1.00000000e+00,   2.00000000e+00,   3.00000000e+00, ...,
         9.99998000e+05,   9.99999000e+05,   1.00000000e+06])

>>> ne.evaluate('a*b-4.1*a > 2.5*b')   # a more complex one
array([False, False, False, ...,  True,  True,  True], dtype=bool)

>>> ne.evaluate("sin(a) + arcsinh(a/b)")   # you can also use functions
array([        NaN,  1.72284457,  1.79067101, ...,  1.09567006,
        0.17523598, -0.09597844])

>>> s = np.array(['abba', 'abbb', 'abbcdef'])
>>> ne.evaluate("'abba' == s")   # string arrays are supported too
array([ True, False, False], dtype=bool)


Datatypes supported internally
==============================

Numexpr operates internally only with the following types:

    * 8-bit boolean (bool)
    * 32-bit signed integer (int)
    * 64-bit signed integer (long)
    * 64-bit, double-precision floating point number (float)
    * 2x64-bit, double-precision complex number (complex)
    * Raw string of bytes (str)

If the arrays in the expression does not match any of these types,
they will be upcasted to one of the above types (following the usual
type inference rules).  Have this in mind when doing estimations about
the memory consumption during the computation of your expressions.

Also, the types in Numexpr conditions are somewhat stricter than those
of Python.  For instance, the only valid constants for booleans are
`True` and `False`, and they are never automatically cast to integers.


Supported Operators
===================

Numexpr supports the set of operators listed below:

    * Logical operators: &, |, ~
    * Comparison operators: <, <=, ==, !=, >=, >
    * Unary arithmetic operators: -
    * Binary arithmetic operators: +, -, *, /, **, %


Supported functions
===================

The next are the current supported set:

    * where(bool, number1, number2): number
        Number1 if the bool condition is true, number2 otherwise.
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
    * {exp,expm1}(float|complex): float|complex
        Exponential and exponential minus one.
    * sqrt(float|complex): float|complex
        Square root.
    * {real,imag}(complex): float
        Real or imaginary part of complex.
    * complex(float, float): complex
        Complex from real and imaginary parts.

More functions can be added if you need them.


How Numexpr can achieve such a high performance?
================================================

The main reason why it achieves better performance than NumPy (or
plain C code) is that it avoids the creation of complete temporaries
for keeping intermediate results, so saving memory bandwidth (the main
bottleneck in many computations in nowadays computers).  Due to this,
it works best with arrays that are large enough (typically larger than
processor caches).

Briefly, it works as follows.  Numexpr parses the expression into its
own op-codes, that will be used in its computing virtual machine.
Then, the array operands are splitted in small chunks (that easily fit
in the cache of the CPU) and passed to the virtual machine.  Then, the
computational phase starts, and the virtual machine applies the
op-code operations for each chunk, saving the outcome in the resulting
array.  It is worth noting that all the temporaries and constants in
the expression are kept in chunks of the same size than the operand
ones, avoiding additional memory (and most specially, memory bandwidth)
consumption.

The result is that Numexpr can get the most of your machine computing
capabilities for array-wise computations.  Just to give you an idea of
its performance, common speed-ups with regard to NumPy are usually
between 0.95x (for very simple expressions, like ’a + 1’) and 4x (for
relatively complex ones, like 'a*b-4.1*a > 2.5*b'), although much
higher speed-ups can be achieved (up to 15x can be seen in not too
esoteric expressions) because this depends on the kind of the
operations and how many operands participates in the expression.  Of
course, Numexpr will perform better (in comparison with NumPy) with
larger matrices, i.e. typically those that does not fit in the cache
of your CPU.  In order to get a better idea on the different speed-ups
that can be achieved for your own platform, you may want to run the
benchmarks in the directory bench/.


Authors
=======

Numexpr is written by David Cooke <david.m.cooke@gmail.com> and Tim
Hochberg <tim.hochberg@ieee.org>.  Francesc Alted
<faltet@pytables.org> contributed support for booleans and for
efficient strided and unaligned array operations.  Ivan Vilata
<ivilata@selidor.net> contributed support for strings.  It is
distributed under the MIT license (see LICENSE.txt file).



.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 70
.. End:
