What it is Numexpr?
===================

Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory (and very important, less memory *bandwith*) than
doing the same calculation in Python.


Examples of use
===============

>>> import numpy as np
>>> import numexpr as ne
>>> a = np.arange(1e6)
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
>>> ne.evaluate("'abba' == s")   # string arrays are suported too
array([ True, False, False], dtype=bool)


How it works?
=============

Briefly, Numexpr parses the expression into its own op-codes, that
will be used in its computing kernel.  Then, the kernel decomposes the
arrays operands in small chunks that easily fit in the cache of the
CPU, and then applies the op-code operations for these, saving each
portion in the resulting array.  It is worth noting that all the
temporaries or constants are kept in chunks of the same size than the
operand ones, so avoiding additional memory (and memory bandwith)
consumption.

The result is that Numexpr can get the most of your machine computing
capabilities for array-wise computations.  Just to give you an idea of
its performance, common speed-ups are usually between 0.95x (for very
simple expressions, like ’a + 1’) and 4x (for relatively complex ones,
like 'a*b-4.1*a > 2.5*b'), although much higher speed-ups can be
achieved (up to 15x can be seen in not too esoteric expressions)
because this depends on the kind of the operations and how many
operands participates in the expression.  In order ot get a better
idea on the different speed-ups for your own platform, you may want to
run the benchmarks in the directory bench/.


Authors
=======

Numexpr is written by David Cooke <david.m.cooke@gmail.com> and Tim
Hochberg <tim.hochberg@ieee.org>. Francesc Alted <faltet@pytables.org>
contributed support for booleans and for efficient strided and
unaligned array operations. Ivan Vilata <ivilata@selidor.net>
contributed support for strings.  It is distributed under the MIT
license (see LICENSE file).



.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 70
.. End:
