======================================================
Numexpr: Fast numerical expression evaluator for NumPy
======================================================

:Author: David M. Cooke, Francesc Alted and others
:Contact: faltet@gmail.com
:URL: https://github.com/pydata/numexpr
:Documentation: http://numexpr.readthedocs.io/en/latest/
:Travis CI: |travis|
:Appveyor: |appveyor|
:PyPi: |version| |pypi|
:readthedocs: |docs|

.. |travis| image:: https://travis-ci.org/pydata/numexpr.png?branch=master
        :target: https://travis-ci.org/pydata/numexpr
.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/pydata/numexpr
        :target: https://ci.appveyor.com/project/pydata/numexpr
.. |pypi| image:: https://img.shields.io/pypi/dm/numexpr.png
        :target: https://pypi.python.org/pypi/numexpr
.. |docs| image:: https://media.readthedocs.org/static/projects/badges/passing.svg
        :target: http://numexpr.readthedocs.io/en/latest
.. |version| image:: https://img.shields.io/pypi/v/numexpr.png
        :target: https://pypi.python.org/pypi/numexpr


What it is Numexpr?
-------------------

Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like :code:`'3*a+4*b'`) are accelerated
and use less memory than doing the same calculation in Python.

In addition, its multi-threaded capabilities can make use of all your
cores -- which generally results in substantial performance scaling compared
to NumPy.

Last but not least, numexpr can make use of Intel's VML (Vector Math
Library, normally integrated in its Math Kernel Library, or MKL).
This allows further acceleration of transcendent expressions.


How Numexpr achieves high performance
-------------------------------------

The main reason why Numexpr achieves better performance than NumPy is
that it avoids allocating memory for intermediate results. This
results in better cache utilization and reduces memory access in
general. Due to this, Numexpr works best with large arrays.

Numexpr parses expressions into its own op-codes that are then used by
an integrated computing virtual machine. The array operands are split
into small chunks that easily fit in the cache of the CPU and passed
to the virtual machine. The virtual machine then applies the
operations on each chunk. It's worth noting that all temporaries and
constants in the expression are also chunked. Chunks are distributed among 
the available cores of the CPU, resulting in highly parallelized code 
execution.

The result is that Numexpr can get the most of your machine computing
capabilities for array-wise computations. Common speed-ups with regard
to NumPy are usually between 0.95x (for very simple expressions like
:code:`'a + 1'`) and 4x (for relatively complex ones like :code:`'a*b-4.1*a >
2.5*b'`), although much higher speed-ups can be achieved for some functions 
and complex math operations (up to 15x in some cases).

Numexpr performs best on matrices that are too large to fit in L1 CPU cache. 
In order to get a better idea on the different speed-ups that can be achieved 
on your platform, run the provided benchmarks.


Examples of use
---------------

::

  >>> import numpy as np
  >>> import numexpr as ne

  >>> a = np.arange(1e6)   # Choose large arrays for better speedups
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


Documentation
-------------

Please see the official documentation at `numexpr.readthedocs.io <https://numexpr.readthedocs.io>`_.
Included is a user guide, benchmark results, and the reference API.


Authors
-------

.. include:: AUTHORS.TXT


License
-------

NumExpr is distributed under the MIT_ license.

.. _MIT: http://www.opensource.org/licenses/mit-license.php


.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 70
.. End:
