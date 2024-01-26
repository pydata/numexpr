========================
Announcing NumExpr 2.9.0
========================

Hi everyone,

NumExpr 2.9.0 is a release offering support for latest versions of PyPy.
The full test suite should pass now, at least for the Python 3.10 version.
Thanks to @27rabbitlt for most of the work and @mgorny and @mattip for
providing help and additional fixes.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.8.7 to 2.8.8
---------------------------

* Fix re_evaluate not taking global_dict as argument. Thanks to Teng Liu
  (@27rabbitlt).

* Fix parsing of simple complex numbers.  Now, `ne.evaluate('1.5j')` works.
  Thanks to Teng Liu (@27rabbitlt).

* Fixes for upcoming NumPy 2.0:

  * Replace npy_cdouble with C++ complex. Thanks to Teng Liu (@27rabbitlt).
  * Add NE_MAXARGS for future numpy change NPY_MAXARGS. Now it is set to 64
    to match NumPy 2.0 value. Thanks to Teng Liu (@27rabbitlt).

What's Numexpr?
---------------

Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.

It has multi-threaded capabilities, as well as support for Intel's
MKL (Math Kernel Library), which allows an extremely fast evaluation
of transcendental functions (sin, cos, tan, exp, log...) while
squeezing the last drop of performance out of your multi-core
processors.  Look here for a some benchmarks of numexpr using MKL:

https://github.com/pydata/numexpr/wiki/NumexprMKL

Its only dependency is NumPy (MKL is optional), so it works well as an
easy-to-deploy, easy-to-use, computational engine for projects that
don't want to adopt other solutions requiring more heavy dependencies.

Where I can find Numexpr?
-------------------------

The project is hosted at GitHub in:

https://github.com/pydata/numexpr

You can get the packages from PyPI as well (but not for RC releases):

http://pypi.python.org/pypi/numexpr

Documentation is hosted at:

http://numexpr.readthedocs.io/en/latest/

Share your experience
---------------------

Let us know of any bugs, suggestions, gripes, kudos, etc. you may
have.

Enjoy data!
