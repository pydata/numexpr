=========================
Announcing NumExpr 2.10.1
=========================

Hi everyone,

NumExpr 2.10.1 continues to establize the support for NumPy 2.0.0.
Also, the default number of 'safe' threads has been upgraded to 16
(instead of previous 8). Finally, preliminary support for Python 3.13;
thanks to Karolina Surma.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.9.0 to 2.10.0
----------------------------

* Support for NumPy 2.0.0.  This is still experimental, so please
  report any issues you find.  Thanks to Clément Robert and Thomas
  Caswell for the work.

* Avoid erroring when OMP_NUM_THREADS is empty string.  Thanks to
  Patrick Hoefler.

* Do not warn if OMP_NUM_THREAD set.

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
