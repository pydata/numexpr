=========================
Announcing NumExpr 2.10.1
=========================

Hi everyone,

NumExpr 2.10.1 continues to stabilize the support for NumPy 2.0.0.
Also, the default number of 'safe' threads has been upgraded to 16
(instead of previous 8). Finally, preliminary support for Python 3.13;
thanks to Karolina Surma.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.10.0 to 2.10.1
-----------------------------

* The default number of 'safe' threads has been upgraded to 16 (instead of
  previous 8). That means that if your CPU has > 16 cores, the default is
  to use 16. You can always override this with the "NUMEXPR_MAX_THREADS"
  environment variable.

* NumPy 1.23 is now the minimum supported.

* Preliminary support for Python 3.13. Thanks to Karolina Surma.

* Fix tests on nthreads detection (closes: #479). Thanks to @avalentino.

* The build process has been modernized and now uses the `pyproject.toml`
  file for more of the configuration options.

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
