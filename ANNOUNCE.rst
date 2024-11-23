=========================
Announcing NumExpr 2.10.2
=========================

Hi everyone,

NumExpr 2.10.2 provides wheels for Python 2.13 for first time.
Also, there is better support for CPUs that do not have a power
of 2 number of cores.  Finally, numexpr is allowed to run with
the multithreading package in Python.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.10.1 to 2.10.2
-----------------------------

* Better support for CPUs that do not have a power of 2 number of
  cores.  See #479 and #490.  Thanks to @avalentino.

* Allow numexpr to run with the multithreading package in Python.
  See PR #496.  Thanks to @emmaai

* Wheels for Python 3.13 are now provided.

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
