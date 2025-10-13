=========================
Announcing NumExpr 2.14.
=========================

Hi everyone,

NumExpr 2.14.0 introduces a couple of patches for tan / tanh and
adds static typing support.
Thanks to Luke Shaw and Joren Hammudoglu (@jorenham) for these contributions.

Project documentation is available at:

https://numexpr.readthedocs.io/

Changes from 2.13.1 to 2.14.0
-----------------------------

* Numerical stability for overflow has been improved for ``tan`` and ``tanh``
  to handle possible overflows for complex numbers.

* Static typing support has been added, making NumExpr compatible with
  static type checkers like `mypy` and `pyright`.
  Thanks to Joren Hammudoglu (@jorenham) for the work.


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
