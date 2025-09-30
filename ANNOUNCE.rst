=========================
Announcing NumExpr 2.13.0
=========================

Hi everyone,

NumExpr 2.13.0 introduced a bunch of new features including new
bitwise operators (&, |, ^, ~), floor division (//). It also adds
many new functions (like hypot, log2, maximum, minimum, nextafter...).
Thanks to Luke Shaw for these contributions.

Project documentation is available at:

https://numexpr.readthedocs.io/

Changes from 2.12.1 to 2.13.0
-----------------------------

* New functionality has been added:
  * Bitwise operators (and, or, not, xor): `&, |, ~, ^`
  * New binary arithmetic operator for floor division: `//`
  * New functions: `signbit`, `hypot`, `copysign`, `nextafter`, `maximum`,
    `minimum`, `log2`, `trunc`, `round` and `sign`.
  * Also enables integer outputs for integer inputs for
    `abs`, `fmod`, `copy`, `ones_like`, `sign` and `round`.

  Thanks to Luke Shaw for the contributions.

* New wheels for Python 3.14 and 3.14t are provided.

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
