=========================
Announcing NumExpr 2.11.0
=========================

Hi everyone,

NumExpr 2.11.0 Initial support for free-threaded Python 3.13t has been added.
This is still experimental, so please report any issues you find.
Finally, Python 3.10 is now the minimum supported version.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.10.2 to 2.11.0
-----------------------------

* Initial support for free-threaded Python 3.13t has been added.
  This is still experimental, so please report any issues you find.
  For more info, see discussions PRs #504, #505 and #508.
  Thanks to @andfoy, @rgommers and @FrancescAlted for the work.

* Fix imaginary evaluation in the form of `1.1e1j`.  This was
  previously not supported and would raise an error.  Thanks to @27rabbitlt
  for the fix.

* The test suite has been modernized to use `pytest` instead of `unittest`.
  This should make it easier to run the tests and contribute to the project.

* Python 3.10 is now the minimum supported version.

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
