========================
Announcing NumExpr 2.8.7
========================

Hi everyone,

NumExpr 2.8.7 is a release to deal with issues related to downstream `pandas`
and other projects where the sanitization blacklist was triggering issue in their
evaluate. Hopefully, the new sanitization code would be much more robust now.

For those who do not wish to have sanitization on by default, it can be changed
by setting an environment variable, `NUMEXPR_SANITIZE=0`.

If you use `pandas` in your packages it is advisable you pin

`numexpr >= 2.8.7`

in your requirements.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.8.5 to 2.8.6
---------------------------

* More permissive rules in sanitizing regular expression: allow to access digits
  after the . with scientific notation.  Thanks to Thomas Vincent.

* Don't reject double underscores that are not at the start or end of a variable
  name (pandas uses those), or scientific-notation numbers with digits after the
  decimal point.  Thanks to Rebecca Palmer.

* Do not use `numpy.alltrue` in the test suite, as it has been deprecated
  (replaced by `numpy.all`).  Thanks to Rebecca Chen.

* Wheels for Python 3.12.  Wheels for 3.7 and 3.8 are not generated anymore.

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
