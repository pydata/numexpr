========================
Announcing NumExpr 2.8.4
========================

Hi everyone, 

This is a maintenance and bug-fix release for NumExpr. In particular, now we have 
added Python 3.11 support. 

Project documentation is available at:

http://numexpr.readthedocs.io/


Changes from 2.8.3 to 2.8.4
---------------------------

* Support for Python 3.11 has been added.
* Thanks to Tobias Hangleiter for an improved accuracy complex `expm1` function.
  While it is 25 % slower, it is significantly more accurate for the real component
  over a range of values and matches NumPy outputs much more closely.
* Thanks to Kirill Kouzoubov for a range of fixes to constants parsing that was 
  resulting in duplicated constants of the same value.
* Thanks to Mark Harfouche for noticing that we no longer need `numpy` version 
  checks. `packaging` is no longer a requirement as a result.


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
