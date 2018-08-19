==========================
 Announcing Numexpr 2.6.8
==========================

Hi everyone, 

#XXX version-specific blurb XXX#

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.6.7 to 2.6.8
---------------------------

- Add check to make sure that `f_locals` is not actually `f_globals` when we 
  do the `f_locals` clear to avoid the #310 memory leak issue.
- Compare NumPy versions using `distutils.version.LooseVersion` to avoid issue
  #312 when working with NumPy development versions.
- As part of `multibuild`, wheels for Python 3.7 for Linux and MacOSX are now 
  available on PyPI.

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


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
