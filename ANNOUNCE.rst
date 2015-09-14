=========================
 Announcing Numexpr 2.4.4
=========================

Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.

It wears multi-threaded capabilities, as well as support for Intel's
MKL (Math Kernel Library), which allows an extremely fast evaluation
of transcendental functions (sin, cos, tan, exp, log...)  while
squeezing the last drop of performance out of your multi-core
processors.  Look here for a some benchmarks of numexpr using MKL:

https://github.com/pydata/numexpr/wiki/NumexprMKL

Its only dependency is NumPy (MKL is optional), so it works well as an
easy-to-deploy, easy-to-use, computational engine for projects that
don't want to adopt other solutions requiring more heavy dependencies.

What's new
==========

This is a maintenance release which contains several bug fixes, like
better testing on Python3 platform and some harmless data race.  Among
the enhancements, AppVeyor support is here and OMP_NUM_THREADS is
honored as a fallback in case NUMEXPR_NUM_THREADS is not set.

In case you want to know more in detail what has changed in this
version, see:

https://github.com/pydata/numexpr/blob/master/RELEASE_NOTES.rst

or have a look at RELEASE_NOTES.txt in the tarball.

Where I can find Numexpr?
=========================

The project is hosted at GitHub in:

https://github.com/pydata/numexpr

You can get the packages from PyPI as well (but not for RC releases):

http://pypi.python.org/pypi/numexpr

Share your experience
=====================

Let us know of any bugs, suggestions, gripes, kudos, etc. you may
have.


Enjoy data!


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
