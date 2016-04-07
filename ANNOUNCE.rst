=========================
 Announcing Numexpr 2.5.2
=========================

Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.

It wears multi-threaded capabilities, as well as support for Intel's
MKL (Math Kernel Library), which allows an extremely fast evaluation
of transcendental functions (sin, cos, tan, exp, log...) while
squeezing the last drop of performance out of your multi-core
processors.  Look here for a some benchmarks of numexpr using MKL:

https://github.com/pydata/numexpr/wiki/NumexprMKL

Its only dependency is NumPy (MKL is optional), so it works well as an
easy-to-deploy, easy-to-use, computational engine for projects that
don't want to adopt other solutions requiring more heavy dependencies.

What's new
==========

This is a maintenance release shaking some remaining problems with VML
(it is nice to see how Anaconda VML's support helps raising hidden
issues).  Now conj() and abs() are actually added as VML-powered
functions, preventing the same problems than log10() before (PR #212);
thanks to Tom Kooij.  Upgrading to this release is highly recommended.

In case you want to know more in detail what has changed in this
version, see:

https://github.com/pydata/numexpr/blob/master/RELEASE_NOTES.rst

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
