=========================
 Announcing Numexpr 2.6.0
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

This is a minor version bump because it introduces a new function.
Also some minor fine tuning for recent CPUs has been done:

- Introduced a new re_evaluate() function for re-evaluating the
  previous executed array expression without any check.  This is meant
  for accelerating loops that are re-evaluating the same expression
  repeatedly without changing anything else than the operands.  If
  unsure, use evaluate() which is safer.

- The BLOCK_SIZE1 and BLOCK_SIZE2 constants have been re-checked in
  order to find a value maximizing most of the benchmarks in bench/
  directory.  The new values (8192 and 16 respectively) give somewhat
  better results (~5%) overall.  The CPU used for fine tuning is a
  relatively new Haswell processor (E3-1240 v3).

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
