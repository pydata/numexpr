==========================
 Announcing Numexpr 2.6.7
==========================

Hi everyone, 

This is a bug-fix release. Thanks to Lehman Garrison for a fix that could 
result in memory leak-like behavior.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.6.6 to 2.6.7
---------------------------

- Thanks to Lehman Garrison for finding and fixing a bug that exhibited memory
  leak-like behavior. The use in `numexpr.evaluate` of `sys._getframe` combined 
  with `.f_locals` from that frame object results an extra refcount on objects 
  in the frame that calls `numexpr.evaluate`, and not `evaluate`'s frame. So if 
  the calling frame remains in scope for a long time (such as a procedural 
  script where `numexpr` is called from the base frame) garbage collection would 
  never occur.
- Imports for the `numexpr.test` submodule were made lazy in the `numexpr` module.

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
