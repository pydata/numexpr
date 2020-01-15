========================
Announcing Numexpr 2.7.1
========================

Hi everyone, 

This is a version bump to add support for Python 3.8 and NumPy 1.18. We are also 
removing support for Python 3.4.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.7.0 to 2.7.1
----------------------------

- Python 3.8 support has been added.
- Python 3.4 support is discontinued.
- The tests are now compatible with NumPy 1.18.
- `site.cfg.example` was updated to use the `libraries` tag instead of `mkl_libs`,
  which is recommended for newer version of NumPy.

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
