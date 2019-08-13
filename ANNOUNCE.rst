=========================
 Announcing Numexpr 2.7.0
=========================

Hi everyone, 

This is a minor version bump for NumExpr. We would like to highlight the changes
made in 2.6.9 (which in retrospec should have been a minor version bump), where
the maximum number of threads spawned can be limited by setting the environment 
variable "NUMEXPR_MAX_THREADS". If this variable is not set, in 2.7.0 the 
historical limit of 8 threads will be used. The lack of a check caused some 
problems on very large hosts in cluster environments in 2.6.9.  

In addition, we are officially dropping Python 2.6 support in this release as 
we cannot perform continuous integration for it.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.6.9 to 2.7.0
----------------------------

- The default number of 'safe' threads has been restored to the historical limit 
  of 8, if the environment variable "NUMEXPR_MAX_THREADS" has not been set.
- Thanks to @eltoder who fixed a small memory leak.
- Support for Python 2.6 has been dropped, as it is no longer available via 
  TravisCI.
- A typo in the test suite that had a less than rather than greater than symbol 
  in the NumPy version check has been corrected thanks to dhomeier.
- The file `site.cfg` was being accidently included in the sdists on PyPi. 
  It has now been excluded.

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
