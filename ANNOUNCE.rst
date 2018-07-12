==========================
 Announcing Numexpr 2.6.6
==========================

Hi everyone, 

This is a bug-fix release. Thanks to Mark Dickinson for a fix to the thread 
barrier that occassionally suffered from spurious wakeups on MacOSX.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.6.4 to 2.6.5
---------------------------

- The maximum thread count can now be set at import-time by setting the 
  environment variable 'NUMEXPR_MAX_THREADS'. The default number of 
  max threads was lowered from 4096 (which was deemed excessive) to 64.
- A number of imports were removed (pkg_resources) or made lazy (cpuinfo) in 
  order to speed load-times for downstream packages (such as `pandas`, `sympy`, 
  and `tables`). Import time has dropped from about 330 ms to 90 ms. Thanks to 
  Jason Sachs for pointing out the source of the slow-down.
- Thanks to Alvaro Lopez Ortega for updates to benchmarks to be compatible with 
  Python 3.
- Travis and AppVeyor now fail if the test module fails or errors.
- Thanks to Mahdi Ben Jelloul for a patch that removed a bug where constants 
  in `where` calls would raise a ValueError.
- Fixed a bug whereby all-constant power operations would lead to infinite 
  recursion.

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
