Hi everyone, 

This is primarily a maintenance release that fixes a number of newly discovered
bugs. The NumPy requirement has increased from 1.6 to 1.7 due to changes with 
`numpy.nditer` flags. Thanks to Caleb P. Burns `ceil` and `floor` functions are 
now supported.

Project documentation is now available at:

http://numexpr.readthedocs.io/

P.S. due to seg-faults occuring for MKL with ceil and floor we have pushed a 
quick patch for 2.6.3 to 2.6.4. Thanks to Christoph Gohkle for the fixes.

==========================
 Announcing Numexpr 2.6.4
==========================

Changes from 2.6.3 to 2.6.4
---------------------------

- Christoph Gohlke noticed a lack of coverage for the 2.6.3 
  `floor` and `ceil` functions for MKL that caused seg-faults in 
  test, so thanks to him for that.

Changes from 2.6.2 to 2.6.3
---------------------------

- Documentation now available at numexpr.readthedocs.io
- Support for floor() and ceil() functions added by Caleb P. Burns.
- NumPy requirement increased from 1.6 to 1.7 due to changes in iterator
  flags (#245).
- Sphinx autodocs support added for documentation on readthedocs.org.
- Fixed a bug where complex constants would return an error, fixing 
  problems with `sympy` when using NumExpr as a backend.
- Fix for #277 whereby arrays of shape (1,...) would be reduced as 
  if they were full reduction. Behavoir now matches that of NumPy.
- String literals are automatically encoded into 'ascii' bytes for 
  convience (see #281).

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
