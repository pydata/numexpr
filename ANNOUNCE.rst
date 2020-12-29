========================
Announcing NumExpr 2.7.2
========================

Hi everyone, 

It's been awhile since the last update to NumExpr, mostly as the existing scientific 
Python tool chain for building wheels on PyPi became defunct and we have had to 
redevelop a new one based on `cibuildwheel` and GitHub Actions. This release also
brings us support (and wheels for) Python 3.9.

There have been a number of changes to enhance how NumExpr works when NumPy 
uses MKL as a backend.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.7.1 to 2.7.2
---------------------------

- Support for Python 2.7 and 3.5 is deprecated and will be discontinued when 
  `cibuildwheels` and/or GitHub Actions no longer support these versions.
- Wheels are now provided for Python 3.7, 3.5, 3.6, 3.7, 3.8, and 3.9 via 
  GitHub Actions.
- The block size is now exported into the namespace as `numexpr.__BLOCK_SIZE1__`
  as a read-only value.
- If using MKL, the number of threads for VML is no longer forced to 1 on loading 
  the module. Testing has shown that VML never runs in multi-threaded mode for 
  the default BLOCKSIZE1 of 1024 elements, and forcing to 1 can have deleterious 
  effects on NumPy functions when built with MKL. See issue #355 for details.
- Use of `ndarray.tostring()` in tests has been switch to `ndarray.tobytes()` 
  for future-proofing deprecation of `.tostring()`, if the version of NumPy is 
  greater than 1.9.
- Added a utility method `get_num_threads` that returns the (maximum) number of 
  threads currently in use by the virtual machine. The functionality of 
  `set_num_threads` whereby it returns the previous value has been deprecated 
  and will be removed in 2.8.X.

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
