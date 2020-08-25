========================
Announcing Numexpr 2.7.2
========================

Hi everyone, 

This is a change in behavior for the Intel VML (Vector Math Library) version, in
that it no longer forces the VML threads to 1, as this could have a negative 
effect on the performance of NumPy itself when it is built with MKL.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.7.1 to 2.7.2
---------------------------

- The block size is now exported into the namespace as `numexpr.__BLOCK_SIZE1__`
  as a read-only value.
- If using MKL, the number of threads for VML is no longer forced to 1 on loading 
  the module. Testing has shown that VML never runs in multi-threaded mode for 
  the default BLOCKSIZE1 of 1024 elements, and forcing to 1 can have deleterious 
  effects on NumPy functions when built with MKL. See issue #355 for details.

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
