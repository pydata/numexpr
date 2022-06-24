========================
Announcing NumExpr 2.8.2
========================

Hi everyone, 

Please find here another maintenance release of NumExpr. Support for Python 3.6 
has been dropped to enable support for NumPy 1.23 (and by extension Python 3.11 
when it is released). Wheels for ARM64 multilinux should be available again after 
troubles with GitHub Actions and Apple Silicon wheels are also now available on 
PyPi for download.

Project documentation is available at:

http://numexpr.readthedocs.io/


Changes from 2.8.1 to 2.8.2
---------------------------

* Support for Python 3.6 has been dropped due to the need to substitute the flag 
  `NPY_ARRAY_WRITEBACKIFCOPY` for `NPY_ARRAY_UPDATEIFCOPY`. This flag change was 
  initiated in NumPy 1.14 and finalized in 1.23. The only changes were made to 
  cases where an unaligned constant was passed in with a pre-allocated output 
  variable:

```
    x = np.empty(5, dtype=np.uint8)[1:].view(np.int32)
    ne.evaluate('3', out=x)
```

  We think the risk of issues is very low, but if you are using NumExpr as a 
  expression evaluation tool you may want to write a test for this edge case.
* Thanks to Matt Einhorn (@matham) for improvements to the GitHub Actions build process to
  add support for Apple Silicon and aarch64.
* Thanks to Biswapriyo Nath (@biswa96) for a fix to allow `mingw` builds on Windows.
* There have been some changes made to not import `platform.machine()` on `sparc`
  but it is highly advised to upgrade to Python 3.9+ to avoid this issue with 
  the Python core package `platform`.
  
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
