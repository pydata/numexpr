========================
Announcing NumExpr 2.8.2
========================

Hi everyone, 

Please find here another maintenance release of NumExpr. Wheels for ARM64 
multilinux should be available again and Apple Silicon wheels are also now 
available on PyPi.

Project documentation is available at:

http://numexpr.readthedocs.io/


Changes from 2.8.0 to 2.8.1
---------------------------

* Thanks to Matt Einhorn for improvements to the GitHub Actions build process to
  add support for Apple Silicon and aarch64.
* Thanks to Biswapriyo Nath for a fix to allow `mingw` builds on Windows.
* Due to the removal of the array flag `NPY_ARRAY_UPDATEIFCOPY`, it's possible for
  older versions of NumExpr (<= 2.8.1) to fail to compile against NumPy >= 1.23.0.
  This flag was removed with no observed change in behavior. The branch of code 
  effected could only be reached with a statement such as:

```
      x=np.zeros(1); 
      ne.evaluate('3', out=x)
```

* There have been some changes made to not import `platform.machine()` on `sparc`
  but it is highly advised to upgrade to Python 3.9+ to avoid this issue with 
  the `platform` package.

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
