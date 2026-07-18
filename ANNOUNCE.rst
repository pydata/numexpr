=========================
Announcing NumExpr 2.14.2
=========================

Hi everyone,

NumExpr 2.14.2 is a maintenance release with several bug fixes, a new
``disable_cache`` option for ``evaluate()``, and updated build/CI support
(Windows ARM64 wheels, dropped Python 3.10, no more free-threaded 3.13
wheels).

Project documentation is available at:

https://numexpr.readthedocs.io/

Changes from 2.14.1 to 2.14.2
-----------------------------

* Added a ``disable_cache`` parameter to ``evaluate()`` to bypass the
  internal expression cache. Thanks to 27rabbitlt.
* Added Windows ARM64 wheel builds.
* Dropped support for Python 3.10.
* No longer build free-threaded Python 3.13 wheels, matching NumPy's own
  support.
* Avoid keeping arrays passed as ``out=`` alive in the ``re_evaluate`` cache
  (#558).
* Guarded out-of-range shift counts (shift amount >= bit width) in the
  integer ``<<``/``>>`` opcodes, which was undefined behavior in C and could
  return garbage results. Thanks to uwezkhan (#559).
* Fixed ``run_interpreter()`` unconditionally returning success even when
  the VM engine failed, so execution errors are now correctly raised
  instead of silently discarded (#557).
* Fixed a reference leak of ``constsig`` on the allocation-failure path in
  ``NumExpr_init()`` (#561).

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
