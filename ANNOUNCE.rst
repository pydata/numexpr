========================
Announcing NumExpr 2.8.5
========================

Hi everyone, 

In 2.8.5 we have added a new function, `validate` which checks an expression `ex`
for validity, for usage where the program is parsing a user input. There are also 
consequences for this sort of usage, since `eval(ex)` is called, and as such we 
do some string sanitization as described below.

Project documentation is available at:

http://numexpr.readthedocs.io/

Changes from 2.8.4 to 2.8.5
---------------------------

* A `validate` function has been added. This function checks the inputs, returning 
  `None` on success or raising an exception on invalid inputs. This function was 
  added as numerous projects seem to be using NumExpr for parsing user inputs.
  `re_evaluate` may be called directly following `validate`.
* As an addendum to the use of NumExpr for parsing user inputs, is that NumExpr
  calls `eval` on the inputs. A regular expression is now applied to help sanitize 
  the input expression string, forbidding '__', ':', and ';'. Attribute access 
  is also banned except for '.r' for real and '.i'  for imag.
* Thanks to timbrist for a fix to behavior of NumExpr with integers to negative 
  powers. NumExpr was pre-checking integer powers for negative values, which 
  was both inefficient and causing parsing errors in some situations. Now NumExpr
  will simply return 0 as a result for such cases. While NumExpr generally tries 
  to follow NumPy behavior, performance is also critical. 
* Thanks to peadar for some fixes to how NumExpr launches threads for embedded 
  applications.
* Thanks to de11n for making parsing of the `site.cfg` for MKL consistent among 
  all shared platforms.


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
