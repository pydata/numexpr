======================================
 Release notes for Numexpr 2.6 series
======================================

Changes from 2.6.0 to 2.6.1
===========================

- Fixed a performance regression in some situations as consequence of
  increasing too much the BLOCK_SIZE1 constant.  After more careful
  benchmarks (both in VML and non-VML modes), the value has been set
  again to 1024 (down from 8192).  The benchmarks have been made with
  a relatively new processor (Intel Xeon E3-1245 v5 @ 3.50GHz), so
  they should work well for a good range of processors again.

- Added NetBSD support to CPU detection.  Thanks to Thomas Klausner.


Changes from 2.5.2 to 2.6.0
===========================

- Introduced a new re_evaluate() function for re-evaluating the
  previous executed array expression without any check.  This is meant
  for accelerating loops that are re-evaluating the same expression
  repeatedly without changing anything else than the operands.  If
  unsure, use evaluate() which is safer.

- The BLOCK_SIZE1 and BLOCK_SIZE2 constants have been re-checked in
  order to find a value maximizing most of the benchmarks in bench/
  directory.  The new values (8192 and 16 respectively) give somewhat
  better results (~5%) overall.  The CPU used for fine tuning is a
  relatively new Haswell processor (E3-1240 v3).

- The '--name' flag for `setup.py` returning the name of the package
  is honored now (issue #215).


Changes from 2.5.1 to 2.5.2
===========================

- conj() and abs() actually added as VML-powered functions, preventing
  the same problems than log10() before (PR #212).  Thanks to Tom Kooij
  for the fix!


Changes from 2.5 to 2.5.1
=========================

- Fix for log10() and conj() functions.  These produced wrong results
  when numexpr was compiled with Intel's MKL (which is a popular build
  since Anaconda ships it by default) and non-contiguous data (issue
  #210).  Thanks to Arne de Laat and Tom Kooij for reporting and
  providing a nice test unit.

- Fix that allows numexpr-powered apps to be profiled with pympler.
  Thanks to @nbecker.


Changes from 2.4.6 to 2.5
=========================

- Added locking for allowing the use of numexpr in multi-threaded
  callers (this does not prevent numexpr to use multiple cores
  simultaneously).  (PR #199, Antoine Pitrou, PR #200, Jenn Olsen).

- Added new min() and max() functions (PR #195, CJ Carey).


Changes from 2.4.5 to 2.4.6
===========================

- Fixed some UserWarnings in Solaris (PR #189, Graham Jones).

- Better handling of MSVC defines. (#168, Francesc Alted).


Changes from 2.4.4 to 2.4.5
===========================

- Undone a 'fix' for a harmless data race.  (#185 Benedikt Reinartz,
  Francesc Alted).

- Ignore NumPy warnings (overflow/underflow, divide by zero and
  others) that only show up in Python3.  Masking these warnings in
  tests is fine because all the results are checked to be
  valid. (#183, Francesc Alted).


Changes from 2.4.3 to 2.4.4
===========================

- Fix bad #ifdef for including stdint on Windows (PR #186, Mike Sarahan).


Changes from 2.4.3 to 2.4.4
===========================

* Honor OMP_NUM_THREADS as a fallback in case NUMEXPR_NUM_THREADS is not
  set. Fixes #161. (PR #175, Stefan Erb).

* Added support for AppVeyor (PR #178 Andrea Bedini)

* Fix to allow numexpr to be imported after eventlet.monkey_patch(),
  as suggested in #118 (PR #180 Ben Moran).

* Fix harmless data race that triggers false positives in ThreadSanitizer.
  (PR #179, Clement Courbet).

* Fixed some string tests on Python 3 (PR #182, Antonio Valentino).


Changes from 2.4.2 to 2.4.3
===========================

* Comparisons with empty strings work correctly now.  Fixes #121 and
  PyTables #184.

Changes from 2.4.1 to 2.4.2
===========================

* Improved setup.py so that pip can query the name and version without
  actually doing the installation.  Thanks to Joris Borgdorff.

Changes from 2.4 to 2.4.1
=========================

* Added more configuration examples for compiling with MKL/VML
  support.  Thanks to Davide Del Vento.

* Symbol MKL_VML changed into MKL_DOMAIN_VML because the former is
  deprecated in newer MKL.  Thanks to Nick Papior Andersen.

* Better determination of methods in `cpuinfo` module.  Thanks to Marc
  Jofre.

* Improved NumPy version determination (handy for 1.10.0).  Thanks
  to Åsmund Hjulstad.

* Benchmarks run now with both Python 2 and Python 3.  Thanks to Zoran
  Plesivčak.

Changes from 2.3.1 to 2.4
=========================

* A new `contains()` function has been added for detecting substrings
  in strings.  Only plain strings (bytes) are supported for now.  See
  PR #135 and ticket #142.  Thanks to Marcin Krol.

* New version of setup.py that allows better management of NumPy
  dependency.  See PR #133.  Thanks to Aleks Bunin.

Changes from 2.3 to 2.3.1
=========================

* Added support for shift-left (<<) and shift-right (>>) binary operators.
  See PR #131. Thanks to fish2000!

* Removed the rpath flag for the GCC linker, because it is probably
  not necessary and it chokes to clang.

Changes from 2.2.2 to 2.3
=========================

* Site has been migrated to https://github.com/pydata/numexpr.  All
  new tickets and PR should be directed there.

* [ENH] A `conj()` function for computing the conjugate of complex
  arrays has been added.  Thanks to David Menéndez.  See PR #125.

* [FIX] Fixed a DeprecationWarning derived of using oa_ndim == 0 and
  op_axes == NULL when using NpyIter_AdvancedNew() and NumPy 1.8.
  Thanks to Mark Wiebe for advise on how to fix this properly.

Changes from 2.2.1 to 2.2.2
===========================

* The `copy_args` argument of `NumExpr` function has been brought
  lack.  This has been mainly necessary for compatibility with
  `PyTables < 3.0`, which I decided to continue to support.  Fixed
  #115.

* The `__nonzero__` method in `ExpressionNode` class has been
  commented out.  This is also for compatibility with `PyTables <
  3.0`.  See #24 for details.

* Fixed the type of some parameters in the C extension so that s390
  architecture compiles.  Fixes #116.  Thank to Antonio Valentino for
  reporting and the patch.

Changes from 2.2 to 2.2.1
=========================

* Fixes a secondary effect of "from numpy.testing import `*`", where
  division is imported now too, so only then necessary functions from
  there are imported now.  Thanks to Christoph Gohlke for the patch.

Changes from 2.1 to 2.2
=======================

* [LICENSE] Fixed a problem with the license of the
  numexpr/win32/pthread.{c,h} files emulating pthreads on Windows
  platforms.  After persmission from the original authors is granted,
  these files adopt the MIT license and can be redistributed without
  problems.  See issue #109 for details
  (https://code.google.com/p/numexpr/issues/detail?id=110).

* [ENH] Improved the algorithm to decide the initial number of threads
  to be used.  This was necessary because by default, numexpr was
  using a number of threads equal to the detected number of cores, and
  this can be just too much for moder systems where this number can be
  too high (and counterporductive for performance in many cases).
  Now, the 'NUMEXPR_NUM_THREADS' environment variable is honored, and
  in case this is not present, a maximum number of *8* threads are
  setup initially.  The new algorithm is fully described in the Users
  Guide now in the note of 'General routines' section:
  https://code.google.com/p/numexpr/wiki/UsersGuide#General_routines.
  Closes #110.

* [ENH] numexpr.test() returns `TestResult` instead of None now.
  Closes #111.

* [FIX] Modulus with zero with integers no longer crashes the
  interpreter.  It nows puts a zero in the result.  Fixes #107.

* [API CLEAN] Removed `copy_args` argument of `evaluate`.  This should
  only be used by old versions of PyTables (< 3.0).

* [DOC] Documented the `optimization` and `truediv` flags of
  `evaluate` in Users Guide
  (https://code.google.com/p/numexpr/wiki/UsersGuide).

Changes from 2.0.1 to 2.1
===========================

* Dropped compatibility with Python < 2.6.

* Improve compatibiity with Python 3:

  - switch from PyString to PyBytes API (requires Python >= 2.6).
  - fixed incompatibilities regarding the int/long API
  - use the Py_TYPE macro
  - use the PyVarObject_HEAD_INIT macro instead of PyObject_HEAD_INIT

* Fixed several issues with different platforms not supporting
  multithreading or subprocess properly (see tickets #75 and #77).

* Now, when trying to use pure Python boolean operators, 'and',
  'or' and 'not', an error is issued suggesting that '&', '|' and
  '~' should be used instead (fixes #24).

Changes from 2.0 to 2.0.1
=========================

* Added compatibility with Python 2.5 (2.4 is definitely not supported
  anymore).

* `numexpr.evaluate` is fully documented now, in particular the new
  `out`, `order` and `casting` parameters.

* Reduction operations are fully documented now.

* Negative axis in reductions are not supported (they have never been
  actually), and a `ValueError` will be raised if they are used.


Changes from 1.x series to 2.0
==============================

- Added support for the new iterator object in NumPy 1.6 and later.

  This allows for better performance with operations that implies
  broadcast operations, fortran-ordered or non-native byte orderings.
  Performance for other scenarios is preserved (except for very small
  arrays).

- Division in numexpr is consistent now with Python/NumPy.  Fixes #22
  and #58.

- Constants like "2." or "2.0" must be evaluated as float, not
  integer.  Fixes #59.

- `evaluate()` function has received a new parameter `out` for storing
  the result in already allocated arrays.  This is very useful when
  dealing with large arrays, and a allocating new space for keeping
  the result is not acceptable.  Closes #56.

- Maximum number of threads raised from 256 to 4096.  Machines with a
  higher number of cores will still be able to import numexpr, but
  limited to 4096 (which is an absurdly high number already).


Changes from 1.4.1 to 1.4.2
===========================

- Multithreaded operation is disabled for small arrays (< 32 KB).
  This allows to remove the overhead of multithreading for such a
  small arrays.  Closes #36.

- Dividing int arrays by zero gives a 0 as result now (and not a
  floating point exception anymore.  This behaviour mimics NumPy.
  Thanks to Gaëtan de Menten for the fix.  Closes #37.

- When compiled with VML support, the number of threads is set to 1
  for VML core, and to the number of cores for the native pthreads
  implementation.  This leads to much better performance.  Closes #39.

- Fixed different issues with reduction operations (`sum`, `prod`).
  The problem is that the threaded code does not work well for
  broadcasting or reduction operations.  Now, the serial code is used
  in those cases.  Closes #41.

- Optimization of "compilation phase" through a better hash.  This can
  lead up to a 25% of improvement when operating with variable
  expressions over small arrays.  Thanks to Gaëtan de Menten for the
  patch.  Closes #43.

- The ``set_num_threads`` now returns the number of previous thread
  setting, as stated in the docstrings.


Changes from 1.4 to 1.4.1
=========================

- Mingw32 can also work with pthreads compatibility code for win32.
  Fixes #31.

- Fixed a problem that used to happen when running Numexpr with
  threads in subprocesses.  It seems that threads needs to be
  initialized whenever a subprocess is created.  Fixes #33.

- The GIL (Global Interpreter Lock) is released during computations.
  This should allow for better resource usage for multithreaded apps.
  Fixes #35.


Changes from 1.3.1 to 1.4
=========================

- Added support for multi-threading in pure C.  This is to avoid the
  GIL and allows to squeeze the best performance in both multi-core
  machines.

- David Cooke contributed a thorough refactorization of the opcode
  machinery for the virtual machine.  With this, it is really easy to
  add more opcodes.  See:

  http://code.google.com/p/numexpr/issues/detail?id=28

  as an example.

- Added a couple of opcodes to VM: where_bbbb and cast_ib. The first
  allow to get boolean arrays out of the `where` function.  The second
  allows to cast a boolean array into an integer one.  Thanks to
  gdementen for his contribution.

- Fix negation of `int64` numbers. Closes #25.

- Using a `npy_intp` datatype (instead of plain `int`) so as to be
  able to manage arrays larger than 2 GB.


Changes from 1.3 to 1.3.1
=========================

- Due to an oversight, ``uint32`` types were not properly supported.
  That has been solved.  Fixes #19.

- Function `abs` for computing the absolute value added.  However, it
  does not strictly follow NumPy conventions.  See ``README.txt`` or
  website docs for more info on this.  Thanks to Pauli Virtanen for
  the patch.  Fixes #20.


Changes from 1.2 to 1.3
=======================

- A new type called internally `float` has been implemented so as to
  be able to work natively with single-precision floating points.
  This prevents the silent upcast to `double` types that was taking
  place in previous versions, so allowing both an improved performance
  and an optimal usage of memory for the single-precision
  computations.  However, the casting rules for floating point types
  slightly differs from those of NumPy.  See:

      http://code.google.com/p/numexpr/wiki/Overview

  or the README.txt file for more info on this issue.

- Support for Python 2.6 added.

- When linking with the MKL, added a '-rpath' option to the link step
  so that the paths to MKL libraries are automatically included into
  the runtime library search path of the final package (i.e. the user
  won't need to update its LD_LIBRARY_PATH or LD_RUN_PATH environment
  variables anymore).  Fixes #16.


Changes from 1.1.1 to 1.2
=========================

- Support for Intel's VML (Vector Math Library) added, normally
  included in Intel's MKL (Math Kernel Library).  In addition, when
  the VML support is on, several processors can be used in parallel
  (see the new `set_vml_num_threads()` function).  With that, the
  computations of transcendental functions can be accelerated quite a
  few.  For example, typical speed-ups when using one single core for
  contiguous arrays are 3x with peaks of 7.5x (for the pow() function).
  When using 2 cores the speed-ups are around 4x and 14x respectively.
  Closes #9.

- Some new VML-related functions have been added:

  * set_vml_accuracy_mode(mode):  Set the accuracy for VML operations.

  * set_vml_num_threads(nthreads): Suggests a maximum number of
    threads to be used in VML operations.

  * get_vml_version():  Get the VML/MKL library version.

  See the README.txt for more info about them.

- In order to easily allow the detection of the MKL, the setup.py has
  been updated to use the numpy.distutils.  So, if you are already
  used to link NumPy/SciPy with MKL, then you will find that giving
  VML support to numexpr works almost the same.

- A new `print_versions()` function has been made available.  This
  allows to quickly print the versions on which numexpr is based on.
  Very handy for issue reporting purposes.

- The `numexpr.numexpr` compiler function has been renamed to
  `numexpr.NumExpr` in order to avoid name collisions with the name of
  the package (!).  This function is mainly for internal use, so you
  should not need to upgrade your existing numexpr scripts.


Changes from 1.1 to 1.1.1
=========================

- The case for multidimensional array operands is properly accelerated
  now.  Added a new benchmark (based on a script provided by Andrew
  Collette, thanks!) for easily testing this case in the future.
  Closes #12.

- Added a fix to avoid the caches in numexpr to grow too much.  The
  dictionary caches are kept now always with less than 256 entries.
  Closes #11.

- The VERSION file is correctly copied now (it was not present for the
  1.1 tar file, I don't know exactly why).  Closes #8.


Changes from 1.0 to 1.1
=======================

- Numexpr can work now in threaded environments.  Fixes #2.

- The test suite can be run programmatically by using
  ``numexpr.test()``.

- Support a more complete set of functions for expressions (including
  those that are not supported by MSVC 7.1 compiler, like the inverse
  hyperbolic or log1p and expm1 functions.  The complete list now is:

    * where(bool, number1, number2): number
        Number1 if the bool condition is true, number2 otherwise.
    * {sin,cos,tan}(float|complex): float|complex
        Trigonometric sinus, cosinus or tangent.
    * {arcsin,arccos,arctan}(float|complex): float|complex
        Trigonometric inverse sinus, cosinus or tangent.
    * arctan2(float1, float2): float
        Trigonometric inverse tangent of float1/float2.
    * {sinh,cosh,tanh}(float|complex): float|complex
        Hyperbolic sinus, cosinus or tangent.
    * {arcsinh,arccosh,arctanh}(float|complex): float|complex
        Hyperbolic inverse sinus, cosinus or tangent.
    * {log,log10,log1p}(float|complex): float|complex
        Natural, base-10 and log(1+x) logarithms.
    * {exp,expm1}(float|complex): float|complex
        Exponential and exponential minus one.
    * sqrt(float|complex): float|complex
        Square root.
    * {real,imag}(complex): float
        Real or imaginary part of complex.
    * complex(float, float): complex
        Complex from real and imaginary parts.



.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
