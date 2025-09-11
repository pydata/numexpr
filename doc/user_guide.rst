NumExpr 2.12 User Guide
======================

The NumExpr package supplies routines for the fast evaluation of
array expressions elementwise by using a vector-based virtual
machine.

Using it is simple::

    >>> import numpy as np
    >>> import numexpr as ne
    >>> a = np.arange(10)
    >>> b = np.arange(0, 20, 2)
    >>> c = ne.evaluate('2*a + 3*b')
    >>> c
    array([ 0,  8, 16, 24, 32, 40, 48, 56, 64, 72])


It is also possible to use NumExpr to validate an expression::

    >>> ne.validate('2*a + 3*b')

which returns `None` on success or raises an exception on invalid inputs.

and it can also re_evaluate an expression::

    >>> b = np.arange(0, 40, 4)
    >>> ne.re_evaluate()

Building
--------

*NumExpr* requires Python_ 3.7 or greater, and NumPy_ 1.13 or greater.  It is
built in the standard Python way:

.. code-block:: bash

    $ pip install .

You must have a C-compiler (i.e. MSVC Build tools on Windows and GCC on Linux) installed.

Then change to a directory that is not the repository directory (e.g. `/tmp`) and
test :code:`numexpr` with:

.. code-block:: bash

    $ python -c "import numexpr; numexpr.test()"

.. _Python: http://python.org
.. _NumPy: http://numpy.scipy.org


Enabling Intel VML support
--------------------------

Starting from release 1.2 on, numexpr includes support for Intel's VML
library.  This allows for better performance on Intel architectures,
mainly when evaluating transcendental functions (trigonometrical,
exponential, ...). It also enables numexpr using several CPU cores.

If you have Intel's MKL (the library that embeds VML), just copy the
:code:`site.cfg.example` that comes in the distribution to :code:`site.cfg` and
edit the latter giving proper directions on how to find your MKL
libraries in your system.  After doing this, you can proceed with the
usual building instructions listed above.  Pay attention to the
messages during the building process in order to know whether MKL has
been detected or not.  Finally, you can check the speed-ups on your
machine by running the :code:`bench/vml_timing.py` script (you can play with
different parameters to the :code:`set_vml_accuracy_mode()` and
:code:`set_vml_num_threads()` functions in the script so as to see how it would
affect performance).

Threadpool Configuration
------------------------

Threads are spawned at import-time, with the number being set by the environment
variable ``NUMEXPR_MAX_THREADS``. The default maximum thread count is **64**.
There is no advantage to spawning more threads than the number of virtual cores
available on the computing node. Practically NumExpr scales at large thread
count (`> 8`) only on very large matrices (`> 2**22`). Spawning large numbers
of threads is not free, and can increase import times for NumExpr or packages
that import it such as Pandas or PyTables.

If desired, the number of threads in the pool used can be adjusted via an
environment variable, ``NUMEXPR_NUM_THREADS`` (preferred) or ``OMP_NUM_THREADS``.
Typically only setting ``NUMEXPR_MAX_THREADS`` is sufficient; the number of
threads used can be adjusted dynamically via ``numexpr.set_num_threads(int)``.
The number of threads can never exceed that set by ``NUMEXPR_MAX_THREADS``.

If the user has not configured the environment prior to importing NumExpr, info
logs will be generated, and the initial number of threads *that are used*_ will
be set to the number of cores detected in the system or 8, whichever is *less*.

Usage::

    import os
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    import numexpr as ne

Usage Notes
-----------

`NumExpr`'s principal routine is::

    evaluate(ex, local_dict=None, global_dict=None, optimization='aggressive', truediv='auto')

where :code:`ex` is a string forming an expression, like :code:`"2*a+3*b"`.  The
values for :code:`a` and :code:`b` will by default be taken from the calling
function's frame (through the use of :code:`sys._getframe()`).
Alternatively, they can be specified using the :code:`local_dict` or
:code:`global_dict` arguments, or passed as keyword arguments.

The :code:`optimization` parameter can take the values :code:`'moderate'`
or :code:`'aggressive'`.  :code:`'moderate'` means that no optimization is made
that can affect precision at all.  :code:`'aggressive'` (the default) means that
the expression can be rewritten in a way that precision *could* be affected, but
normally very little.  For example, in :code:`'aggressive'` mode, the
transformation :code:`x~**3` -> :code:`x*x*x` is made, but not in
:code:`'moderate'` mode.

The `truediv` parameter specifies whether the division is a 'floor division'
(False) or a 'true division' (True).  The default is the value of
`__future__.division` in the interpreter.  See PEP 238 for details.

Expressions are cached, so reuse is fast.  Arrays or scalars are
allowed for the variables, which must be of type 8-bit boolean (bool),
32-bit signed integer (int), 64-bit signed integer (long),
double-precision floating point number (float), 2x64-bit,
double-precision complex number (complex) or raw string of bytes
(str).  If they are not in the previous set of types, they will be
properly upcasted for internal use (the result will be affected as
well).  The arrays must all be the same size.


Datatypes supported internally
------------------------------

*NumExpr* operates internally only with the following types:

    * 8-bit boolean (bool)
    * 32-bit signed integer (int or int32)
    * 64-bit signed integer (long or int64)
    * 32-bit single-precision floating point number (float or float32)
    * 64-bit, double-precision floating point number (double or float64)
    * 2x64-bit, double-precision complex number (complex or complex128)
    * Raw string of bytes (str in Python 2.7, bytes in Python 3+, numpy.str in both cases)

If the arrays in the expression does not match any of these types,
they will be upcasted to one of the above types (following the usual
type inference rules, see below).  Have this in mind when doing
estimations about the memory consumption during the computation of
your expressions.

Also, the types in NumExpr conditions are somewhat stricter than those
of Python.  For instance, the only valid constants for booleans are
:code:`True` and :code:`False`, and they are never automatically cast to integers.


Casting rules
-------------

Casting rules in NumExpr follow closely those of *NumPy*.  However, for
implementation reasons, there are some known exceptions to this rule,
namely:

    * When an array with type :code:`int8`, :code:`uint8`, :code:`int16` or
      :code:`uint16` is used inside NumExpr, it is internally upcasted to an
      :code:`int` (or :code:`int32` in NumPy notation).
    * When an array with type :code:`uint32` is used inside NumExpr, it is
      internally upcasted to a :code:`long` (or :code:`int64` in NumPy notation).
    * A floating point function (e.g. :code:`sin`) acting on :code:`int8` or
      :code:`int16` types returns a :code:`float64` type, instead of the
      :code:`float32` that is returned by NumPy functions.  This is mainly due
      to the absence of native :code:`int8` or :code:`int16` types in NumExpr.
    * In operations implying a scalar and an array, the normal rules of casting
      are used in NumExpr, in contrast with NumPy, where array types takes
      priority.  For example, if :code:`a` is an array of type :code:`float32`
      and :code:`b` is an scalar of type :code:`float64` (or Python :code:`float`
      type, which is equivalent), then :code:`a*b` returns a :code:`float64` in
      NumExpr, but a :code:`float32` in NumPy (i.e. array operands take priority
      in determining the result type).  If you need to keep the result a
      :code:`float32`, be sure you use a :code:`float32` scalar too.


Supported operators
-------------------

*NumExpr* supports the set of operators listed below:

    * Bitwise operators (and, or, not, xor): :code:`&, |, ~, ^`
    * Comparison operators: :code:`<, <=, ==, !=, >=, >`
    * Unary arithmetic operators: :code:`-`
    * Binary arithmetic operators: :code:`+, -, *, /, **, %, <<, >>`


Supported functions
-------------------

The next are the current supported set:

    * :code:`where(bool, number1, number2): number` -- number1 if the bool condition
      is true, number2 otherwise.
    * :code:`{isinf, isnan, isfinite}(float|complex): bool` -- returns element-wise True
      for ``inf`` or ``NaN``, ``NaN``, not ``inf`` respectively.
    * :code:`{sin,cos,tan}(float|complex): float|complex` -- trigonometric sine,
      cosine or tangent.
    * :code:`{arcsin,arccos,arctan}(float|complex): float|complex` -- trigonometric
      inverse sine, cosine or tangent.
    * :code:`arctan2(float1, float2): float` -- trigonometric inverse tangent of
      float1/float2.
    * :code:`{sinh,cosh,tanh}(float|complex): float|complex` -- hyperbolic sine,
      cosine or tangent.
    * :code:`{arcsinh,arccosh,arctanh}(float|complex): float|complex` -- hyperbolic
      inverse sine, cosine or tangent.
    * :code:`{log,log10,log1p}(float|complex): float|complex` -- natural, base-10 and
      log(1+x) logarithms.
    * :code:`{exp,expm1}(float|complex): float|complex` -- exponential and exponential
      minus one.
    * :code:`sqrt(float|complex): float|complex` -- square root.
    * :code:`abs(float|complex): float|complex`  -- absolute value.
    * :code:`conj(complex): complex` -- conjugate value.
    * :code:`{real,imag}(complex): float` -- real or imaginary part of complex.
    * :code:`complex(float, float): complex` -- complex from real and imaginary
      parts.
    * :code:`contains(np.str, np.str): bool` -- returns True for every string in :code:`op1` that
      contains :code:`op2`.

Notes
-----

    * :code:`abs()` for complex inputs returns a :code:`complex` output too.  This is a
      departure from NumPy where a :code:`float` is returned instead.  However,
      NumExpr is not flexible enough yet so as to allow this to happen.
      Meanwhile, if you want to mimic NumPy behaviour, you may want to select the
      real part via the :code:`real` function (e.g. :code:`real(abs(cplx))`) or via the
      :code:`real` selector (e.g. :code:`abs(cplx).real`).

More functions can be added if you need them. Note however that NumExpr 2.6 is
in maintenance mode and a new major revision is under development.

Supported reduction operations
------------------------------

The next are the current supported set:

  * :code:`sum(number, axis=None)`: Sum of array elements over a given axis.
    Negative axis are not supported.
  * :code:`prod(number, axis=None)`: Product of array elements over a given axis.
    Negative axis are not supported.

*Note:* because of internal limitations, reduction operations must appear the
last in the stack.  If not, it will be issued an error like::

    >>> ne.evaluate('sum(1)*(-1)')
    RuntimeError: invalid program: reduction operations must occur last

General routines
----------------

  * :code:`evaluate(expression, local_dict=None, global_dict=None,
    optimization='aggressive', truediv='auto')`:  Evaluate a simple array
    expression element-wise.  See examples above.
  * :code:`re_evaluate(local_dict=None)`:  Re-evaluate the last array expression
    without any check.  This is meant for accelerating loops that are re-evaluating
    the same expression repeatedly without changing anything else than the operands.
    If unsure, use evaluate() which is safer.
  * :code:`test()`:  Run all the tests in the test suite.
  * :code:`print_versions()`:  Print the versions of software that numexpr relies on.
  * :code:`set_num_threads(nthreads)`: Sets a number of threads to be used in operations.
    Returns the previous setting for the number of threads.  See note below to see
    how the number of threads is set via environment variables.

    If you are using VML, you may want to use *set_vml_num_threads(nthreads)* to
    perform the parallel job with VML instead.  However, you should get very
    similar performance with VML-optimized functions, and VML's parallelizer
    cannot deal with common expressions like `(x+1)*(x-2)`, while NumExpr's
    one can.

  * :code:`detect_number_of_cores()`: Detects the number of cores on a system.


Intel's VML specific support routines
-------------------------------------

When compiled with Intel's VML (Vector Math Library), you will be able
to use some additional functions for controlling its use. These are:

  * :code:`set_vml_accuracy_mode(mode)`:  Set the accuracy for VML operations.

    The :code:`mode` parameter can take the values:

    - :code:`'low'`: Equivalent to VML_LA - low accuracy VML functions are called
    - :code:`'high'`: Equivalent to VML_HA - high accuracy VML functions are called
    - :code:`'fast'`: Equivalent to VML_EP - enhanced performance VML functions are called

    It returns the previous mode.

    This call is equivalent to the :code:`vmlSetMode()` in the VML library. See:

    http://www.intel.com/software/products/mkl/docs/webhelp/vml/vml_DataTypesAccuracyModes.html

    for more info on the accuracy modes.

  * :code:`set_vml_num_threads(nthreads)`: Suggests a maximum number of
    threads to be used in VML operations.

    This function is equivalent to the call
    :code:`mkl_domain_set_num_threads(nthreads, MKL_VML)` in the MKL library.
    See:

    http://www.intel.com/software/products/mkl/docs/webhelp/support/functn_mkl_domain_set_num_threads.html

    for more info about it.

  * :code:`get_vml_version()`:  Get the VML/MKL library version.


Authors
-------

.. include:: ../AUTHORS.txt

License
-------

NumExpr is distributed under the MIT_ license.

.. _MIT: http://www.opensource.org/licenses/mit-license.php
