Functions and Function signatures
=================================

Adding functions
----------------

In order to add new functions to ``numexpr``, currently it is necessary to edit several files. Consider adding a function
``out_type myfunc(arg_type)``.

* ``numexpr/expressions.py``
Add ``'myfunc': func(numpy.myfunc, out_dtype),`` to the dict of functions, ``functions = {...``. If the return type of the function is ``bool``, add
the function to the list ``if opcode in ("isnan", "isfinite"):`` in the ``__init__`` function of the ``FuncNode`` class.
In the future it might be nice to refactor this function since it sets the output type based on the type of the inputs in general.

* ``numexpr/necompiler.py``
Add ``"myfunc"`` to the list of functions:

.. code-block:: python3

    "floor",
    "isnan",
    "isfinite",
    "myfunc"
    ]

* ``numexpr/functions.hpp``
Find the correct function signature ``FUNC_OA`` where ``O`` is the return type, and ``A`` the argument type(s). For example, if the function
is ``double myfunc(double)``, one should edit within the ``FUNC_DD`` clause. If you cannot find your function signature you will have to add it,
following the template of the other functions.
Most likely, you will want to add support for several function signatures (e.g. double -> bool and float -> bool) and so you will have to add the
function in two clauses. If your function has a float input, you will see that there are 5 arguments in the
``FUNC_OA`` macro, and you will have to add ``myfunc2`` here is order to compile on MSVC machines (i.e. Windows, see following).
Example:

.. code-block:: cpp
   :emphasize-lines: 6, 20

    #ifndef FUNC_DD
    #define ELIDE_FUNC_DD
    #define FUNC_DD(...)
    #endif
    ...
    FUNC_DD(FUNC_MYFUNC_DD, "myfunc_dd", myfunc, vdMyfunc)
    FUNC_DD(FUNC_DD_LAST,    NULL,          NULL,  NULL)
    #ifdef ELIDE_FUNC_DD
    #undef ELIDE_FUNC_DD
    #undef FUNC_DD
    #endif

    ...

    #ifndef FUNC_FF
    #define ELIDE_FUNC_FF
    #define FUNC_FF(...)
    #endif
    ...
    FUNC_FF(FUNC_MYFUNC_FF, "myfunc_ff", myfuncf, myfuncf2, vfMyfunc)
    FUNC_FF(FUNC_FF_LAST,    NULL,       NULL,    NULL,     NULL)
    #ifdef ELIDE_FUNC_FF
    #undef ELIDE_FUNC_FF
    #undef FUNC_FF
    #endif

* ``numexpr/msvc_function_stubs.hpp``
In order to support float arguments, due to oddities of MSVC, you have to provide explicit support for your function in this file.
Add ``#define myfuncf(x)  ((float)floor((double)(x)))`` (if your function is float -> float) to the ``#if`` clause at the top of the file
which is for old versions of MSVC which did not have support for single precision fucntions. Then in the body, add an inline function

.. code-block:: cpp

    inline float myfuncf2(float x) {
        return myfuncf(x);
    }

This is the function that appears as the ``f_win32`` parameter in ``functions.hpp``.

* ``numexpr/tests/test_numexpr.py``
Don't forget to add a test for your function!

Adding function signatures
--------------------------
It may so happen that you cannot find your desired function signature in ``functions.hpp``. This means you will have to add it yourself!
This involves editing a few more files. In addition, there may be certain bespoke changes, specific to the function signature
that you may have to make (see Notes, below)

* ``numexpr/functions.hpp``
Firstly, add clause(s) for your function signature. For example, if the function signature is ``bool(double)`` and ``bool(float)``, add
``FUNC_BD`` and ``FUNC_BF`` clauses (in the latter case you will need the macro to take 5 arguments for MSVC-compatibility.)

.. code-block:: cpp

    #ifndef FUNC_BD
    #define ELIDE_FUNC_BD
    #define FUNC_BD(...)
    #endif
    ...
    FUNC_BD(FUNC_BD_LAST,    NULL,          NULL,  NULL)
    #ifdef ELIDE_FUNC_BD
    #undef ELIDE_FUNC_BD
    #undef FUNC_BD
    #endif

    #ifndef FUNC_BF
    #define ELIDE_FUNC_BF
    #define FUNC_BF(...)
    #endif
    ...
    FUNC_BF(FUNC_BF_LAST,    NULL,     NULL,     NULL,  NULL)
    #ifdef ELIDE_FUNC_BF
    #undef ELIDE_FUNC_BF
    #undef FUNC_BF
    #endif

The ultimate source of the functions in the macro ``FUNC_BF(...)`` are the headers included in ``numexpr/interpreter.cpp`` (in particular
``numexpr/numexpr_config.hpp``, which can be used to overwrite ``<math.h>`` functions), so the functions should be available from there.

* ``numexpr/interp_body.cpp``
Add case support for OPCODES associated to your new function signatures via e.g. ``case OP_FUNC_BFN`` and ``case OP_FUNC_BDN``, following
the framework suggested by the other functions:

.. code-block:: cpp

    case OP_FUNC_BFN:
    #ifdef USE_VML
                VEC_ARG1_VML(functions_bf_vml[arg2](BLOCK_SIZE,
                                                    (float*)x1, (bool*)dest));
    #else
                VEC_ARG1(b_dest = functions_bf[arg2](f1));
    #endif

Note that it is important that the out variable matches the output type of the function (i.e. ``b_dest`` for bool, ``f_dest`` for float etc.)

* ``numexpr/interpreter.hpp``
Add clauses to read the ``functions.hpp`` macros correctly

.. code-block:: cpp

    enum FuncBFCodes {
    #define FUNC_BF(fop, ...) fop,
    #include "functions.hpp"
    #undef FUNC_BF
    };

* ``numexpr/interpreter.cpp``
Add clauses to generate the FUNC_CODES from the ``functions.hpp`` header, making sure to include clauses for ``_WIN32`` and
``VML`` as necessary accoridng to the framework suggested by the other functions.

.. code-block:: cpp

    typedef bool (*FuncBFPtr)(float);
    #ifdef _WIN32
    FuncBFPtr functions_bf[] = {
    #define FUNC_BF(fop, s, f, f_win32, ...) f_win32,
    #include "functions.hpp"
    #undef FUNC_BF
    };
    #else
    FuncBFPtr functions_bf[] = {
    #define FUNC_BF(fop, s, f, ...) f,
    #include "functions.hpp"
    #undef FUNC_BF
    };
    #endif

    #ifdef USE_VML
    typedef void (*FuncBFPtr_vml)(MKL_INT, const float*, bool*);
    FuncBFPtr_vml functions_bf_vml[] = {
    #define FUNC_BF(fop, s, f, f_win32, f_vml) f_vml,
    #include "functions.hpp"
    #undef FUNC_BF
    };
    #endif

Add case handling to the ``check_program`` function

.. code-block:: cpp

    else if (op == OP_FUNC_BDN) {
        if (arg < 0 || arg >= FUNC_BD_LAST) {
            PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
            return -1;
        }
    }
    else if (op == OP_FUNC_BFN) {
        if (arg < 0 || arg >= FUNC_BF_LAST) {
            PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
            return -1;
        }
    }

* ``numexpr/module.cpp``
Add code here to define the ``FUNC_OA`` macros you require

.. code-block:: cpp

    #define FUNC_BF(name, sname, ...)  add_func(name, sname);
    #define FUNC_BD(name, sname, ...)  add_func(name, sname);
    ...
    #include "functions.hpp"
    ...
    #undef FUNC_BD
    #undef FUNC_BF

* ``numexpr/opcodes.hpp``
Finally, add the ``OP_FUNC_BDN`` etc. codes here. It is necessary for the OPCODES in the file to be in (ascending order) with
``NOOP`` as 0 and ``OP_LAST`` as the largest number. Secondly, all reduction OPCODES must appear last. Hence, after adding your
function signatures (just before the reduction OPCODES) it is necessary to increment all succeeding OPCODES.

.. code-block:: cpp

    OPCODE(106, OP_FUNC_BDN, "func_bdn", Tb, Td, Tn, T0)
    OPCODE(107, OP_FUNC_BFN, "func_bfn", Tb, Tf, Tn, T0)

Notes
-----
In many cases this process will not be very smooth since one relies on the internal C/C++ standard functions (which can be fussy,
to varying degrees on different platforms). Some common gotchas are then:

* OPCODES are currently only supported up to 255 - if it becomes necessary to increment further, one will have to change the ``latin_1``
encoding used in ``quadrupleToString`` in ``necompiler.py``. In addition, since the OPCDE table is assumed to be of type ``unsigned char``
the ``get_return_sig`` function in ``numexpr/interpreter.cpp`` may have to be changed (possibly other changes too).

* Depending on the new function signature (above all if the out type is different to the input types), one may have to edit the ``__init__``
function in the ``FuncNode`` class in ``expressions.py``.

* Depending on MSVC support, namespace clashes, casting problems, it may be necessary to make various changes to ``numexpr/numexpr_config.hpp``
and ``numexpr/msvc_function_stubs.hpp``. For example, in PR #523, non-clashing wrappers were introduced for ``isnan`` and ``isfinite`` since
the float versions ``isnanf, isfinitef`` were inconsistently defined (and output ints) - depending on how strict the platform interpreter is, the implicit cast
from int to bool was acceptable or not for example. In addition, the base functions were in different namespaces or had different names across platforms.
