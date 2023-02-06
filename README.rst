============================================================
NumExpr_mod: Fast numerical expression evaluator for NumPy
============================================================

:Author: Alexander K.
:URL: https://github.com/MrCheatak/numexpr_mod


What is NumExpr?
----------------
Please refer to the original `Numexpr <https://github.com/pydata/numexpr>`_ repo.


Installation
------------

From wheels
^^^^^^^^^^^

NumExpr is available for install via `pip` for a wide range of platforms and 
Python versions (which may be browsed at: https://pypi.org/project/numexpr/#files). 
Installation can be performed as::

    pip install numexpr_mod

From Source
^^^^^^^^^^^

On most \*nix systems your compilers will already be present. However if you 
are using a virtual environment with a substantially newer version of Python than
your system Python you may be prompted to install a new version of `gcc` or `clang`.

For Windows, you will need to install the Microsoft Visual C++ Build Tools 
(which are free) first. The version depends on which version of Python you have 
installed:

https://wiki.python.org/moin/WindowsCompilers

For Python 3.6+ simply installing the latest version of MSVC build tools should 
be sufficient. Note that wheels found via pip do not include MKL support. Wheels 
available via `conda` will have MKL, if the MKL backend is used for NumPy.

See `requirements.txt` for the required version of NumPy.

NumExpr is built in the standard Python way::

  python setup.py build install

You can test `numexpr` with::

  python -c "import numexpr_mod; numexpr_mod.test()"

Do not test NumExpr in the source directory or you will generate import errors.

Usage
-----

::

    >>> import numexpr_mod as ne
    >>> import numpy as np

    >>> a = np.array([1,2,3,4,5])
    >>> b = np.array([6,7,8,9,0])

    >>> ne.cache_expression('a + b', 'sum_ab')
    {'ex': <numexpr_mod.NumExpr object at 0x1090e36b0>, 'argnames': ['a', 'b'], 'kwargs': {'out': None, 'order': 'K', 'casting': 'safe', 'ex_uses_vml': False}}
    >>> ne.re_evaluate('sum_ab')
    array([ 7,  9, 11, 13,  5], dtype=int64)
    >>> ne.evaluate('a + b')
    array([ 7,  9, 11, 13,  5], dtype=int64)


Documentation
-------------

Please see the official documentation at `numexpr.readthedocs.io <https://numexpr.readthedocs.io>`_.
Included is a user guide, benchmark results, and the reference API.


Authors
-------

Please see `AUTHORS.txt <https://github.com/pydata/numexpr/blob/master/AUTHORS.txt>`_.


License
-------

NumExpr is distributed under the `MIT <http://www.opensource.org/licenses/mit-license.php>`_ license.


.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 70
.. End:
