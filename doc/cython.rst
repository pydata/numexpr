Using NumExpr within Cython
===========================

NumExpr looks through the `locals` dict to find the variables that it was 
called with.  However, a known limitation of Cython is that it often does not 
populate the Python frame:

`Limitation of Cython <http://docs.cython.org/en/latest/src/userguide/limitations.html>`_

A workaround to call NumExpr within Cython is to populate the `local_dict` 
keyword argument so that NumExpr does not look in the `locals` dict. E.g.

.. code-block:: Python

    a = np.arange(2.0**16)
    out = np.empty_like(a)
    neObj = ne3.NumExpr( 'out=a*a', local_dict={'a':a, 'out':out} )
    neObj( out=out, a=a )
    np.testing.assert_array_almost_equal( out, a*a )

