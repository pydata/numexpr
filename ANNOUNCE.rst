========================
NumExpr 2.8.5 (mod)
========================

Changes from base 2.8.5
---------------------------

In the modified version it is possible to precompile expressions and invoke them by name:
  >>> a = np.array(1, 2, 3, 4, 5)
  >>> b = np.array(6, 7, 8, 9, 0)
  >>> ne.cache_expression("a + b", 'sum_ab')
  >>> ne.re_evaluate('sum_ab')
  array([ 0. ,  1.5,  3. ,  4.5,  6. ,  7.5,  9. , 10.5, 12. , 13.5])

It is required that variables remain the same type when cached expression is used.