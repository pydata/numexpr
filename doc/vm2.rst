Performance of the Virtual Machine in NumExpr2.0
================================================

Numexpr 2.0 leverages a new virtual machine completely based on the new ndarray 
iterator introduced in NumPy 1.6.  This represents a nice combination of the 
advantages of using the new iterator, while retaining the ability to avoid 
copies in memory as well as the multi-threading capabilities of the previous 
virtual machine (1.x series).

The increased performance of the new virtual machine can be seen in several 
scenarios, like:

  * *Broadcasting*.  Expressions containing arrays that needs to be broadcasted, 
    will not need additional memory (i.e. they will be broadcasted on-the-fly).
  * *Non-native dtypes*.  These will be translated to native dtypes on-the-fly, 
    so there is not need to convert the whole arrays first.
  * *Fortran-ordered arrays*.  The new iterator will find the best path to 
    optimize operations on such arrays, without the need to transpose them first.

There is a drawback though: performance with small arrays suffers a bit because 
of higher set-up times for the new virtual machine.  See below for detailed 
benchmarks.

Some benchmarks for best-case scenarios
---------------------------------------

Here you have some benchmarks of some scenarios where the new virtual machine 
actually represents an advantage in terms of speed (also memory, but this is 
not shown here).  As you will see, the improvement is notable in many areas, 
ranging from 3x to 6x faster operations.

Broadcasting
^^^^^^^^^^^^

    >>> a = np.arange(1e3)
    >>> b = np.arange(1e6).reshape(1e3, 1e3)

    >>> timeit ne.evaluate("a*(b+1)")   # 1.4.2
    100 loops, best of 3: 16.4 ms per loop

    >>> timeit ne.evaluate("a*(b+1)")  # 2.0
    100 loops, best of 3: 5.2 ms per loop


Non-native types
^^^^^^^^^^^^^^^^

    >>> a = np.arange(1e6, dtype=">f8")
    >>> b = np.arange(1e6, dtype=">f8")

    >>> timeit ne.evaluate("a*(b+1)")  # 1.4.2
    100 loops, best of 3: 17.2 ms per loop

    >>> timeit ne.evaluate("a*(b+1)")  # 2.0
    100 loops, best of 3: 6.32 ms per loop


Fortran-ordered arrays
^^^^^^^^^^^^^^^^^^^^^^

    >>> a = np.arange(1e6).reshape(1e3, 1e3).copy('F')
    >>> b = np.arange(1e6).reshape(1e3, 1e3).copy('F')

    >>> timeit ne.evaluate("a*(b+1)")  # 1.4.2
    10 loops, best of 3: 32.8 ms per loop

    >>> timeit ne.evaluate("a*(b+1)")  # 2.0
    100 loops, best of 3: 5.62 ms per loop



Mix of 'non-native' arrays, Fortran-ordered, and using broadcasting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> a = np.arange(1e3, dtype='>f8').copy('F')
    >>> b = np.arange(1e6, dtype='>f8').reshape(1e3, 1e3).copy('F')

    >>> timeit ne.evaluate("a*(b+1)")  # 1.4.2
    10 loops, best of 3: 21.2 ms per loop

    >>> timeit ne.evaluate("a*(b+1)")  # 2.0
    100 loops, best of 3: 5.22 ms per loop


Longer setup-time
^^^^^^^^^^^^^^^^^

The only drawback of the new virtual machine is during the computation of 
small arrays::

    >>> a = np.arange(10)
    >>> b = np.arange(10)

    >>> timeit ne.evaluate("a*(b+1)")  # 1.4.2
    10000 loops, best of 3: 22.1 µs per loop

    >>> timeit ne.evaluate("a*(b+1)")  # 2.0
    10000 loops, best of 3: 30.6 µs per loop


i.e. the new virtual machine takes a bit more time to set-up (around 8 µs in 
this machine).  However, this should be not too important because for such a 
small arrays NumPy is always a better option::

    >>> timeit c = a*(b+1)
    100000 loops, best of 3: 4.16 µs per loop


And for arrays large enough the difference is negligible::

    >>> a = np.arange(1e6)
    >>> b = np.arange(1e6)

    >>> timeit ne.evaluate("a*(b+1)")  # 1.4.2
    100 loops, best of 3: 5.77 ms per loop

    >>> timeit ne.evaluate("a*(b+1)")  # 2.0
    100 loops, best of 3: 5.77 ms per loop


Conclusion
----------

The new virtual machine introduced in numexpr 2.0 brings more performance in 
many different scenarios (broadcast, non-native dtypes, Fortran-orderd arrays), 
while it shows slightly worse performance for small arrays.  However, as 
numexpr is more geared to compute large arrays, the new virtual machine should 
be good news for numexpr users in general.