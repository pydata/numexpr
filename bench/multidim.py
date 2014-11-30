###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

# Script to check that multidimensional arrays are speed-up properly too
# Based on a script provided by Andrew Collette.

from __future__ import print_function
import numpy as np
import numexpr as nx
import time

test_shapes = [
    (100*100*100),
    (100*100,100),
    (100,100,100),
    ]

test_dtype = 'f4'
nruns = 10                   # Ensemble for timing

def chunkify(chunksize):
    """ Very stupid "chunk vectorizer" which keeps memory use down.
        This version requires all inputs to have the same number of elements,
        although it shouldn't be that hard to implement simple broadcasting.
    """

    def chunkifier(func):

        def wrap(*args):

            assert len(args) > 0
            assert all(len(a.flat) == len(args[0].flat) for a in args)

            nelements = len(args[0].flat)
            nchunks, remain = divmod(nelements, chunksize)

            out = np.ndarray(args[0].shape)

            for start in range(0, nelements, chunksize):
                #print(start)
                stop = start+chunksize
                if start+chunksize > nelements:
                    stop = nelements-start
                iargs = tuple(a.flat[start:stop] for a in args)
                out.flat[start:stop] = func(*iargs)
            return out

        return wrap

    return chunkifier

test_func_str = "63 + (a*b) + (c**2) + b"

def test_func(a, b, c):
    return 63 + (a*b) + (c**2) + b

test_func_chunked = chunkify(100*100)(test_func)

for test_shape in test_shapes:
    test_size = np.product(test_shape)
    # The actual data we'll use
    a = np.arange(test_size, dtype=test_dtype).reshape(test_shape)
    b = np.arange(test_size, dtype=test_dtype).reshape(test_shape)
    c = np.arange(test_size, dtype=test_dtype).reshape(test_shape)


    start1 = time.time()
    for idx in range(nruns):
        result1 = test_func(a, b, c)
    stop1 = time.time()

    start2 = time.time()
    for idx in range(nruns):
        result2 = nx.evaluate(test_func_str)
    stop2 = time.time()

    start3 = time.time()
    for idx in range(nruns):
        result3 = test_func_chunked(a, b, c)
    stop3 = time.time()

    print("%s %s (average of %s runs)" % (test_shape, test_dtype, nruns))
    print("Simple: ", (stop1-start1)/nruns)
    print("Numexpr: ", (stop2-start2)/nruns)
    print("Chunked: ", (stop3-start3)/nruns)


