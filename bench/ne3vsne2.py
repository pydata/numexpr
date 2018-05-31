
# -*- coding: utf-8 -*-
"""
Benchmarking computation time scaling with threads and array size.

@author: Robert A. McLeod
"""

import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne2
import numexpr3 as ne3
import timeit
from time import perf_counter, sleep




def bench(expr, arrSizes, dtypes, N_threads=4, tries=5):
    print("Benchmarking for {} threads".format(N_threads))
    times_np = np.zeros([len(dtypes), len(arrSizes)], dtype='float64')
    times_ne2 = np.zeros([len(dtypes), len(arrSizes)], dtype='float64')
    times_ne3 = np.zeros([len(dtypes), len(arrSizes)], dtype='float64')
    for I, dtype in enumerate(dtypes):
        for J, arrSize in enumerate(arrSizes):
        
            setup_ne3 = '''
import numpy as np
import numexpr3 as ne3
ne3.set_nthreads({0})

np.random.seed(42)
A = np.random.uniform( size={1} ).astype('{2}')
B = np.random.uniform( size={1} ).astype('{2}')
C = np.random.uniform( size={1} ).astype('{2}')
out = np.zeros( {1}, dtype='{2}' )
neFunc = ne3.NumExpr( 'out={3}' )
'''.format(N_threads, arrSize, dtype, expr)

            setup_ne2 = '''
import numpy as np
import numexpr as ne
ne.set_num_threads({0})

np.random.seed(42)
A = np.random.uniform( size={1} ).astype('{2}')
B = np.random.uniform( size={1} ).astype('{2}')
C = np.random.uniform( size={1} ).astype('{2}')
out = np.zeros( {1}, dtype='{2}' )
'''.format(N_threads, arrSize, dtype)

            setup_np = '''
import numpy as np
from numpy import sqrt
try:
    import mkl
    mkl.set_num_threads({0})
except ImportError:
    pass

np.random.seed(42)
A = np.random.uniform( size={1} ).astype('{2}')
B = np.random.uniform( size={1} ).astype('{2}')
C = np.random.uniform( size={1} ).astype('{2}')
out = np.zeros( {1}, dtype='{2}' )
'''.format(N_threads, arrSize, dtype)

            times_np[I,J] = timeit.timeit('out = ' + expr, setup_np, number=tries)

            times_ne3[I,J] = timeit.timeit('neFunc()'.format(expr),
                                        setup_ne3, number=tries)

            times_ne2[I,J] = timeit.timeit("ne.evaluate('{0}', out=out)".format(expr),
                                        setup_ne2, number=tries)

        # times_np /= tries
        # times_ne2 /= tries
        # times_ne3 /= tries

        fit_np = np.polyfit(arraySizes, times_np[I,:], 1)
        fit_ne2 = np.polyfit(arraySizes, times_ne2[I,:], 1)
        fit_ne3 = np.polyfit(arraySizes, times_ne3[I,:], 1)


        plt.figure()
        plt.plot(arraySizes/1024, times_np[I,:], '.-', label='NumPy', markerfacecolor='k')
        plt.plot(arraySizes/1024, times_ne2[I,:], '.-',  label='NumExpr2', markerfacecolor='k')
        plt.plot(arraySizes/1024, times_ne3[I,:], '.-', label='NumExpr3', markerfacecolor='k')
        plt.legend(loc='best')
        plt.xlabel('Array size (kElements)')
        plt.ylabel('Computation time (s)')
        plt.xlim([0, np.max(arraySizes/1024)])
        plt.ylim([0, np.max(times_np)])
        plt.title("'{0}' with {1}".format(expr, dtype))
        plt.savefig("NE2vsNE3_{}.png".format(dtype), dpi=200)

        print( "===RESULTS for {} for dtype {}===".format(expr,dtype) )
        print( "    Mean speedup for NE3 versus NumPy: {:.1f} %".format(100.0*fit_np[0]/fit_ne3[0]) )
        print( "    Mean speedup for NE3 versus NE2: {:.1f} %".format(100.0*fit_ne2[0]/fit_ne3[0]) )


N_threads = 4
expr = 'sqrt(A*B + 2*C)'
arraySizes = np.logspace(14,22,25,base=2)
dtypes = ['float64', 'complex128']
bench( expr, arraySizes, dtypes, N_threads=N_threads, tries=5 )