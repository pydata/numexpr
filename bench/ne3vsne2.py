
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
from time import time, sleep

expr = 'A_d * B_d + C_d'

setup_ne3 = '''
import numpy as np
import numexpr3 as ne3
ne3.set_num_threads(12)

np.random.seed(42)
A_d = np.random.uniform( size=arrSize )
B_d = np.random.uniform( size=arrSize )
C_d = np.random.uniform( size=arrSize )
out_d = np.zeros( arrSize )
'''

setup_ne2 = '''
import numpy as np
import numexpr as ne2
ne2.set_num_threads(12)

np.random.seed(42)
A_d = np.random.uniform( size=arrSize )
B_d = np.random.uniform( size=arrSize )
C_d = np.random.uniform( size=arrSize )
out_d = np.zeros( arrSize )
'''

tries = 100
arraySizes = 2**np.arange(14,22)
times_np = np.zeros_like(arraySizes, dtype='float64' )
times_ne3 = np.zeros_like(arraySizes, dtype='float64' )
times_ne2 = np.zeros_like(arraySizes, dtype='float64' )

ne3.set_num_threads(8)
ne2.set_num_threads(8)

#### FLOAT64 ####
for I, arrSize in enumerate( arraySizes ):
    times_np[I] = timeit.timeit( 'out_d=' + expr, 
            'arrSize={}\n'.format(arrSize) + setup_ne3, number=tries )
    
        
# times_ne3[I] = timeit.timeit( "ne3.evaluate('out_d={}',stackDepth=3)".format(expr), 
#            'arrSize={}\n'.format(arrSize) + setup_ne3, number=tries )
# imes_ne2[I] = timeit.timeit( '''ne2.evaluate('{}', out=out_d); 
#ne2._names_cache = ne2.utils.CacheDict(255);
#ne2._numexpr_cache = ne2.utils.CacheDict(255);'''.format(expr), 
#            'arrSize={}\n'.format(arrSize) + setup_ne2, number=tries )

    
for I, arrSize in enumerate( arraySizes ):
    np.random.seed(42)
    A_d = np.random.uniform( size=arrSize )
    B_d = np.random.uniform( size=arrSize )
    C_d = np.random.uniform( size=arrSize )
    out_d = np.zeros( arrSize )    
    t0 = time()
    for J in np.arange(tries):
        ne3.evaluate( 'out_d=A_d*B_d + C_d', stackDepth=3 )
    times_ne3[I] += time() - t0


for I, arrSize in enumerate( arraySizes ):
    np.random.seed(42)
    A_d = np.random.uniform( size=arrSize )
    B_d = np.random.uniform( size=arrSize )
    C_d = np.random.uniform( size=arrSize )
    out_d = np.zeros( arrSize )
    t0 = time()
    for J in np.arange(tries):
        ne2.evaluate( 'A_d*B_d + C_d', out=out_d )
        # Turn off the CacheDict as it defeats the purpose of benchmarking here
        ne2._names_cache.clear()
        ne2._numexpr_cache.clear()
    times_ne2[I] += time() - t0
              
    
times_np /= tries
times_ne2 /= tries
times_ne3 /= tries

plt.figure()
plt.plot( arraySizes/1024, times_np, '.-', label='NumPy', markerfacecolor='k' )
plt.plot( arraySizes/1024, times_ne2, '.-',  label='NumExpr2', markerfacecolor='k' )
plt.plot( arraySizes/1024, times_ne3, '.-', label='NumExpr3', markerfacecolor='k' )
plt.legend( loc='best' )
plt.xlabel( 'Array size (kElements)' )
plt.ylabel( 'Computation time (s)' )
plt.xlim( [0, np.max(arraySizes/1024)] )
plt.ylim( [0, np.max(times_np)] )
plt.title( "'a*b + c' with float-64" )
plt.savefig( "NE2vsNE3_float64.png", dpi=200 )


### COMPLEX64 ####
times_np_F = np.zeros_like(arraySizes, dtype='float64' )
times_ne3_F = np.zeros_like(arraySizes, dtype='float64' )
times_ne2_F = np.zeros_like(arraySizes, dtype='float64' )


cexpr = 'A_F*B_F + C_F'
setup_np_F = '''
import numpy as np
import numexpr3 as ne3
ne3.set_num_threads(12)

np.random.seed(42)
A_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
B_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
C_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
'''
for I, arrSize in enumerate( arraySizes ):
    
    times_np_F[I] = timeit.timeit( 'out_F=' + cexpr, 
            'arrSize={}\n'.format(arrSize) + setup_np_F, number=tries )
    

for I, arrSize in enumerate( arraySizes ):
    np.random.seed(42)
    A_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
    B_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
    C_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
    out_F = np.zeros( arrSize, dtype='complex64' )    
    t0 = time()
    for J in np.arange(tries):
        ne3.evaluate( 'out_F=A_F*B_F + C_F', stackDepth=3 )
    times_ne3_F[I] += time() - t0


for I, arrSize in enumerate( arraySizes ):
    np.random.seed(42)
    A_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
    B_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
    C_F = np.random.uniform( size=arrSize ).astype( 'complex64' )
    out_D = np.zeros( arrSize, dtype='complex128' )
    t0 = time()
    for J in np.arange(tries):
        ne2.evaluate( 'A_F*B_F + C_F', out=out_D )
        # Turn off the CacheDict as it defeats the purpose of benchmarking here
        ne2._names_cache.clear()
        ne2._numexpr_cache.clear()
    times_ne2_F[I] += time() - t0
              
    
times_np_F /= tries
times_ne2_F /= tries
times_ne3_F /= tries

plt.figure()
plt.plot( arraySizes/1024, times_np_F, '.-', label='NumPy', markerfacecolor='k' )
plt.plot( arraySizes/1024, times_ne2_F, '.-',  label='NumExpr2', markerfacecolor='k' )
plt.plot( arraySizes/1024, times_ne3_F, '.-', label='NumExpr3', markerfacecolor='k' )
plt.legend( loc='best' )
plt.xlabel( 'Array size (kElements)' )
plt.ylabel( 'Computation time (s)' )
plt.xlim( [0, np.max(arraySizes/1024)] )
plt.ylim( [0, np.max(times_np)] )
plt.title( "'a*b + c' with complex-64" )
plt.savefig( "NE2vsNE3_complex64.png", dpi=200 )