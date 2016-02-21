# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:30:55 2015

@author: rmcleod
"""

import numpy as np
import numexpr as ne

ne.set_num_threads(48)

print( ne.nthreads )

#### COMPLEX64 #####
a = np.random.normal( size=[512,512] ).astype('float32')
b = np.random.normal( size=[512,512] ).astype('float32')
c = np.random.normal( size=[512,512] ).astype('float32')
z = a + 1.0j*b
y = b + 1.0j*c

# print( "z.dtype = " + str(z.dtype) )

test0 = ne.evaluate( "a + b" )
print( "Float test0.dtype = " + str(test0.dtype) )

test1 = ne.evaluate( "z*y + 3.0*z**2" )
print( "Float test1.dtype = " + str(test1.dtype) )
test2 = ne.evaluate( "3.0 + 3.0*z**2 - 50*z" )
print( "Float test2.dtype = " + str(test2.dtype) )

try:
    test3 = ne.evaluate( "real(z)" )
    print( "Float test3.dtype = " + str(test3.dtype) )
except NotImplementedError, err:
    print( err )

try:
    test4 = ne.evaluate( "conj(z)" )
    print( "Float test4.dtype = " + str(test4.dtype) )
except NotImplementedError, err:
    print( err )
    
try:
    test5 = ne.evaluate( "complex(a,b)" )
    print( "Float test5.dtype = " + str(test5.dtype) )
except (NotImplementedError, RuntimeError), err:
    print( err )
    
#### COMPLEX128 #####
ad = np.random.normal( size=[512,512] ).astype('float64')
bd = np.random.normal( size=[512,512] ).astype('float64')
cd = np.random.normal( size=[512,512] ).astype('float64')
zd = ad + 1.0j*bd
yd = bd + 1.0j*cd

# print( "z.dtype = " + str(z.dtype) )
test0d = ne.evaluate( "ad + bd" )
print( "Double test0.dtype = " + str(test0d.dtype) )

test1d = ne.evaluate( "zd*yd + 3.0*zd**2" )
print( "Double test1.dtype = " + str(test1d.dtype) )
test2d = ne.evaluate( "3.0 + 3.0*zd**2 - 50*zd" )
print( "Double test2.dtype = " + str(test2d.dtype) )

try:
    test3d = ne.evaluate( "real(zd)" )
    print( "Double test3.dtype = " + str(test3d.dtype) )
except NotImplementedError, err:
    print( err )

try:
    test4d = ne.evaluate( "conj(zd)" )
    print( "Double test4.dtype = " + str(test4d.dtype) )
except NotImplementedError, err:
    print( err )
    
try:
    test5d = ne.evaluate( "complex(ad,bd)" )
    print( "Double test5.dtype = " + str(test5d.dtype) )
except (NotImplementedError, RuntimeError), err:
    print( err )