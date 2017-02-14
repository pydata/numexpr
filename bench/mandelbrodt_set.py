#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:49:36 2017

@author: rmcleod
"""
import numpy as np
import numexpr3 as ne3
import numexpr as ne2
import timeit
from time import time
import matplotlib.pyplot as plt

ne3.set_num_threads(4)
ne2.set_num_threads(4)

def mandelbrot_ne3(c, maxiter):
    output = np.zeros(c.shape, dtype='float32')
    notdone = np.zeros(c.shape, dtype='bool')
    z = np.zeros(c.shape, dtype='complex64' )

    # Almost 30 % of the time in a comparison appears to be in the 
    # cast to npy_bool
    neObj1 = ne3.NumExpr( 'notdone = abs2(z) < 4.0' )
    neObj2 = ne3.NumExpr( 'z = where(notdone,z*z+c,z)' )
    for it in np.arange(maxiter, dtype='float32'):
        # Here 'it' changes, but the AST parser doesn't know that and treats it
        # as a const if we use 'where(notdone, it, output)'
        # What we really need is an iter( ops, range ) function inside 
        # ne3.  This is an interesting case, since really here we see a 
        # major limitation in NumExpr working inside a loop.
        neObj1.run( check_arrays=False )
        output[notdone] = it
        neObj2.run( check_arrays=False )
    
    output[output == maxiter-1] = 0
    return output

def mandelbrot_set_ne3(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype='float32')
    r2 = np.linspace(ymin, ymax, height, dtype='float32')
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_ne3(c,maxiter)
    return (r1,r2,n3.T) 

import numexpr as ne2

def mandelbrot_ne2(c, maxiter):
    output = np.zeros(c.shape)
    z = np.zeros(c.shape, np.complex64)
    for it in range(maxiter):
        notdone = ne2.evaluate('z.real*z.real + z.imag*z.imag < 4.0')
        output[notdone] = it
        z = ne2.evaluate('where(notdone,z**2+c,z)')
    output[output == maxiter-1] = 0    
    return output

def mandelbrot_set_ne2(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype='float32')
    r2 = np.linspace(ymin, ymax, height, dtype='float32')
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_ne2(c,maxiter)
    return (r1,r2,n3.T) 

t0 = time()
_, _, set1_n3 = mandelbrot_set_ne3(-2.0,0.5,-1.25,1.25,1000,1000,80)
t1 = time()
print( "NE3 Time set#1 = %s"%(t1-t0) )

t2 = time()
_, _, set2_n3 = mandelbrot_set_ne3(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
t3 = time()
print( "NE3 Time set#2 = %s"%(t3-t2) )

t4 = time()
_, _, set1_n2 = mandelbrot_set_ne2(-2.0,0.5,-1.25,1.25,1000,1000,80)
t5 = time()
print( "NE2 Time set#1 = %s"%(t5-t4) )
t6 = time()
_, _, set2_n2 = mandelbrot_set_ne2(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
t7 = time()
print( "NE2 Time set#2 = %s"%(t7-t6) )

plt.figure()
plt.imshow( set1_n3 )

plt.figure()
plt.imshow( set2_n3 )
