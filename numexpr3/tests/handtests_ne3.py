# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:16:07 2017
@author: Robert A. McLeod

These are some hand-tests for debugging the NumExpr3 alpha, to be replaced by 
a proper unittest module in the future.
"""

import numpy as np
from time import time
import numexpr as ne2
import numexpr3 as ne3

# Simple operation, comparison with Ne2 and NumPy for break-even point
ne3.set_num_threads(12)
ne2.set_num_threads(12)

arrSize = int(2**20-42) # The minus is to make the last block a different size

print( "Array size: {:.2f}k".format(arrSize/1024 ))


a = np.pi*np.ones( arrSize )
b = 0.5*np.ones( arrSize )
c = 42*np.ones( arrSize )
yesno = np.random.uniform( size=arrSize ) > 0.5
out = np.zeros( arrSize )
out_ne2 = np.zeros( arrSize )
out_int = np.zeros( arrSize, dtype='int32' )

# Initialize threads with un-tracked calls
ne2.evaluate( 'a+b+1' )
neObj = ne3.NumExpr( 'out = a + b + 1')
# So what is with stackDepth?  Why does it change constantly?
neObj.run( b=b, a=a, out=out )

# Try a std::cmath function
# Not very fast for NE3 here, evidently the GNU cmath isn't being 
# vectorized despite them being inline functions.
t60 = time()
neObj = ne3.NumExpr( 'out_magic1 = sqrt(b)')
neObj.disassemble()
neObj.run( check_arrays=False )
t61 = time()
ne2.evaluate( 'sqrt(b)' )
t62 = time()
np.testing.assert_array_almost_equal( np.sqrt(b), out_magic1 )
print( "---------------------" )
print( "Ne3 completed single call: %.2e s"%(t61-t60) )
print( "Ne2 completed single call: %.2e s"%(t62-t61) )

# Old-school expression with no assignment, just a return.
neExpr = ne3.NumExpr( "a*b" )
expr_out = neExpr.run( a=a, b=b )
np.testing.assert_array_almost_equal( a*b, expr_out )

# For some reason NE3 is significantly faster if we do not call NE2 
# in-between.  I wonder if there's something funny with Python handling 
# of the two modules.
t0 = time()
neObj = ne3.NumExpr( 'out=a*b' )
neObj.run( b=b, a=a, out=out )
t1 = time()

t2 = time()
ne2.evaluate( 'a*b', out=out_ne2 )
t3 = time()
out_np = a*b
t4 = time()
print( "---------------------" )
print( "Ne3 completed simple-op a*b: %.2e s"%(t1-t0) )
print( "Ne2 completed simple-op a*b: %.2e s"%(t3-t2) )
print( "numpy completed simple-op a*b: %.2e s"%(t4-t3) )
np.testing.assert_array_almost_equal( out_np, out )


# In-place op
# Not such a huge gap, only ~ 200 %
neObj = ne3.NumExpr( 'out=a+b' )
neObj.run( b=b, a=a, out=out )

t50 = time()
inplace = ne3.NumExpr( 'a = a*a' )
inplace.run( check_arrays=False )
t51 = time()
inplace_ne2 = ne2.evaluate( 'a*a', out=a )
t52 = time()
print( "---------------------" )
print( "Ne3 in-place op: %.2e s"%(t51-t50) )
print( "Ne2 in-place op: %.2e s"%(t52-t51) )
del inplace


##############################################
###### Multi-line with named temporary  ######
##############################################
# Are there fewer temporaries in the ne2 program?
# 
#    # Run once for each of NE2 and NE3 to start interpreters

t40 = time()
neObj = ne3.NumExpr( 'temp = a*a + b*c - a; out_magic = c / sqrt(temp)' )
result = neObj.run( b=b, a=a, c=c )
t41 = time()
temp = ne2.evaluate( 'a*a + b*c - a' )
out_ne2 = ne2.evaluate( 'c / sqrt(temp)' )
t42 = time()
print( "---------------------" )
print( "Ne3 completed multiline: %.2e s"%(t41-t40) )
print( "Ne2 completed multiline: %.2e s"%(t42-t41) )
np.testing.assert_array_almost_equal( c / np.sqrt(a*a + b*c - a), out_magic )

# Magic output on __call() rather than __binop
test = ne3.evaluate( 'out_magic1=sqrt(a)' )
np.testing.assert_array_almost_equal( np.sqrt(a), out_magic1 )

   

'''
 Comparing NE3 versus NE2 optimizations:
 1.) So we have extra casts, we should make sure scalars are the right 
      dtype in Python or in C?  
 2.) ne2 has a power optimization and I don't yet.  I would prefer the 
     pow optimization be at the function level in the C-interpreter.
 3.) We have more temporaries sometimes still?
 4.) Can we do in-place operations with temporaries to further optimize?
     In-place operations are faster in NE2 than binops, and about the same 
     speed in NE3.
 5.) ne2 drops the first b*b into the zeroth (return) register. We would 
     need more logic for when the output is valid as a temporary.
'''

# Where/Ternary
expr = 'out = where(yesno,a,b)'
neObj = ne3.NumExpr( expr )
neObj.run( out=out, yesno=yesno, a=a, b=b )
#neObj.print_names()
np.testing.assert_array_almost_equal( np.where(yesno,a,b), out )

# Try a C++/11 cmath function
# Note behavoir here is different from NumPy... which returns double.
expr = "out_int = round(a)"
neObj = ne3.NumExpr( expr )
neObj.run( out_int=out_int, a=a )
# round() doesn't work on Windows?
try:
    np.testing.assert_array_almost_equal( np.round(a).astype('int32'), out_int )
except AssertionError as e:
    print( e )

# Try C++/11 FMA function
t10 = time()
expr = "out = fma(a,b,c)"
neObj = ne3.NumExpr( expr )
neObj.run( out=out, a=a, b=b, c=c  )
t11 = time()
ne2.evaluate( "a*b+c", out=out_ne2 )
t12 = time() 
out_np = a*b+c
t13 = time()
# FMA doesn't scale well with large arrays.
np.testing.assert_array_almost_equal( out_np, out )
print( "---------------------" )
print( "Ne3 completed fused multiply-add: %.2e s"%(t11-t10) )
print( "Ne2 completed multiply-add: %.2e s"%(t12-t11) )
print( "numpy completed multiply-add: %.2e s"%(t13-t12) )


#######################################
###### TESTING COMPLEX NUMBERS  #######
#######################################
pi2 = np.pi/2.0
ncx = np.random.uniform( -pi2, pi2, arrSize ).astype('complex64') + 1j
ncy = np.complex64(1) + 1j*np.random.uniform( -pi2, pi2, arrSize ).astype('complex64')
out_c = np.zeros( arrSize, dtype='complex64' )
out_c_ne2 = np.zeros( arrSize, dtype='complex128' )

t20 = time()
neObj = ne3.NumExpr( 'out_c = ncx*ncy'  )
neObj.run( out_c=out_c, ncx=ncx, ncy=ncy )
t21 = time()
ne2.evaluate( 'ncx*ncy', out=out_c_ne2 )
t22 = time()
out_c_np = ncx*ncy
t23 = time()

np.testing.assert_array_almost_equal( out_c_np, out_c )

print( "---------------------" )
print( "Ne3 completed complex64 ncx*ncy: %.2e s"%(t21-t20) )
print( "Ne2 completed complex128 ncx*ncy: %.2e s"%(t22-t21) )
print( "numpy completed complex64 ncx*ncy: %.2e s"%(t23-t22) )

neObj = ne3.NumExpr( 'out_c = sqrt(ncx)'  )
neObj.run( out_c=out_c, ncx=ncx )
np.testing.assert_array_almost_equal( np.sqrt(ncx), out_c )
neObj = ne3.NumExpr( 'out_c = exp(ncx)'  )
neObj.run( out_c=out_c, ncx=ncx )
np.testing.assert_array_almost_equal( np.exp(ncx), out_c )
neObj = ne3.NumExpr( 'out_c = cosh(ncx)'  )
neObj.run( out_c=out_c, ncx=ncx )
np.testing.assert_array_almost_equal( np.cosh(ncx), out_c )


########################################
###### TESTING COMPLEX Intel VML  ######
########################################
# distutils...
#interpreter._set_vml_num_threads(4)
#expr = 'out = a + b'
#neObj = NumExpr( expr, lib=LIB_VML )
#neObj.run( out, a, b )

#################################
###### TEST with striding  ######
#################################
# DEBUG: why do calls to NE2 slow down NE3?  Maybe the interpreter 
# is doing some extra work in the background?
neObj = ne3.NumExpr( 'out=a+b' )
neObj.run( out, a, b )

da = a[::4]
db = b[::4]
out_stride1 = np.empty_like(da)
out_stride2 = np.empty_like(da)
t30 = time()
neObj = ne3.NumExpr( 'out_stride1 = da*db' )
neObj.run( out_stride1, da, db )
t31 = time()
ne2.evaluate( 'da*db', out=out_stride2 )
t32 = time()
print( "---------------------" )
print( "Strided computation:" )
print( "Ne3 completed (strided) a*b: %.2e s"%(t31-t30) )
print( "Ne2 completed (strided) a*b: %.2e s"%(t32-t31) )
   
np.testing.assert_array_almost_equal( out_stride1, da*db )


####################
#### COMPARISON ####
####################
out_bool = np.zeros_like(a, dtype='bool')
neObj = ne3.NumExpr( 'out_bool = b > a' )
# No check of inputs
neObj.run( check_arrays=False )


###################################################
# Uint8                                           #
# A case where NumExpr2 returns the wrong result. #
###################################################

lena = np.random.randint( 0, 255, [3600,3200] ).astype('uint8')
paul = np.random.randint( 0, 255, [3600,3200] ).astype('uint8')

t70 = time()
ne3.evaluate( 'sum_image = lena - paul' )
t71 = time()
sum_ne2 = ne2.evaluate( 'lena - paul' )
t72 = time()

np.testing.assert_array_almost_equal( sum_image, lena-paul )
print( "uint8 image processing:" )
print( "Ne3 completed lena - paul: %.2e s"%(t71-t70) )
print( "Ne2 completed lena - paul: %.2e s"%(t72-t71) )
