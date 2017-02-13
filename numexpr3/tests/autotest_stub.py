# -*- coding: utf-8 -*-
"""
autotest_stub.py
Created on Mon Feb 13 09:54:32 2017
@author: Robert A. McLeod

Using the operations produced by code_generators/interp_generator.py, 

There should be two version: 
    1.) single-threaded, with a small array
    2.) multi-threaded, with an array > 4*BLOCK_SIZE, or 2**16
"""

import unittest
import numpy as np
import numexpr3 as ne

SMALL_SIZE = 256
LARGE_SIZE = 65536

np.random.seed(42)
# Float-64
A_d = np.random.uniform( size=LARGE_SIZE )
B_d = np.random.uniform( size=LARGE_SIZE )
C_d = np.random.uniform( size=LARGE_SIZE )
# Float-32
A_f = A_d.astype('float32')
B_f = B_d.astype('float32')
C_f = C_d.astype('float32')
# Int-64
A_l = np.random.randint( 0, high=100, size=LARGE_SIZE )
B_l = np.random.randint( 0, high=100, size=LARGE_SIZE )
C_l = np.random.randint( 0, high=100, size=LARGE_SIZE )
# Int-32
A_i = A_l.astype('int32')
B_i = B_l.astype('int32')
C_i = C_l.astype('int32')
# Int-16
A_h = A_l.astype('int16')
B_h = B_l.astype('int16')
C_h = C_l.astype('int16')
# Int-8
A_b = A_l.astype('int8')
B_b = B_l.astype('int8')
C_b = C_l.astype('int8')
# UInt-64
A_L = A_l.astype('uint64')
B_L = B_l.astype('uint64')
C_L = C_l.astype('uint64')
# UInt-32
A_I = A_l.astype('uint32')
B_I = B_l.astype('uint32')
C_I = C_l.astype('uint32')
# UInt-16
A_H = A_l.astype('uint16')
B_H = B_l.astype('uint16')
C_H = C_l.astype('uint16')
# UInt-8
A_B = A_l.astype('uint8')
B_B = B_l.astype('uint8')
C_B = C_l.astype('uint8')
# Bool
A_1 = (A_l > 50).astype('bool')
B_1 = (B_l > 50).astype('bool')
C_1 = (C_l > 50).astype('bool')
# Complex-64
A_F = A_f + 1j*B_f
B_F = B_f + 1j*C_f
C_F = C_f + 1j*A_f
# Complex-128
A_D = A_d + 1j*B_d
B_D = B_d + 1j*C_d
C_D = C_d + 1j*A_d
    
class autotest_numexpr(unittest.TestCase):
    
    def setUp(self):
        # Can put any threading or other control statements in here.
        pass
    
#GENERATOR_INSERT_POINT
    

def run():
    from numexpr3 import __version__
    print( "NumExpr3 auto-test for {} ".format(__version__) )
    unittest.main( exit=False )
    
if __name__ == "__main__":
    # Should generally call "python -m unittest -v numexpr3.test" for continuous integration
    run()
    
    
    