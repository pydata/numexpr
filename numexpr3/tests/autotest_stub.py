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
import numexpr3 as ne3
import os


    
class autotest_numexpr(unittest.TestCase):


    
    def setUp(self):
        # Don't use powers of 2 for sizes as BLOCK_SIZEs are powers of 2, and we want 
        # to test when we have sub-sized blocks in the last cycle through the program.
        SMALL_SIZE = 100
        LARGE_SIZE = 80000

        np.random.seed(42)

        # Float-64
        self.A_d = np.random.uniform( -1.0, 1.0, size=LARGE_SIZE )
        self.B_d = np.random.uniform( -1.0, 1.0, size=LARGE_SIZE )
        self.C_d = np.random.uniform( -1.0, 1.0, size=LARGE_SIZE )
        # Float-32
        self.A_f = self.A_d.astype('float32')
        self.B_f = self.B_d.astype('float32')
        self.C_f = self.C_d.astype('float32')

        if os.name == 'nt':
            # Int-64
            self.A_q = np.random.randint( -100, high=100, size=LARGE_SIZE ).astype('int64')
            self.B_q = np.random.randint( -100, high=100, size=LARGE_SIZE ).astype('int64')
            self.C_q = np.random.randint( -100, high=100, size=LARGE_SIZE ).astype('int64')
            # Int-32
            self.A_l = self.A_q.astype('int32')
            self.B_l = self.B_q.astype('int32')
            self.C_l = self.C_q.astype('int32')
            # UInt-64
            self.A_Q = self.A_q.astype('uint64')
            self.B_Q = self.B_q.astype('uint64')
            self.C_Q = self.C_q.astype('uint64')
            # UInt-32
            self.A_L = self.A_q.astype('uint32')
            self.B_L = self.B_q.astype('uint32')
            self.C_L = self.C_q.astype('uint32')

        else:
            # Int-64
            self.A_l = np.random.randint( -100, high=100, size=LARGE_SIZE ).astype('int64')
            self.B_l = np.random.randint( -100, high=100, size=LARGE_SIZE ).astype('int64')
            self.C_l = np.random.randint( -100, high=100, size=LARGE_SIZE ).astype('int64')
            # Int-32
            self.A_i = self.A_l.astype('int32')
            self.B_i = self.B_l.astype('int32')
            self.C_i = self.C_l.astype('int32')
            # UInt-64
            self.A_L = self.A_l.astype('uint64')
            self.B_L = self.B_l.astype('uint64')
            self.C_L = self.C_l.astype('uint64')
            # UInt-32
            self.A_I = self.A_l.astype('uint32')
            self.B_I = self.B_l.astype('uint32')
            self.C_I = self.C_l.astype('uint32')

        # Int-16
        self.A_h = self.A_l.astype('int16')
        self.B_h = self.B_l.astype('int16')
        self.C_h = self.C_l.astype('int16')
        # UInt-16
        self.A_H = self.A_l.astype('uint16')
        self.B_H = self.B_l.astype('uint16')
        self.C_H = self.C_l.astype('uint16')

        # Int-8
        self.A_b = self.A_l.astype('int8')
        self.B_b = self.B_l.astype('int8')
        self.C_b = self.C_l.astype('int8')
        # UInt-8
        self.A_B = self.A_l.astype('uint8')
        self.B_B = self.B_l.astype('uint8')
        self.C_B = self.C_l.astype('uint8')
        # Bool
        self.A_1 = (self.A_l > 50).astype('bool')
        self.B_1 = (self.B_l > 50).astype('bool')
        self.C_1 = (self.C_l > 50).astype('bool')
        # Complex-64
        self.A_F = self.A_f + 1j*self.B_f
        self.B_F = self.B_f + 1j*self.C_f
        self.C_F =self. C_f + 1j*self.A_f
        # Complex-128
        self.A_D = self.A_d + 1j*self.B_d
        self.B_D = self.B_d + 1j*self.C_d
        self.C_D = self.C_d + 1j*self.A_d
        pass # End of setup
    
#GENERATOR_INSERT_POINT
    

def run():
    from numexpr3 import __version__
    print( "NumExpr3 auto-test for {} ".format(__version__) )
    unittest.main( exit=False )
    
if __name__ == "__main__":
    # Should generally call "python -m unittest -v numexpr3.test" for continuous integration
    run()
    
    
    