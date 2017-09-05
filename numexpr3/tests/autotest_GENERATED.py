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
        self.A_d = np.random.uniform( -1.0, 1.0, size=SMALL_SIZE )
        self.B_d = np.random.uniform( -1.0, 1.0, size=SMALL_SIZE )
        self.C_d = np.random.uniform( -1.0, 1.0, size=SMALL_SIZE )
        # Float-32
        self.A_f = self.A_d.astype('float32')
        self.B_f = self.B_d.astype('float32')
        self.C_f = self.C_d.astype('float32')

        if os.name == 'nt':
            # Int-64
            self.A_q = np.random.randint( -100, high=100, size=SMALL_SIZE ).astype('int64')
            self.B_q = np.random.randint( -100, high=100, size=SMALL_SIZE ).astype('int64')
            self.C_q = np.random.randint( -100, high=100, size=SMALL_SIZE ).astype('int64')
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
            self.A_l = np.random.randint( -100, high=100, size=SMALL_SIZE ).astype('int64')
            self.B_l = np.random.randint( -100, high=100, size=SMALL_SIZE ).astype('int64')
            self.C_l = np.random.randint( -100, high=100, size=SMALL_SIZE ).astype('int64')
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
    
    def test_add_111(self):
        out = ne3.NumExpr('self.A_1 + self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 + self.B_1)
    def test_add_bbb(self):
        out = ne3.NumExpr('self.A_b + self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b + self.B_b)
    def test_add_hhh(self):
        out = ne3.NumExpr('self.A_h + self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h + self.B_h)
    def test_add_lll(self):
        out = ne3.NumExpr('self.A_l + self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l + self.B_l)
    def test_add_qqq(self):
        out = ne3.NumExpr('self.A_q + self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q + self.B_q)
    def test_add_BBB(self):
        out = ne3.NumExpr('self.A_B + self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B + self.B_B)
    def test_add_HHH(self):
        out = ne3.NumExpr('self.A_H + self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H + self.B_H)
    def test_add_LLL(self):
        out = ne3.NumExpr('self.A_L + self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L + self.B_L)
    def test_add_QQQ(self):
        out = ne3.NumExpr('self.A_Q + self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q + self.B_Q)
    def test_add_fff(self):
        out = ne3.NumExpr('self.A_f + self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f + self.B_f)
    def test_add_ddd(self):
        out = ne3.NumExpr('self.A_d + self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d + self.B_d)
    def test_sub_bbb(self):
        out = ne3.NumExpr('self.A_b - self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b - self.B_b)
    def test_sub_hhh(self):
        out = ne3.NumExpr('self.A_h - self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h - self.B_h)
    def test_sub_lll(self):
        out = ne3.NumExpr('self.A_l - self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l - self.B_l)
    def test_sub_qqq(self):
        out = ne3.NumExpr('self.A_q - self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q - self.B_q)
    def test_sub_BBB(self):
        out = ne3.NumExpr('self.A_B - self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B - self.B_B)
    def test_sub_HHH(self):
        out = ne3.NumExpr('self.A_H - self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H - self.B_H)
    def test_sub_LLL(self):
        out = ne3.NumExpr('self.A_L - self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L - self.B_L)
    def test_sub_QQQ(self):
        out = ne3.NumExpr('self.A_Q - self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q - self.B_Q)
    def test_sub_fff(self):
        out = ne3.NumExpr('self.A_f - self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f - self.B_f)
    def test_sub_ddd(self):
        out = ne3.NumExpr('self.A_d - self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d - self.B_d)
    def test_mult_111(self):
        out = ne3.NumExpr('self.A_1 * self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 * self.B_1)
    def test_mult_bbb(self):
        out = ne3.NumExpr('self.A_b * self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b * self.B_b)
    def test_mult_hhh(self):
        out = ne3.NumExpr('self.A_h * self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h * self.B_h)
    def test_mult_lll(self):
        out = ne3.NumExpr('self.A_l * self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l * self.B_l)
    def test_mult_qqq(self):
        out = ne3.NumExpr('self.A_q * self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q * self.B_q)
    def test_mult_BBB(self):
        out = ne3.NumExpr('self.A_B * self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B * self.B_B)
    def test_mult_HHH(self):
        out = ne3.NumExpr('self.A_H * self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H * self.B_H)
    def test_mult_LLL(self):
        out = ne3.NumExpr('self.A_L * self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L * self.B_L)
    def test_mult_QQQ(self):
        out = ne3.NumExpr('self.A_Q * self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q * self.B_Q)
    def test_mult_fff(self):
        out = ne3.NumExpr('self.A_f * self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f * self.B_f)
    def test_mult_ddd(self):
        out = ne3.NumExpr('self.A_d * self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d * self.B_d)
    def test_div_d11(self):
        out = ne3.NumExpr('self.A_1 / self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 / self.B_1)
    def test_div_dbb(self):
        out = ne3.NumExpr('self.A_b / self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b / self.B_b)
    def test_div_dhh(self):
        out = ne3.NumExpr('self.A_h / self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h / self.B_h)
    def test_div_dll(self):
        out = ne3.NumExpr('self.A_l / self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l / self.B_l)
    def test_div_dqq(self):
        out = ne3.NumExpr('self.A_q / self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q / self.B_q)
    def test_div_dBB(self):
        out = ne3.NumExpr('self.A_B / self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B / self.B_B)
    def test_div_dHH(self):
        out = ne3.NumExpr('self.A_H / self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H / self.B_H)
    def test_div_dLL(self):
        out = ne3.NumExpr('self.A_L / self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L / self.B_L)
    def test_div_dQQ(self):
        out = ne3.NumExpr('self.A_Q / self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q / self.B_Q)
    def test_div_fff(self):
        out = ne3.NumExpr('self.A_f / self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f / self.B_f)
    def test_div_ddd(self):
        out = ne3.NumExpr('self.A_d / self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d / self.B_d)
    def test_pow_fff(self):
        out = ne3.NumExpr('self.A_f ** self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f ** self.B_f)
    def test_pow_ddd(self):
        out = ne3.NumExpr('self.A_d ** self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d ** self.B_d)
    def test_mod_fff(self):
        out = ne3.NumExpr('self.A_f % self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f % self.B_f)
    def test_mod_ddd(self):
        out = ne3.NumExpr('self.A_d % self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d % self.B_d)
    def test_where_1111(self):
        out = ne3.NumExpr('where( self.A_1, self.B_1, self.C_1 )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_1, self.C_1 ))
    def test_where_b1bb(self):
        out = ne3.NumExpr('where( self.A_1, self.B_b, self.C_b )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_b, self.C_b ))
    def test_where_h1hh(self):
        out = ne3.NumExpr('where( self.A_1, self.B_h, self.C_h )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_h, self.C_h ))
    def test_where_l1ll(self):
        out = ne3.NumExpr('where( self.A_1, self.B_l, self.C_l )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_l, self.C_l ))
    def test_where_q1qq(self):
        out = ne3.NumExpr('where( self.A_1, self.B_q, self.C_q )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_q, self.C_q ))
    def test_where_B1BB(self):
        out = ne3.NumExpr('where( self.A_1, self.B_B, self.C_B )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_B, self.C_B ))
    def test_where_H1HH(self):
        out = ne3.NumExpr('where( self.A_1, self.B_H, self.C_H )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_H, self.C_H ))
    def test_where_L1LL(self):
        out = ne3.NumExpr('where( self.A_1, self.B_L, self.C_L )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_L, self.C_L ))
    def test_where_Q1QQ(self):
        out = ne3.NumExpr('where( self.A_1, self.B_Q, self.C_Q )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_Q, self.C_Q ))
    def test_where_f1ff(self):
        out = ne3.NumExpr('where( self.A_1, self.B_f, self.C_f )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_f, self.C_f ))
    def test_where_d1dd(self):
        out = ne3.NumExpr('where( self.A_1, self.B_d, self.C_d )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_d, self.C_d ))
    def test_where_F1FF(self):
        out = ne3.NumExpr('where( self.A_1, self.B_F, self.C_F )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_F, self.C_F ))
    def test_where_D1DD(self):
        out = ne3.NumExpr('where( self.A_1, self.B_D, self.C_D )')()
        np.testing.assert_array_almost_equal(out,np.where( self.A_1, self.B_D, self.C_D ))
    def test_ones_like_11(self):
        out = ne3.NumExpr('ones_like( self.A_1 )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_1 ))
    def test_ones_like_bb(self):
        out = ne3.NumExpr('ones_like( self.A_b )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_b ))
    def test_ones_like_hh(self):
        out = ne3.NumExpr('ones_like( self.A_h )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_h ))
    def test_ones_like_ll(self):
        out = ne3.NumExpr('ones_like( self.A_l )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_l ))
    def test_ones_like_qq(self):
        out = ne3.NumExpr('ones_like( self.A_q )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_q ))
    def test_ones_like_BB(self):
        out = ne3.NumExpr('ones_like( self.A_B )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_B ))
    def test_ones_like_HH(self):
        out = ne3.NumExpr('ones_like( self.A_H )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_H ))
    def test_ones_like_LL(self):
        out = ne3.NumExpr('ones_like( self.A_L )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_L ))
    def test_ones_like_QQ(self):
        out = ne3.NumExpr('ones_like( self.A_Q )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_Q ))
    def test_ones_like_ff(self):
        out = ne3.NumExpr('ones_like( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_f ))
    def test_ones_like_dd(self):
        out = ne3.NumExpr('ones_like( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.ones_like( self.A_d ))
    def test_lshift_bbb(self):
        out = ne3.NumExpr('self.A_b << self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b << self.B_b)
    def test_lshift_hhh(self):
        out = ne3.NumExpr('self.A_h << self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h << self.B_h)
    def test_lshift_lll(self):
        out = ne3.NumExpr('self.A_l << self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l << self.B_l)
    def test_lshift_qqq(self):
        out = ne3.NumExpr('self.A_q << self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q << self.B_q)
    def test_lshift_BBB(self):
        out = ne3.NumExpr('self.A_B << self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B << self.B_B)
    def test_lshift_HHH(self):
        out = ne3.NumExpr('self.A_H << self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H << self.B_H)
    def test_lshift_LLL(self):
        out = ne3.NumExpr('self.A_L << self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L << self.B_L)
    def test_lshift_QQQ(self):
        out = ne3.NumExpr('self.A_Q << self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q << self.B_Q)
    def test_rshift_bbb(self):
        out = ne3.NumExpr('self.A_b >> self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b >> self.B_b)
    def test_rshift_hhh(self):
        out = ne3.NumExpr('self.A_h >> self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h >> self.B_h)
    def test_rshift_lll(self):
        out = ne3.NumExpr('self.A_l >> self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l >> self.B_l)
    def test_rshift_qqq(self):
        out = ne3.NumExpr('self.A_q >> self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q >> self.B_q)
    def test_rshift_BBB(self):
        out = ne3.NumExpr('self.A_B >> self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B >> self.B_B)
    def test_rshift_HHH(self):
        out = ne3.NumExpr('self.A_H >> self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H >> self.B_H)
    def test_rshift_LLL(self):
        out = ne3.NumExpr('self.A_L >> self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L >> self.B_L)
    def test_rshift_QQQ(self):
        out = ne3.NumExpr('self.A_Q >> self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q >> self.B_Q)
    def test_bitand_111(self):
        out = ne3.NumExpr('self.A_1 & self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 & self.B_1)
    def test_bitand_bbb(self):
        out = ne3.NumExpr('self.A_b & self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b & self.B_b)
    def test_bitand_hhh(self):
        out = ne3.NumExpr('self.A_h & self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h & self.B_h)
    def test_bitand_lll(self):
        out = ne3.NumExpr('self.A_l & self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l & self.B_l)
    def test_bitand_qqq(self):
        out = ne3.NumExpr('self.A_q & self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q & self.B_q)
    def test_bitand_BBB(self):
        out = ne3.NumExpr('self.A_B & self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B & self.B_B)
    def test_bitand_HHH(self):
        out = ne3.NumExpr('self.A_H & self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H & self.B_H)
    def test_bitand_LLL(self):
        out = ne3.NumExpr('self.A_L & self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L & self.B_L)
    def test_bitand_QQQ(self):
        out = ne3.NumExpr('self.A_Q & self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q & self.B_Q)
    def test_bitor_111(self):
        out = ne3.NumExpr('self.A_1 | self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 | self.B_1)
    def test_bitor_bbb(self):
        out = ne3.NumExpr('self.A_b | self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b | self.B_b)
    def test_bitor_hhh(self):
        out = ne3.NumExpr('self.A_h | self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h | self.B_h)
    def test_bitor_lll(self):
        out = ne3.NumExpr('self.A_l | self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l | self.B_l)
    def test_bitor_qqq(self):
        out = ne3.NumExpr('self.A_q | self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q | self.B_q)
    def test_bitor_BBB(self):
        out = ne3.NumExpr('self.A_B | self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B | self.B_B)
    def test_bitor_HHH(self):
        out = ne3.NumExpr('self.A_H | self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H | self.B_H)
    def test_bitor_LLL(self):
        out = ne3.NumExpr('self.A_L | self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L | self.B_L)
    def test_bitor_QQQ(self):
        out = ne3.NumExpr('self.A_Q | self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q | self.B_Q)
    def test_bitxor_111(self):
        out = ne3.NumExpr('self.A_1 ^ self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 ^ self.B_1)
    def test_bitxor_bbb(self):
        out = ne3.NumExpr('self.A_b ^ self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b ^ self.B_b)
    def test_bitxor_hhh(self):
        out = ne3.NumExpr('self.A_h ^ self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h ^ self.B_h)
    def test_bitxor_lll(self):
        out = ne3.NumExpr('self.A_l ^ self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l ^ self.B_l)
    def test_bitxor_qqq(self):
        out = ne3.NumExpr('self.A_q ^ self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q ^ self.B_q)
    def test_bitxor_BBB(self):
        out = ne3.NumExpr('self.A_B ^ self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B ^ self.B_B)
    def test_bitxor_HHH(self):
        out = ne3.NumExpr('self.A_H ^ self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H ^ self.B_H)
    def test_bitxor_LLL(self):
        out = ne3.NumExpr('self.A_L ^ self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L ^ self.B_L)
    def test_bitxor_QQQ(self):
        out = ne3.NumExpr('self.A_Q ^ self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q ^ self.B_Q)
    def test_logical_and_111(self):
        out = ne3.NumExpr('logical_and( self.A_1, self.B_1 )')()
        np.testing.assert_array_almost_equal(out,np.logical_and( self.A_1, self.B_1 ))
    def test_logical_or_111(self):
        out = ne3.NumExpr('logical_or( self.A_1, self.B_1 )')()
        np.testing.assert_array_almost_equal(out,np.logical_or( self.A_1, self.B_1 ))
    def test_gt_111(self):
        out = ne3.NumExpr('self.A_1 > self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 > self.B_1)
    def test_gt_1bb(self):
        out = ne3.NumExpr('self.A_b > self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b > self.B_b)
    def test_gt_1hh(self):
        out = ne3.NumExpr('self.A_h > self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h > self.B_h)
    def test_gt_1ll(self):
        out = ne3.NumExpr('self.A_l > self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l > self.B_l)
    def test_gt_1qq(self):
        out = ne3.NumExpr('self.A_q > self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q > self.B_q)
    def test_gt_1BB(self):
        out = ne3.NumExpr('self.A_B > self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B > self.B_B)
    def test_gt_1HH(self):
        out = ne3.NumExpr('self.A_H > self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H > self.B_H)
    def test_gt_1LL(self):
        out = ne3.NumExpr('self.A_L > self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L > self.B_L)
    def test_gt_1QQ(self):
        out = ne3.NumExpr('self.A_Q > self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q > self.B_Q)
    def test_gt_1ff(self):
        out = ne3.NumExpr('self.A_f > self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f > self.B_f)
    def test_gt_1dd(self):
        out = ne3.NumExpr('self.A_d > self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d > self.B_d)
    def test_gte_111(self):
        out = ne3.NumExpr('self.A_1 >= self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 >= self.B_1)
    def test_gte_1bb(self):
        out = ne3.NumExpr('self.A_b >= self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b >= self.B_b)
    def test_gte_1hh(self):
        out = ne3.NumExpr('self.A_h >= self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h >= self.B_h)
    def test_gte_1ll(self):
        out = ne3.NumExpr('self.A_l >= self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l >= self.B_l)
    def test_gte_1qq(self):
        out = ne3.NumExpr('self.A_q >= self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q >= self.B_q)
    def test_gte_1BB(self):
        out = ne3.NumExpr('self.A_B >= self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B >= self.B_B)
    def test_gte_1HH(self):
        out = ne3.NumExpr('self.A_H >= self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H >= self.B_H)
    def test_gte_1LL(self):
        out = ne3.NumExpr('self.A_L >= self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L >= self.B_L)
    def test_gte_1QQ(self):
        out = ne3.NumExpr('self.A_Q >= self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q >= self.B_Q)
    def test_gte_1ff(self):
        out = ne3.NumExpr('self.A_f >= self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f >= self.B_f)
    def test_gte_1dd(self):
        out = ne3.NumExpr('self.A_d >= self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d >= self.B_d)
    def test_lt_111(self):
        out = ne3.NumExpr('self.A_1 < self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 < self.B_1)
    def test_lt_1bb(self):
        out = ne3.NumExpr('self.A_b < self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b < self.B_b)
    def test_lt_1hh(self):
        out = ne3.NumExpr('self.A_h < self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h < self.B_h)
    def test_lt_1ll(self):
        out = ne3.NumExpr('self.A_l < self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l < self.B_l)
    def test_lt_1qq(self):
        out = ne3.NumExpr('self.A_q < self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q < self.B_q)
    def test_lt_1BB(self):
        out = ne3.NumExpr('self.A_B < self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B < self.B_B)
    def test_lt_1HH(self):
        out = ne3.NumExpr('self.A_H < self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H < self.B_H)
    def test_lt_1LL(self):
        out = ne3.NumExpr('self.A_L < self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L < self.B_L)
    def test_lt_1QQ(self):
        out = ne3.NumExpr('self.A_Q < self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q < self.B_Q)
    def test_lt_1ff(self):
        out = ne3.NumExpr('self.A_f < self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f < self.B_f)
    def test_lt_1dd(self):
        out = ne3.NumExpr('self.A_d < self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d < self.B_d)
    def test_lte_111(self):
        out = ne3.NumExpr('self.A_1 <= self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 <= self.B_1)
    def test_lte_1bb(self):
        out = ne3.NumExpr('self.A_b <= self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b <= self.B_b)
    def test_lte_1hh(self):
        out = ne3.NumExpr('self.A_h <= self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h <= self.B_h)
    def test_lte_1ll(self):
        out = ne3.NumExpr('self.A_l <= self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l <= self.B_l)
    def test_lte_1qq(self):
        out = ne3.NumExpr('self.A_q <= self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q <= self.B_q)
    def test_lte_1BB(self):
        out = ne3.NumExpr('self.A_B <= self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B <= self.B_B)
    def test_lte_1HH(self):
        out = ne3.NumExpr('self.A_H <= self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H <= self.B_H)
    def test_lte_1LL(self):
        out = ne3.NumExpr('self.A_L <= self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L <= self.B_L)
    def test_lte_1QQ(self):
        out = ne3.NumExpr('self.A_Q <= self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q <= self.B_Q)
    def test_lte_1ff(self):
        out = ne3.NumExpr('self.A_f <= self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f <= self.B_f)
    def test_lte_1dd(self):
        out = ne3.NumExpr('self.A_d <= self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d <= self.B_d)
    def test_eq_111(self):
        out = ne3.NumExpr('self.A_1 == self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 == self.B_1)
    def test_eq_1bb(self):
        out = ne3.NumExpr('self.A_b == self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b == self.B_b)
    def test_eq_1hh(self):
        out = ne3.NumExpr('self.A_h == self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h == self.B_h)
    def test_eq_1ll(self):
        out = ne3.NumExpr('self.A_l == self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l == self.B_l)
    def test_eq_1qq(self):
        out = ne3.NumExpr('self.A_q == self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q == self.B_q)
    def test_eq_1BB(self):
        out = ne3.NumExpr('self.A_B == self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B == self.B_B)
    def test_eq_1HH(self):
        out = ne3.NumExpr('self.A_H == self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H == self.B_H)
    def test_eq_1LL(self):
        out = ne3.NumExpr('self.A_L == self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L == self.B_L)
    def test_eq_1QQ(self):
        out = ne3.NumExpr('self.A_Q == self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q == self.B_Q)
    def test_eq_1ff(self):
        out = ne3.NumExpr('self.A_f == self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f == self.B_f)
    def test_eq_1dd(self):
        out = ne3.NumExpr('self.A_d == self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d == self.B_d)
    def test_noteq_111(self):
        out = ne3.NumExpr('self.A_1 != self.B_1')()
        np.testing.assert_array_almost_equal(out,self.A_1 != self.B_1)
    def test_noteq_1bb(self):
        out = ne3.NumExpr('self.A_b != self.B_b')()
        np.testing.assert_array_almost_equal(out,self.A_b != self.B_b)
    def test_noteq_1hh(self):
        out = ne3.NumExpr('self.A_h != self.B_h')()
        np.testing.assert_array_almost_equal(out,self.A_h != self.B_h)
    def test_noteq_1ll(self):
        out = ne3.NumExpr('self.A_l != self.B_l')()
        np.testing.assert_array_almost_equal(out,self.A_l != self.B_l)
    def test_noteq_1qq(self):
        out = ne3.NumExpr('self.A_q != self.B_q')()
        np.testing.assert_array_almost_equal(out,self.A_q != self.B_q)
    def test_noteq_1BB(self):
        out = ne3.NumExpr('self.A_B != self.B_B')()
        np.testing.assert_array_almost_equal(out,self.A_B != self.B_B)
    def test_noteq_1HH(self):
        out = ne3.NumExpr('self.A_H != self.B_H')()
        np.testing.assert_array_almost_equal(out,self.A_H != self.B_H)
    def test_noteq_1LL(self):
        out = ne3.NumExpr('self.A_L != self.B_L')()
        np.testing.assert_array_almost_equal(out,self.A_L != self.B_L)
    def test_noteq_1QQ(self):
        out = ne3.NumExpr('self.A_Q != self.B_Q')()
        np.testing.assert_array_almost_equal(out,self.A_Q != self.B_Q)
    def test_noteq_1ff(self):
        out = ne3.NumExpr('self.A_f != self.B_f')()
        np.testing.assert_array_almost_equal(out,self.A_f != self.B_f)
    def test_noteq_1dd(self):
        out = ne3.NumExpr('self.A_d != self.B_d')()
        np.testing.assert_array_almost_equal(out,self.A_d != self.B_d)
    def test_abs_bb(self):
        out = ne3.NumExpr('abs( self.A_b )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_b ))
    def test_abs_hh(self):
        out = ne3.NumExpr('abs( self.A_h )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_h ))
    def test_abs_ll(self):
        out = ne3.NumExpr('abs( self.A_l )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_l ))
    def test_abs_qq(self):
        out = ne3.NumExpr('abs( self.A_q )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_q ))
    def test_abs_ff(self):
        out = ne3.NumExpr('abs( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_f ))
    def test_abs_dd(self):
        out = ne3.NumExpr('abs( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_d ))
    def test_arccos_ff(self):
        out = ne3.NumExpr('arccos( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.arccos( self.A_f ))
    def test_arccos_dd(self):
        out = ne3.NumExpr('arccos( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.arccos( self.A_d ))
    def test_arcsin_ff(self):
        out = ne3.NumExpr('arcsin( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.arcsin( self.A_f ))
    def test_arcsin_dd(self):
        out = ne3.NumExpr('arcsin( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.arcsin( self.A_d ))
    def test_arctan_ff(self):
        out = ne3.NumExpr('arctan( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.arctan( self.A_f ))
    def test_arctan_dd(self):
        out = ne3.NumExpr('arctan( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.arctan( self.A_d ))
    def test_arctan2_fff(self):
        out = ne3.NumExpr('arctan2( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,np.arctan2( self.A_f, self.B_f ))
    def test_arctan2_ddd(self):
        out = ne3.NumExpr('arctan2( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,np.arctan2( self.A_d, self.B_d ))
    def test_ceil_ff(self):
        out = ne3.NumExpr('ceil( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.ceil( self.A_f ))
    def test_ceil_dd(self):
        out = ne3.NumExpr('ceil( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.ceil( self.A_d ))
    def test_cos_ff(self):
        out = ne3.NumExpr('cos( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.cos( self.A_f ))
    def test_cos_dd(self):
        out = ne3.NumExpr('cos( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.cos( self.A_d ))
    def test_cosh_ff(self):
        out = ne3.NumExpr('cosh( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.cosh( self.A_f ))
    def test_cosh_dd(self):
        out = ne3.NumExpr('cosh( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.cosh( self.A_d ))
    def test_exp_ff(self):
        out = ne3.NumExpr('exp( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.exp( self.A_f ))
    def test_exp_dd(self):
        out = ne3.NumExpr('exp( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.exp( self.A_d ))
    def test_fabs_ff(self):
        out = ne3.NumExpr('fabs( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.fabs( self.A_f ))
    def test_fabs_dd(self):
        out = ne3.NumExpr('fabs( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.fabs( self.A_d ))
    def test_floor_ff(self):
        out = ne3.NumExpr('floor( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.floor( self.A_f ))
    def test_floor_dd(self):
        out = ne3.NumExpr('floor( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.floor( self.A_d ))
    def test_fmod_fff(self):
        out = ne3.NumExpr('fmod( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,np.fmod( self.A_f, self.B_f ))
    def test_fmod_ddd(self):
        out = ne3.NumExpr('fmod( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,np.fmod( self.A_d, self.B_d ))
    def test_log_ff(self):
        out = ne3.NumExpr('log( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.log( self.A_f ))
    def test_log_dd(self):
        out = ne3.NumExpr('log( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.log( self.A_d ))
    def test_log10_ff(self):
        out = ne3.NumExpr('log10( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.log10( self.A_f ))
    def test_log10_dd(self):
        out = ne3.NumExpr('log10( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.log10( self.A_d ))
    def test_sin_ff(self):
        out = ne3.NumExpr('sin( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.sin( self.A_f ))
    def test_sin_dd(self):
        out = ne3.NumExpr('sin( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.sin( self.A_d ))
    def test_sinh_ff(self):
        out = ne3.NumExpr('sinh( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.sinh( self.A_f ))
    def test_sinh_dd(self):
        out = ne3.NumExpr('sinh( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.sinh( self.A_d ))
    def test_sqrt_ff(self):
        out = ne3.NumExpr('sqrt( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.sqrt( self.A_f ))
    def test_sqrt_dd(self):
        out = ne3.NumExpr('sqrt( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.sqrt( self.A_d ))
    def test_tan_ff(self):
        out = ne3.NumExpr('tan( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.tan( self.A_f ))
    def test_tan_dd(self):
        out = ne3.NumExpr('tan( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.tan( self.A_d ))
    def test_tanh_ff(self):
        out = ne3.NumExpr('tanh( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.tanh( self.A_f ))
    def test_tanh_dd(self):
        out = ne3.NumExpr('tanh( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.tanh( self.A_d ))
    def test_isfinite_1f(self):
        out = ne3.NumExpr('isfinite( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.isfinite( self.A_f ))
    def test_isfinite_1d(self):
        out = ne3.NumExpr('isfinite( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.isfinite( self.A_d ))
    def test_isinf_1f(self):
        out = ne3.NumExpr('isinf( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.isinf( self.A_f ))
    def test_isinf_1d(self):
        out = ne3.NumExpr('isinf( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.isinf( self.A_d ))
    def test_isnan_1f(self):
        out = ne3.NumExpr('isnan( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.isnan( self.A_f ))
    def test_isnan_1d(self):
        out = ne3.NumExpr('isnan( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.isnan( self.A_d ))
    def test_signbit_1f(self):
        out = ne3.NumExpr('signbit( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.signbit( self.A_f ))
    def test_signbit_1d(self):
        out = ne3.NumExpr('signbit( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.signbit( self.A_d ))
    def test_arccosh_ff(self):
        out = ne3.NumExpr('arccosh( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.arccosh( self.A_f ))
    def test_arccosh_dd(self):
        out = ne3.NumExpr('arccosh( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.arccosh( self.A_d ))
    def test_arcsinh_ff(self):
        out = ne3.NumExpr('arcsinh( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.arcsinh( self.A_f ))
    def test_arcsinh_dd(self):
        out = ne3.NumExpr('arcsinh( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.arcsinh( self.A_d ))
    def test_arctanh_ff(self):
        out = ne3.NumExpr('arctanh( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.arctanh( self.A_f ))
    def test_arctanh_dd(self):
        out = ne3.NumExpr('arctanh( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.arctanh( self.A_d ))
    def test_cbrt_ff(self):
        out = ne3.NumExpr('cbrt( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.cbrt( self.A_f ))
    def test_cbrt_dd(self):
        out = ne3.NumExpr('cbrt( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.cbrt( self.A_d ))
    def test_copysign_fff(self):
        out = ne3.NumExpr('copysign( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,np.copysign( self.A_f, self.B_f ))
    def test_copysign_ddd(self):
        out = ne3.NumExpr('copysign( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,np.copysign( self.A_d, self.B_d ))
    def test_exp2_ff(self):
        out = ne3.NumExpr('exp2( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.exp2( self.A_f ))
    def test_exp2_dd(self):
        out = ne3.NumExpr('exp2( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.exp2( self.A_d ))
    def test_expm1_ff(self):
        out = ne3.NumExpr('expm1( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.expm1( self.A_f ))
    def test_expm1_dd(self):
        out = ne3.NumExpr('expm1( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.expm1( self.A_d ))
    def test_fmax_fff(self):
        out = ne3.NumExpr('fmax( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,np.fmax( self.A_f, self.B_f ))
    def test_fmax_ddd(self):
        out = ne3.NumExpr('fmax( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,np.fmax( self.A_d, self.B_d ))
    def test_fmin_fff(self):
        out = ne3.NumExpr('fmin( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,np.fmin( self.A_f, self.B_f ))
    def test_fmin_ddd(self):
        out = ne3.NumExpr('fmin( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,np.fmin( self.A_d, self.B_d ))
    def test_hypot_fff(self):
        out = ne3.NumExpr('hypot( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,np.hypot( self.A_f, self.B_f ))
    def test_hypot_ddd(self):
        out = ne3.NumExpr('hypot( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,np.hypot( self.A_d, self.B_d ))
    def test_log1p_ff(self):
        out = ne3.NumExpr('log1p( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.log1p( self.A_f ))
    def test_log1p_dd(self):
        out = ne3.NumExpr('log1p( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.log1p( self.A_d ))
    def test_log2_ff(self):
        out = ne3.NumExpr('log2( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.log2( self.A_f ))
    def test_log2_dd(self):
        out = ne3.NumExpr('log2( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.log2( self.A_d ))
    def test_nextafter_fff(self):
        out = ne3.NumExpr('nextafter( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,np.nextafter( self.A_f, self.B_f ))
    def test_nextafter_ddd(self):
        out = ne3.NumExpr('nextafter( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,np.nextafter( self.A_d, self.B_d ))
    def test_rint_ff(self):
        out = ne3.NumExpr('rint( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.rint( self.A_f ))
    def test_rint_dd(self):
        out = ne3.NumExpr('rint( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.rint( self.A_d ))
    def test_round_lf(self):
        out = ne3.NumExpr('round( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.round( self.A_f ))
    def test_round_ld(self):
        out = ne3.NumExpr('round( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.round( self.A_d ))
    def test_trunc_ff(self):
        out = ne3.NumExpr('trunc( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.trunc( self.A_f ))
    def test_trunc_dd(self):
        out = ne3.NumExpr('trunc( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.trunc( self.A_d ))
    def test_complex_Fff(self):
        out = ne3.NumExpr('complex( self.A_f, self.B_f )')()
        np.testing.assert_array_almost_equal(out,self.A_f + 1j*self.B_f)
    def test_complex_Ddd(self):
        out = ne3.NumExpr('complex( self.A_d, self.B_d )')()
        np.testing.assert_array_almost_equal(out,self.A_d + 1j*self.B_d)
    def test_real_fF(self):
        out = ne3.NumExpr('real( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.real( self.A_F ))
    def test_real_dD(self):
        out = ne3.NumExpr('real( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.real( self.A_D ))
    def test_imag_fF(self):
        out = ne3.NumExpr('imag( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.imag( self.A_F ))
    def test_imag_dD(self):
        out = ne3.NumExpr('imag( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.imag( self.A_D ))
    def test_abs_fF(self):
        out = ne3.NumExpr('abs( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_F ))
    def test_abs_dD(self):
        out = ne3.NumExpr('abs( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.abs( self.A_D ))
    def test_add_FFF(self):
        out = ne3.NumExpr('self.A_F + self.B_F')()
        np.testing.assert_array_almost_equal(out,self.A_F + self.B_F)
    def test_add_DDD(self):
        out = ne3.NumExpr('self.A_D + self.B_D')()
        np.testing.assert_array_almost_equal(out,self.A_D + self.B_D)
    def test_sub_FFF(self):
        out = ne3.NumExpr('self.A_F - self.B_F')()
        np.testing.assert_array_almost_equal(out,self.A_F - self.B_F)
    def test_sub_DDD(self):
        out = ne3.NumExpr('self.A_D - self.B_D')()
        np.testing.assert_array_almost_equal(out,self.A_D - self.B_D)
    def test_mult_FFF(self):
        out = ne3.NumExpr('self.A_F * self.B_F')()
        np.testing.assert_array_almost_equal(out,self.A_F * self.B_F)
    def test_mult_DDD(self):
        out = ne3.NumExpr('self.A_D * self.B_D')()
        np.testing.assert_array_almost_equal(out,self.A_D * self.B_D)
    def test_div_FFF(self):
        out = ne3.NumExpr('self.A_F / self.B_F')()
        np.testing.assert_array_almost_equal(out,self.A_F / self.B_F,decimal=4)
    def test_div_DDD(self):
        out = ne3.NumExpr('self.A_D / self.B_D')()
        np.testing.assert_array_almost_equal(out,self.A_D / self.B_D)
    def test_neg_FF(self):
        out = ne3.NumExpr('-self.A_F')()
        np.testing.assert_array_almost_equal(out,-self.A_F)
    def test_neg_DD(self):
        out = ne3.NumExpr('-self.A_D')()
        np.testing.assert_array_almost_equal(out,-self.A_D)
    def test_conj_FF(self):
        out = ne3.NumExpr('conj( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.conj( self.A_F ))
    def test_conj_DD(self):
        out = ne3.NumExpr('conj( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.conj( self.A_D ))
    def test_conj_ff(self):
        out = ne3.NumExpr('conj( self.A_f )')()
        np.testing.assert_array_almost_equal(out,np.conj( self.A_f ))
    def test_conj_dd(self):
        out = ne3.NumExpr('conj( self.A_d )')()
        np.testing.assert_array_almost_equal(out,np.conj( self.A_d ))
    def test_sqrt_FF(self):
        out = ne3.NumExpr('sqrt( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.sqrt( self.A_F ))
    def test_sqrt_DD(self):
        out = ne3.NumExpr('sqrt( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.sqrt( self.A_D ))
    def test_log_FF(self):
        out = ne3.NumExpr('log( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.log( self.A_F ))
    def test_log_DD(self):
        out = ne3.NumExpr('log( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.log( self.A_D ))
    def test_log1p_FF(self):
        out = ne3.NumExpr('log1p( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.log1p( self.A_F ))
    def test_log1p_DD(self):
        out = ne3.NumExpr('log1p( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.log1p( self.A_D ))
    def test_log10_FF(self):
        out = ne3.NumExpr('log10( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.log10( self.A_F ))
    def test_log10_DD(self):
        out = ne3.NumExpr('log10( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.log10( self.A_D ))
    def test_exp_FF(self):
        out = ne3.NumExpr('exp( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.exp( self.A_F ))
    def test_exp_DD(self):
        out = ne3.NumExpr('exp( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.exp( self.A_D ))
    def test_expm1_FF(self):
        out = ne3.NumExpr('expm1( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.expm1( self.A_F ))
    def test_expm1_DD(self):
        out = ne3.NumExpr('expm1( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.expm1( self.A_D ))
    def test_pow_FFF(self):
        out = ne3.NumExpr('self.A_F ** self.B_F')()
        np.testing.assert_array_almost_equal(out,self.A_F ** self.B_F,decimal=4)
    def test_pow_DDD(self):
        out = ne3.NumExpr('self.A_D ** self.B_D')()
        np.testing.assert_array_almost_equal(out,self.A_D ** self.B_D)
    def test_arccos_FF(self):
        out = ne3.NumExpr('arccos( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.arccos( self.A_F ))
    def test_arccos_DD(self):
        out = ne3.NumExpr('arccos( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.arccos( self.A_D ))
    def test_arccosh_FF(self):
        out = ne3.NumExpr('arccosh( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.arccosh( self.A_F ))
    def test_arccosh_DD(self):
        out = ne3.NumExpr('arccosh( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.arccosh( self.A_D ))
    def test_arcsin_FF(self):
        out = ne3.NumExpr('arcsin( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.arcsin( self.A_F ))
    def test_arcsin_DD(self):
        out = ne3.NumExpr('arcsin( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.arcsin( self.A_D ))
    def test_arcsinh_FF(self):
        out = ne3.NumExpr('arcsinh( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.arcsinh( self.A_F ))
    def test_arcsinh_DD(self):
        out = ne3.NumExpr('arcsinh( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.arcsinh( self.A_D ))
    def test_arctan_FF(self):
        out = ne3.NumExpr('arctan( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.arctan( self.A_F ))
    def test_arctan_DD(self):
        out = ne3.NumExpr('arctan( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.arctan( self.A_D ))
    def test_arctanh_FF(self):
        out = ne3.NumExpr('arctanh( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.arctanh( self.A_F ))
    def test_arctanh_DD(self):
        out = ne3.NumExpr('arctanh( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.arctanh( self.A_D ))
    def test_cos_FF(self):
        out = ne3.NumExpr('cos( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.cos( self.A_F ))
    def test_cos_DD(self):
        out = ne3.NumExpr('cos( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.cos( self.A_D ))
    def test_cosh_FF(self):
        out = ne3.NumExpr('cosh( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.cosh( self.A_F ))
    def test_cosh_DD(self):
        out = ne3.NumExpr('cosh( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.cosh( self.A_D ))
    def test_sin_FF(self):
        out = ne3.NumExpr('sin( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.sin( self.A_F ))
    def test_sin_DD(self):
        out = ne3.NumExpr('sin( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.sin( self.A_D ))
    def test_sinh_FF(self):
        out = ne3.NumExpr('sinh( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.sinh( self.A_F ))
    def test_sinh_DD(self):
        out = ne3.NumExpr('sinh( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.sinh( self.A_D ))
    def test_tan_FF(self):
        out = ne3.NumExpr('tan( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.tan( self.A_F ))
    def test_tan_DD(self):
        out = ne3.NumExpr('tan( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.tan( self.A_D ))
    def test_tanh_FF(self):
        out = ne3.NumExpr('tanh( self.A_F )')()
        np.testing.assert_array_almost_equal(out,np.tanh( self.A_F ))
    def test_tanh_DD(self):
        out = ne3.NumExpr('tanh( self.A_D )')()
        np.testing.assert_array_almost_equal(out,np.tanh( self.A_D ))

    

def run():
    from numexpr3 import __version__
    print( "NumExpr3 auto-test for {} ".format(__version__) )
    unittest.main( exit=False )
    
if __name__ == "__main__":
    # Should generally call "python -m unittest -v numexpr3.test" for continuous integration
    run()
    
    
    