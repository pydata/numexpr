###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: BSD
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################
from __future__ import absolute_import, print_function

import sys
import platform

import numpy as np
import numpy.testing as npt
import numexpr3 as ne3
import unittest
import logging
import pickle
import time
logger = logging.getLogger('ne3.test')

import warnings
# Ignore RuntimeWarnings related to divide-by-zero, as we are intentionally 
# dooing divide-by-zero
warnings.filterwarnings("ignore")

# Recommended minimum versions
minimum_numpy_version = '1.7'

# Don't use powers of 2 for sizes as BLOCK_SIZEs are powers of 2, and we want 
# to test when we have sub-sized blocks in the last cycle through the program.
SMALL_SIZE = 100.0
LARGE_SIZE = 80000.0

class test_numexpr(unittest.TestCase):
    '''Testing with 1 thread for serial operation'''

    def setUp(self, N_threads=1, ssize=SMALL_SIZE):
        logger.info( "**Run NumExpr() tests with {} thread(s) over {} element arrays**".format(N_threads, ssize) )
        ne3.set_nthreads(N_threads)
        self.ssize = ssize

    def test_scalars(self):
        logger.info( 'Test scalars' )
        a = np.array([1., 2, 3])
        b = np.array([4., 5, 6])
        c = np.array([7., 8, 9])
        out = ne3.NumExpr( '2.0 * a + 3.0 * b * c' )()
        npt.assert_array_almost_equal( out, np.array([86., 124., 168.]))

    def test_changing_array_size(self):
        logger.info('Test input with keywords, changing input array size')
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])
        a2 = np.arange(self.ssize).astype(a.dtype)
        b2 = np.arange(3.0, self.ssize+3.0).astype(b.dtype)
        out = ne3.NumExpr( 'a*b' )(a=a2, b=b2)
        npt.assert_array_almost_equal( out, a2*b2 )

    def test_changing_array_shape(self):
        logger.info('Test changing input array shape')
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])
        a2 = np.arange(self.ssize).astype(a.dtype).reshape( int(self.ssize/20), 4, 5)
        b2 = np.arange(3.0, self.ssize+3.0).astype(b.dtype).reshape( int(self.ssize/20), 4, 5)
        out = ne3.NumExpr( 'a*b' )(a=a2, b=b2)
        npt.assert_array_almost_equal( out, a2*b2 )

    def test_verify_input(self):
        logger.info('Test input with verify=True')
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])
        func = ne3.NumExpr( 'a*b' )
        a = np.arange(self.ssize).astype(a.dtype)
        b = np.arange(3.0, self.ssize+3.0).astype(b.dtype)
        out = func(verify=True)
        npt.assert_array_almost_equal( out, a*b )

    def test_weakref_expiry(self):
        # I have no idea how to turn on gc in unittest. I can't find anything
        # in unittest that does that:
        # https://github.com/python/cpython/tree/master/Lib/unittest
        # This script works fine when run independantly.
        '''
        import gc
        gc.enable()
        logger.warning('Test expiry of weak reference')
        x = np.arange(self.ssize)
        func = ne3.NumExpr( 'x+x' )
        logging.warning( 'x is tracked: ' + str(gc.is_tracked(x) ) )
        del x  # kill original array, should expire weak ref in func.registers
        gc.collect(generation=2)
        gc.collect(generation=1)
        gc.collect(generation=0)
        logging.warning( "Garbage is {}".format(gc.garbage ) )
        # For some reason the garbage collection isn't performed inside unittest?
        x = np.arange(self.ssize) + 50
        out = func( verify=False )
        npt.assert_array_almost_equal( x+x, out )
        '''
        pass

    def test_copy_output(self):
        logger.info( 'Test copy output' )
        x = np.arange(self.ssize)
        out = ne3.NumExpr('x')()
        npt.assert_array_almost_equal(x, out)

    def test_copy_assign(self):
        logger.info( 'Test copy assignment' )
        x = np.arange(self.ssize)
        ne3.NumExpr('out=x')()
        npt.assert_array_almost_equal(x, locals()['out'] )

    def test_rational_func(self):
        logger.info( 'Test rational func' )
        a = np.arange(self.ssize)
        b = np.arange(self.ssize) * 0.1
        func = ne3.NumExpr( '(a + 2.0*b) / (1 + a + 4*b*b)' )
        x = (a + 2 * b) / (1 + a + 4 * b * b)
        y = func(a=a, b=b)
        npt.assert_array_almost_equal(x, y)

    def test_secondary_output(self):
        logger.info( 'Test two outputs' )
        y = np.arange(self.ssize)
        out1 = np.empty_like(y)
        out2 = np.empty_like(y)
        ne3.NumExpr( 'out1 = y + y; out2 = out1 + 4' )()
        npt.assert_array_almost_equal( y+y, out1 )
        npt.assert_array_almost_equal( y+y+4, out2 )

    def test_inplace(self):
        logger.info( 'Test in-place operation' )
        x = np.arange(self.ssize).reshape(int(self.ssize/10), 10)
        ne3.NumExpr('x = x + 3')()
        npt.assert_array_almost_equal(x, np.arange(self.ssize).reshape(int(self.ssize/10), 10) + 3)

    def test_inplace_intermediate(self):
        # When the return array is also a named intermediate assignment target
        print( 'Test in-place named intermediate' )
        y = np.arange(self.ssize)
        x = np.empty_like(y)
        ne3.NumExpr( 'x = 3.5 * y; x = x - y' )()
        npt.assert_array_almost_equal( x, (3.5*y) - y )

    def test_named_intermediate_magic_output(self):
        print( 'Test in-place named intermediate with magic output' )
        y = np.arange(self.ssize)
        ne3.NumExpr( 'x = 3.5 * y; x = x - y' )()
        npt.assert_array_almost_equal( locals()['x'], (3.5*y) - y )

    def test_changing_singleton(self):
        # When a single-valued array is changed (i.e. it should be KIND_ARRAY, not KIND_SCALAR)
        logger.info( 'Test changing singleton' )
        pi = np.pi
        y = np.arange(self.ssize)
        x = np.empty_like(y)
        func = ne3.NumExpr( 'x = y*pi' )
        func( x=x, y=y, pi=pi*2 )
        npt.assert_array_almost_equal( x, y*np.pi*2 )

    def test_simple_strides(self):
        # It may make more sense to do strided operations on all functions
        # in autotest_GENERATED
        logger.info( 'Test simple strides' )
        a = np.arange(self.ssize)[::3]
        b = np.arange(self.ssize,0,-1)[::3]
        out = ne3.NumExpr( 'a-b' )()
        npt.assert_array_almost_equal( out, a-b )

    def test_broadcasting_with_strides(self):
        logger.info( 'Test broadcasting with strides' )
        a = np.arange(100).reshape(10,10)[::2]
        c = np.arange(10)
        d = np.arange(5).reshape(5,1)
        # Test with just arrays
        neObj1 = ne3.NumExpr('a + c')
        npt.assert_array_equal( neObj1(), a + c )
        neObj2 = ne3.NumExpr('a + d')
        npt.assert_array_equal( neObj2(), a + d )
        # And with scalars
        neObj3 = ne3.NumExpr( '2*a + 3*c' )
        npt.assert_array_equal( neObj3(), 2*a + 3*c )

    def test_all_scalar(self):
        logger.info( 'Test all scalar: DEBUG' )
        import numpy as np
        a = 3.
        b = 4.
        expr = ne3.NumExpr('np.pi*(2*a+3*b)')
        npt.assert_allclose( expr(), np.pi*(2*a + 3*b) )

    def test_run_vs_call(self):
        logger.info( 'Compare run versus __call__' )
        a = np.arange(self.ssize)
        b = np.arange(self.ssize)
        expr = ne3.NumExpr('2*a+3*b')
        npt.assert_array_almost_equal( expr(a=a, b=b), expr.run(a=a, b=b) )

    def test_floor_div(self):
        logger.info( 'Test floor div3' )
        x = np.arange(self.ssize, dtype='int64')
        intSize = np.int64(self.ssize)
        y = ne3.NumExpr('intSize//x')()
        npt.assert_array_equal(y, intSize//x)
        
    def test_true_div(self):
        logger.info( 'Test true div' )
        x = np.arange(self.ssize, dtype='int64')
        intSize = np.int64(self.ssize)
        npt.assert_array_equal(ne3.NumExpr('intSize/x')(), intSize/x)

    def test_complex64(self):
        logger.info( 'Test complex64 ' )
        def complex64_func(a, b):
            c = np.zeros(a.shape, dtype='complex64')
            c.real = a; c.imag = b
            return c

        a = np.linspace( -1, 1, self.ssize, dtype='float32' )
        b = np.linspace( -1, 1, self.ssize, dtype='float32' )
        z = ( a + 1j * b ).astype( 'complex64' )
        x = z.imag
        x = np.sin(complex64_func(a,b)).real + z.imag
        func = ne3.NumExpr('sin(complex(a, b)).real + z.imag')
        npt.assert_array_almost_equal(x, func())
		
    def test_complex128(self):
        logger.info( 'Test complex128' )
        def complex_func(a, b):
            c = np.zeros(a.shape, dtype='complex128')
            c.real = a; c.imag = b
            return c

        a = np.linspace( -1, 1, self.ssize, dtype='float64' )
        b = np.linspace( -1, 1, self.ssize, dtype='float64' )
        z = a + 1j * b
        x = z.imag
        x = np.sin(complex_func(a, b)).real + z.imag
        ne3.NumExpr('y = sin(complex(a, b)).real + z.imag')()
        npt.assert_array_almost_equal(x, locals()['y'])

        
    def test_complex_strides(self):
        logger.info( 'Test complex strides' )
        a = np.linspace( -1, 1, self.ssize, dtype='float32' )
        b = np.linspace( -0.01, 0.01, self.ssize, dtype='float32' )
        z1 = (a + 1j * b)[::2]
        z2 = (a - 1j * b)[::2]
        ne3.evaluate('out = z1 + z2' )
        npt.assert_array_almost_equal( locals()['out'], z1+z2 )


    def test_list_literal(self):
        logger.info( 'Test list literal' )
        a = np.arange(3)
        out = ne3.NumExpr('a < [2,2,2]')()
        npt.assert_array_equal( out, a < np.array([2,2,2]) )

    def test_casting(self):
        logging.info( "Test safe, unsafe, and func helper casting")
        
        i = np.arange(self.ssize).astype('int32')
        # 'Safe' casts
        f = ne3.NumExpr('float32(i)')()
        # 'Unsafe' casts
        b = ne3.NumExpr('int8(i)')()
        assert( f.dtype == 'float32' )
        npt.assert_array_equal( i.astype('float32'), f )
        assert( b.dtype == 'int8')
        npt.assert_array_equal( i.astype('int8'), b )
        # Helper casts for functions
        sin_ne = ne3.NumExpr( 'sin(i)' )()
        sin_np = np.sin(i)
        assert( sin_np.dtype == sin_ne.dtype )
        npt.assert_array_almost_equal( sin_np, sin_ne )

    # Non-unit length multiple strides are not supported any more as it 
    # isn't supported with the SIMD auto-vectorization.ion.


class test_numexpr_max(test_numexpr):
    '''Testing with maximum threads and large arrays for parallel operation'''
    
    def setUp(self, N_threads=-1, ssize=LARGE_SIZE):
        if N_threads < 0:
            N_threads = ne3.get_ncores()
        logger.info( "**Run NumExpr() tests with {} thread(s) over {} element arrays**".format(N_threads, ssize) )
        ne3.set_nthreads(N_threads)
        self.ssize = ssize

class test_evaluate(unittest.TestCase):

    def setUp(self, N_threads=1, ssize=SMALL_SIZE):
        logger.info( "**Run evaluate() tests with {} thread(s) over {} element arrays**".format(N_threads, ssize) )
        ne3.set_nthreads(N_threads)
        self.ssize = ssize

    def test_eval_versus_run(self):
        logger.info( 'Compare evaluate and .run' )
        a = np.arange(100.0).reshape(10, 10)[::2]
        b = np.arange(10.0)
        expr = ne3.NumExpr('2*a+3*b')
        npt.assert_array_almost_equal(expr(a=a, b=b), expr.run(a=a, b=b))

    def test_prealloc(self):
        logger.info( 'Test pre-allocated output evaluate' )
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])
        c = np.array([7., 8., 9.])
        y = 2*a + 3*b*c
        x = np.empty_like(y)
        ne3.evaluate('x=2*a + 3*b*c')
        npt.assert_array_equal(x, y)

    def test_magic(self):
        logger.info( 'Test magic output evaluate' )
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])
        c = np.array([7., 8., 9.])
        y = 2*a + 3*b*c
        ne3.evaluate('x_magic=2*a + 3*b*c')
        # For some reason, only in unittest, does y_magic not appear in the 
        # scope unless explicitely looked for in locals. This works in normal
        # scripts however.
        npt.assert_array_equal( locals()['x_magic'], y)

    def test_copy(self):
        logger.info( 'Test copy evaluate with preallocated output' )
        x = np.arange(SMALL_SIZE)
        y = np.zeros_like(x)
        ne3.evaluate('y=x')
        npt.assert_array_equal(x, y)
        
    def test_copy_magic(self):
        logger.info( 'Test copy evaluate with magic output' )
        x = np.arange(self.ssize)
        ne3.evaluate('y_magic=x')
        npt.assert_array_equal(x, locals()['y_magic'] )

    def test_rational(self):
        logger.info( 'Test rational evaluation' )
        a = np.arange(1e5)
        b = np.arange(1e5) * 0.1
        x = (a + 2 * b) / (1 + a + 4 * b * b)
        ne3.evaluate('y=(a + 2*b) / (1 + a + 4*b*b)')
        npt.assert_array_almost_equal(x, locals()['y'] )
		
class test_reductions(unittest.TestCase):
    # NE3 reductions not implemented yet
    '''
    def test_reductions_DEPRECATED(self):
        # Check that they compile OK.
        assert_equal(disassemble(
            NumExpr('sum(x**2+2, axis=None)', [('x', double)])),
                     [(b'mul_ddd', b't3', b'r1[x]', b'r1[x]'),
                      (b'add_ddd', b't3', b't3', b'c2[2.0]'),
                      (b'sum_ddn', b'r0', b't3', None)])
        assert_equal(disassemble(
            NumExpr('sum(x**2+2, axis=1)', [('x', double)])),
                     [(b'mul_ddd', b't3', b'r1[x]', b'r1[x]'),
                      (b'add_ddd', b't3', b't3', b'c2[2.0]'),
                      (b'sum_ddn', b'r0', b't3', 1)])
        assert_equal(disassemble(
            NumExpr('prod(x**2+2, axis=2)', [('x', double)])),
                     [(b'mul_ddd', b't3', b'r1[x]', b'r1[x]'),
                      (b'add_ddd', b't3', b't3', b'c2[2.0]'),
                      (b'prod_ddn', b'r0', b't3', 2)])
        # Check that full reductions work.
        x = zeros(1e5) + .01  # checks issue #41
        assert_allclose(evaluate('sum(x+2,axis=None)'), sum(x + 2, axis=None))
        assert_allclose(evaluate('sum(x+2,axis=0)'), sum(x + 2, axis=0))
        assert_allclose(evaluate('prod(x,axis=0)'), prod(x, axis=0))

        x = arange(10.0)
        assert_allclose(evaluate('sum(x**2+2,axis=0)'), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate('prod(x**2+2,axis=0)'), prod(x ** 2 + 2, axis=0))

        x = arange(100.0)
        assert_allclose(evaluate('sum(x**2+2,axis=0)'), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate('prod(x-1,axis=0)'), prod(x - 1, axis=0))
        x = linspace(0.1, 1.0, 2000)
        assert_allclose(evaluate('sum(x**2+2,axis=0)'), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate('prod(x-1,axis=0)'), prod(x - 1, axis=0))

        # Check that reductions along an axis work
        y = arange(9.0).reshape(3, 3)
        assert_allclose(evaluate('sum(y**2, axis=1)'), sum(y ** 2, axis=1))
        assert_allclose(evaluate('sum(y**2, axis=0)'), sum(y ** 2, axis=0))
        assert_allclose(evaluate('sum(y**2, axis=None)'), sum(y ** 2, axis=None))
        assert_allclose(evaluate('prod(y**2, axis=1)'), prod(y ** 2, axis=1))
        assert_allclose(evaluate('prod(y**2, axis=0)'), prod(y ** 2, axis=0))
        assert_allclose(evaluate('prod(y**2, axis=None)'), prod(y ** 2, axis=None))
        # Check integers
        x = arange(10.)
        x = x.astype(int)
        assert_allclose(evaluate('sum(x**2+2,axis=0)'), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate('prod(x**2+2,axis=0)'), prod(x ** 2 + 2, axis=0))
        # Check longs
        x = x.astype(long)
        assert_allclose(evaluate('sum(x**2+2,axis=0)'), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate('prod(x**2+2,axis=0)'), prod(x ** 2 + 2, axis=0))
        # Check complex
        x = x + .1j
        assert_allclose(evaluate('sum(x**2+2,axis=0)'), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate('prod(x-1,axis=0)'), prod(x - 1, axis=0))
    

    def test_reduction_axis_DEPRECATED(self):
        y = np.arange(9.0).reshape(3, 3)
        try:
            evaluate('sum(y, axis=2)')
        except ValueError:
            pass
        else:
            raise ValueError('should raise exception!')
        try:
            evaluate('sum(y, axis=-3)')
        except ValueError:
            pass
        else:
            raise ValueError('should raise exception!')
        try:
            # Negative axis are not supported
            evaluate('sum(y, axis=-1)')
        except ValueError:
            pass
        else:
            raise ValueError('should raise exception!')
'''

class test_strings(unittest.TestCase):
    # NE3 strings and unicode support not implemented yet
    '''
    BLOCK_SIZE1 = 128
    BLOCK_SIZE2 = 8
    str_list1 = [b'foo', b'bar', b'', b'  ']
    str_list2 = [b'foo', b'', b'x', b' ']
    str_nloops = len(str_list1) * (BLOCK_SIZE1 + BLOCK_SIZE2 + 1)
    str_array1 = array(str_list1 * str_nloops)
    str_array2 = array(str_list2 * str_nloops)
    str_constant = b'doodoo'

    def test_null_chars(self):
        str_list = [
            b'\0\0\0', b'\0\0foo\0', b'\0\0foo\0b', b'\0\0foo\0b\0',
            b'foo\0', b'foo\0b', b'foo\0b\0', b'foo\0bar\0baz\0\0']
        for s in str_list:
            r = evaluate('s')
            self.assertEqual(s, r.tostring())  # check *all* stored data

    def test_compare_copy(self):
        sarr = self.str_array1
        expr = 'sarr'
        res1 = eval(expr)
        res2 = evaluate(expr)
        assert_array_equal(res1, res2)

    def test_compare_array(self):
        sarr1 = self.str_array1
        sarr2 = self.str_array2
        expr = 'sarr1 >= sarr2'
        res1 = eval(expr)
        res2 = evaluate(expr)
        assert_array_equal(res1, res2)

    def test_compare_variable(self):
        sarr = self.str_array1
        svar = self.str_constant
        expr = 'sarr >= svar'
        res1 = eval(expr)
        res2 = evaluate(expr)
        assert_array_equal(res1, res2)

    def test_compare_constant(self):
        sarr = self.str_array1
        expr = 'sarr >= %r' % self.str_constant
        res1 = eval(expr)
        res2 = evaluate(expr)
        assert_array_equal(res1, res2)

    def test_add_string_array(self):
        sarr1 = self.str_array1
        sarr2 = self.str_array2
        expr = 'sarr1 + sarr2'
        self.assert_missing_op('add_sss', expr, locals())

    def test_empty_string1(self):
        a = np.array(['', 'pepe'])
        b = np.array(['pepe2', ''])
        res = evaluate('(a == '') & (b == 'pepe2')')
        assert_array_equal(res, np.array([True, False]))
        res2 = evaluate('(a == 'pepe') & (b == '')')
        assert_array_equal(res2, np.array([False, True]))

    def test_empty_string2(self):
        a = np.array(['p', 'pepe'])
        b = np.array(['pepe2', ''])
        res = evaluate('(a == '') & (b == 'pepe2')')
        assert_array_equal(res, np.array([False, False]))
        res2 = evaluate('(a == 'pepe') & (b == '')')
        assert_array_equal(res, np.array([False, False]))

    def test_add_numeric_array(self):
        sarr = self.str_array1
        narr = arange(len(sarr), dtype='int32')
        expr = 'sarr >= narr'
        self.assert_missing_op('ge_bsi', expr, locals())

    def assert_missing_op(self, op, expr, local_dict):
        msg = 'expected NotImplementedError regarding '%s'' % op
        try:
            evaluate(expr, local_dict)
        except NotImplementedError as nie:
            if ''%s'' % op not in nie.args[0]:
                self.fail(msg)
        else:
            self.fail(msg)

    def test_compare_prefix(self):
        # Check comparing two strings where one is a prefix of the
        # other.
        for s1, s2 in [(b'foo', b'foobar'), (b'foo', b'foo\0bar'),
                       (b'foo\0a', b'foo\0bar')]:
            self.assertTrue(evaluate('s1 < s2'))
            self.assertTrue(evaluate('s1 <= s2'))
            self.assertTrue(evaluate('~(s1 == s2)'))
            self.assertTrue(evaluate('~(s1 >= s2)'))
            self.assertTrue(evaluate('~(s1 > s2)'))

        # Check for NumPy array-style semantics in string equality.
        s1, s2 = b'foo', b'foo\0\0'
        self.assertTrue(evaluate('s1 == s2'))

        def test_str_contains_basic0(self):
        res = evaluate('contains(b'abc', b'ab')')
        assert_equal(res, True)

    def test_str_contains_basic1(self):
        haystack = array([b'abc', b'def', b'xyz', b'x11', b'za'])
        res = evaluate('contains(haystack, b'ab')')
        assert_equal(res, [True, False, False, False, False])

    def test_str_contains_basic2(self):
        haystack = array([b'abc', b'def', b'xyz', b'x11', b'za'])
        res = evaluate('contains(b'abcd', haystack)')
        assert_equal(res, [True, False, False, False, False])

    def test_str_contains_basic3(self):
        haystacks = array(
            [b'abckkk', b'adef', b'xyz', b'x11abcp', b'za', b'abc'])
        needles = array(
            [b'abc', b'def', b'aterr', b'oot', b'zu', b'ab'])
        res = evaluate('contains(haystacks, needles)')
        assert_equal(res, [True, True, False, False, False, True])

    def test_str_contains_basic4(self):
        needles = array(
            [b'abc', b'def', b'aterr', b'oot', b'zu', b'ab c', b' abc',
             b'abc '])
        res = evaluate('contains(b'test abc here', needles)')
        assert_equal(res, [True, False, False, False, False, False, True, True])


    def test_str_contains_basic5(self):
        needles = array(
            [b'abc', b'ab c', b' abc', b' abc ', b'\tabc', b'c h'])
        res = evaluate('contains(b'test abc here', needles)')
        assert_equal(res, [True, False, True, True, False, True])


    def test_str_contains_listproduct(self):
        from itertools import product

        small = [
            'It w', 'as th', 'e Whit', 'e Rab', 'bit,', ' tro', 'tting',
            ' sl', 'owly', ' back ', 'again,', ' and', ' lo', 'okin', 'g a',
            'nxious', 'ly a', 'bou', 't a', 's it w', 'ent,', ' as i', 'f it',
            ' had l', 'ost', ' some', 'thi', 'ng; a', 'nd ', 'she ', 'heard ',
            'it mut', 'terin', 'g to ', 'its', 'elf ', ''The',
            ' Duch', 'ess! T', 'he ', 'Duches', 's! Oh ', 'my dea', 'r paws',
            '! Oh ', 'my f', 'ur ', 'and ', 'whiske', 'rs! ', 'She', ''ll g',
            'et me', ' ex', 'ecu', 'ted, ', 'as su', 're a', 's f', 'errets',
            ' are f', 'errets', '! Wh', 'ere ', 'CAN', ' I hav', 'e d',
            'roppe', 'd t', 'hem,', ' I wo', 'nder?', '' A', 'lice',
            ' gu', 'essed', ' in a', ' mom', 'ent ', 'tha', 't it w', 'as ',
            'looki', 'ng f', 'or ', 'the fa', 'n and ', 'the', ' pai',
            'r of w', 'hit', 'e kid', ' glo', 'ves', ', and ', 'she ',
            'very g', 'ood', '-na', 'turedl', 'y be', 'gan h', 'unt', 'ing',
            ' about', ' for t', 'hem', ', but', ' they ', 'wer', 'e nowh',
            'ere to', ' be', ' se', 'en--', 'ever', 'ythin', 'g seem', 'ed ',
            'to ', 'have c', 'hang', 'ed ', 'since', ' he', 'r swim', ' in',
            ' the', ' pool,', ' and', ' the g', 'reat ', 'hal', 'l, w', 'ith',
            ' th', 'e gl', 'ass t', 'abl', 'e and ', 'the', ' li', 'ttle',
            ' doo', 'r, ha', 'd v', 'ani', 'shed c', 'omp', 'lete', 'ly.']
        big = [
            'It wa', 's the', ' W', 'hit', 'e ', 'Ra', 'bb', 'it, t', 'ro',
            'tting s', 'lowly', ' back ', 'agai', 'n, and', ' l', 'ookin',
            'g ', 'an', 'xiously', ' about ', 'as it w', 'ent, as', ' if ',
            'it had', ' los', 't ', 'so', 'mething', '; and', ' she h',
            'eard ', 'it ', 'mutteri', 'ng to', ' itself', ' 'The ',
            'Duchess', '! ', 'Th', 'e ', 'Duchess', '! Oh m', 'y de',
            'ar paws', '! ', 'Oh my ', 'fu', 'r and w', 'hiskers', '! She'',
            'll ', 'get', ' me ', 'execute', 'd,', ' a', 's ', 'su', 're as ',
            'fe', 'rrets', ' are f', 'errets!', ' Wher', 'e CAN', ' I ha',
            've dro', 'pped t', 'hem', ', I ', 'won', 'der?' A',
            'lice g', 'uess', 'ed ', 'in a m', 'omen', 't that', ' i',
            't was l', 'ook', 'ing f', 'or th', 'e ', 'fan and', ' th', 'e p',
            'air o', 'f whit', 'e ki', 'd glove', 's, and ', 'she v', 'ery ',
            'good-na', 'tu', 'redl', 'y be', 'gan hun', 'ti', 'ng abou',
            't for t', 'he', 'm, bu', 't t', 'hey ', 'were n', 'owhere',
            ' to b', 'e s', 'een-', '-eve', 'rythi', 'ng see', 'me', 'd ',
            'to ha', 've', ' c', 'hanged', ' sinc', 'e her s', 'wim ',
            'in the ', 'pool,', ' an', 'd the g', 'rea', 't h', 'all, wi',
            'th the ', 'glas', 's t', 'able an', 'd th', 'e littl', 'e door,',
            ' had va', 'ni', 'shed co', 'mpletel', 'y.']
        p = list(product(small, big))
        python_in = [x[0] in x[1] for x in p]
        a = [x[0].encode() for x in p]
        b = [x[1].encode() for x in p]
        res = [bool(x) for x in evaluate('contains(b, a)')]
        assert_equal(res, python_in)

    def test_str_contains_withemptystr1(self):
        withemptystr = array([b'abc', b'def', b''])
        res = evaluate('contains(b'abcd', withemptystr)')
        assert_equal(res, [True, False, True])
    
    def test_str_contains_withemptystr2(self):
        withemptystr = array([b'abc', b'def', b''])
        res = evaluate('contains(withemptystr, b'')')
        assert_equal(res, [True, True, True])
'''
# End of test_string


# Cases for testing arrays with dimensions that can be zero.
class test_zerodim(unittest.TestCase):
    def test_zerodim1d(self):
        logger.info( 'Zerodim 1D' )
        a0 = np.array([], dtype='int32')
        a1 = np.array([], dtype='float64')

        r0 = ne3.evaluate('a0 + a1')
        r1 = ne3.evaluate('a0 * a1')

        npt.assert_array_equal(r0, a1)
        npt.assert_array_equal(r1, a1)

    def test_zerodim3d(self):
        logger.info( 'Zerodim 3D' )
        a0 = np.array([], dtype='int32').reshape(0, 2, 4)
        a1 = np.array([], dtype='float64').reshape(0, 2, 4)

        r0 = ne3.evaluate('a0 + a1')
        r1 = ne3.evaluate('a0 * a1')

        npt.assert_array_equal(r0, a1)
        npt.assert_array_equal(r1, a1)


# Tests for threading/multiprocessing/concurrent.futures

def future_worker(neObj: ne3.NumExpr, data: dict, N_threads: int=2):
    '''
    Multiprocessing cannot deal with bound methods, as they aren't pickleable.
    '''
    ne3.set_nthreads(N_threads)
    return neObj(**data)


# Case test for subprocesses (via multiprocessing module)
'''
class test_multicore(unittest.TestCase):

    def test_changing_nthreads_increment(self):
        a = np.linspace(-1, 1, LARGE_SIZE)
        b = ((0.25 * a + 0.75) * a - 1.5) * a - 2
        c = np.empty_like(a)
        for nthreads in [1,2,3,2,1,2,3]:
            ne3.set_nthreads(nthreads)
            ne3.evaluate('c=((0.25*a + 0.75)*a - 1.5)*a - 2')
            npt.assert_array_almost_equal(b, c)

    def test_pickle(self):
        y = np.arange(LARGE_SIZE)
        func = ne3.NumExpr( '(y+2)*y' )
        out1 = func()

        pickledFunc = pickle.dumps( func )
        del func, y

        func2 = pickle.loads(pickledFunc)
        y2 = np.arange(LARGE_SIZE)
        out2 = func2(y=y2)
        npt.assert_array_equal(out1, out2)
        return

    
    def test_threading(self):
        # The NumExpr module has a global state so it can only execute one 
        # operate per process at a time.  Look to solutions such as `dask` 
        # or `concurrent.futures` if you want to use NumExpr in distributed 
        # computing.
        import threading
        logger.info( 'Testing threading' )

        class ThreadTest(threading.Thread):
            def run(self):
                a = np.arange(32.0)
                npt.assert_array_equal(ne3.evaluate('a*a'), a*a)

        test = ThreadTest()
        test.start()

    def test_futures(self):
        # Run a hybrid calculation with two processes each running two pthreads 
        # inside NumExpr.
        logger.info( 'Testing concurrent.futures' )
        try:
            import concurrent.futures as cf
        except ImportError:
            return
        a = np.linspace(1E-7, np.pi, 2*LARGE_SIZE)
        splitA = np.array_split( a, 2 )

        neObj = ne3.NumExpr( 'log(a)' )
        asyncExecutor = cf.ProcessPoolExecutor( max_workers = 2 )   

        workers = [asyncExecutor.submit( future_worker, neObj, {'a':chunkOfA}) for chunkOfA in splitA]
        result =  np.hstack( [worker.result() for worker in workers] )

        npt.assert_array_almost_equal( result, np.log(a) )
'''

def test(verbosity=2):
    '''
    Run all the tests in the test suite.
    '''
    ne3.print_info()
    return unittest.TextTestRunner(verbosity=verbosity).run( suite() )


test.__test__ = False


def suite():
    import unittest
    import platform as pl

    theSuite = unittest.TestSuite()
    niter = 1

    from . import autotest_GENERATED
    theSuite.addTest( unittest.makeSuite( autotest_GENERATED.autotest_numexpr ) )

    for n in range(niter):
        theSuite.addTest( unittest.makeSuite( test_numexpr ) )
        if 'sparc' not in platform.machine():
            theSuite.addTest( unittest.makeSuite( test_numexpr_max ) )

        theSuite.addTest( unittest.makeSuite( test_evaluate) )
        theSuite.addTest( unittest.makeSuite( test_reductions ) )
        theSuite.addTest( unittest.makeSuite( test_strings ) )
        theSuite.addTest( unittest.makeSuite( test_zerodim ) )

        # multiprocessing module is not supported on Hurd/kFreeBSD
        # if pl.system().lower() not in ('gnu', 'gnu/kfreebsd'):
        #     theSuite.addTest( unittest.makeSuite( test_multicore ) )

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
