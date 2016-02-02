###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################
from __future__ import absolute_import, print_function

import os
import sys
import platform
import warnings
from contextlib import contextmanager

import numpy as np
from numpy import (
    array, arange, empty, zeros, int32, int64, uint16, complex_, float64, rec,
    copy, ones_like, where, alltrue, linspace,
    sum, prod, sqrt, fmod,
    sin, cos, tan, arcsin, arccos, arctan, arctan2,
    sinh, cosh, tanh, arcsinh, arccosh, arctanh,
    log, log1p, log10, exp, expm1, conj)
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose)
from numpy import shape, allclose, array_equal, ravel, isnan, isinf

import numexpr
from numexpr import E, NumExpr, evaluate, disassemble, use_vml

import unittest

TestCase = unittest.TestCase

double = np.double


# Recommended minimum versions
minimum_numpy_version = "1.6"


class test_numexpr(TestCase):
    """Testing with 1 thread"""
    nthreads = 1

    def setUp(self):
        numexpr.set_num_threads(self.nthreads)

    def test_simple(self):
        ex = 2.0 * E.a + 3.0 * E.b * E.c
        sig = [('a', double), ('b', double), ('c', double)]
        func = NumExpr(ex, signature=sig)
        x = func(array([1., 2, 3]), array([4., 5, 6]), array([7., 8, 9]))
        assert_array_equal(x, array([86., 124., 168.]))

    def test_simple_expr_small_array(self):
        func = NumExpr(E.a)
        x = arange(100.0)
        y = func(x)
        assert_array_equal(x, y)

    def test_simple_expr(self):
        func = NumExpr(E.a)
        x = arange(1e6)
        y = func(x)
        assert_array_equal(x, y)

    def test_rational_expr(self):
        func = NumExpr((E.a + 2.0 * E.b) / (1 + E.a + 4 * E.b * E.b))
        a = arange(1e6)
        b = arange(1e6) * 0.1
        x = (a + 2 * b) / (1 + a + 4 * b * b)
        y = func(a, b)
        assert_array_almost_equal(x, y)

    def test_reductions(self):
        # Check that they compile OK.
        assert_equal(disassemble(
            NumExpr("sum(x**2+2, axis=None)", [('x', double)])),
                     [(b'mul_ddd', b't3', b'r1[x]', b'r1[x]'),
                      (b'add_ddd', b't3', b't3', b'c2[2.0]'),
                      (b'sum_ddn', b'r0', b't3', None)])
        assert_equal(disassemble(
            NumExpr("sum(x**2+2, axis=1)", [('x', double)])),
                     [(b'mul_ddd', b't3', b'r1[x]', b'r1[x]'),
                      (b'add_ddd', b't3', b't3', b'c2[2.0]'),
                      (b'sum_ddn', b'r0', b't3', 1)])
        assert_equal(disassemble(
            NumExpr("prod(x**2+2, axis=2)", [('x', double)])),
                     [(b'mul_ddd', b't3', b'r1[x]', b'r1[x]'),
                      (b'add_ddd', b't3', b't3', b'c2[2.0]'),
                      (b'prod_ddn', b'r0', b't3', 2)])
        # Check that full reductions work.
        x = zeros(1e5) + .01  # checks issue #41
        assert_allclose(evaluate("sum(x+2,axis=None)"), sum(x + 2, axis=None))
        assert_allclose(evaluate("sum(x+2,axis=0)"), sum(x + 2, axis=0))
        assert_allclose(evaluate("prod(x,axis=0)"), prod(x, axis=0))

        x = arange(10.0)
        assert_allclose(evaluate("sum(x**2+2,axis=0)"), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate("prod(x**2+2,axis=0)"), prod(x ** 2 + 2, axis=0))

        x = arange(100.0)
        assert_allclose(evaluate("sum(x**2+2,axis=0)"), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate("prod(x-1,axis=0)"), prod(x - 1, axis=0))
        x = linspace(0.1, 1.0, 2000)
        assert_allclose(evaluate("sum(x**2+2,axis=0)"), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate("prod(x-1,axis=0)"), prod(x - 1, axis=0))

        # Check that reductions along an axis work
        y = arange(9.0).reshape(3, 3)
        assert_allclose(evaluate("sum(y**2, axis=1)"), sum(y ** 2, axis=1))
        assert_allclose(evaluate("sum(y**2, axis=0)"), sum(y ** 2, axis=0))
        assert_allclose(evaluate("sum(y**2, axis=None)"), sum(y ** 2, axis=None))
        assert_allclose(evaluate("prod(y**2, axis=1)"), prod(y ** 2, axis=1))
        assert_allclose(evaluate("prod(y**2, axis=0)"), prod(y ** 2, axis=0))
        assert_allclose(evaluate("prod(y**2, axis=None)"), prod(y ** 2, axis=None))
        # Check integers
        x = arange(10.)
        x = x.astype(int)
        assert_allclose(evaluate("sum(x**2+2,axis=0)"), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate("prod(x**2+2,axis=0)"), prod(x ** 2 + 2, axis=0))
        # Check longs
        x = x.astype(long)
        assert_allclose(evaluate("sum(x**2+2,axis=0)"), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate("prod(x**2+2,axis=0)"), prod(x ** 2 + 2, axis=0))
        # Check complex
        x = x + .1j
        assert_allclose(evaluate("sum(x**2+2,axis=0)"), sum(x ** 2 + 2, axis=0))
        assert_allclose(evaluate("prod(x-1,axis=0)"), prod(x - 1, axis=0))

    def test_in_place(self):
        x = arange(10000.).reshape(1000, 10)
        evaluate("x + 3", out=x)
        assert_equal(x, arange(10000.).reshape(1000, 10) + 3)
        y = arange(10)
        evaluate("(x - 3) * y + (x - 3)", out=x)
        assert_equal(x, arange(10000.).reshape(1000, 10) * (arange(10) + 1))

    def test_axis(self):
        y = arange(9.0).reshape(3, 3)
        try:
            evaluate("sum(y, axis=2)")
        except ValueError:
            pass
        else:
            raise ValueError("should raise exception!")
        try:
            evaluate("sum(y, axis=-3)")
        except ValueError:
            pass
        else:
            raise ValueError("should raise exception!")
        try:
            # Negative axis are not supported
            evaluate("sum(y, axis=-1)")
        except ValueError:
            pass
        else:
            raise ValueError("should raise exception!")

    def test_r0_reuse(self):
        assert_equal(disassemble(NumExpr("x * x + 2", [('x', double)])),
                     [(b'mul_ddd', b'r0', b'r1[x]', b'r1[x]'),
                      (b'add_ddd', b'r0', b'r0', b'c2[2.0]')])

    def test_str_contains_basic0(self):
        res = evaluate('contains(b"abc", b"ab")')
        assert_equal(res, True)

    def test_str_contains_basic1(self):
        haystack = array([b'abc', b'def', b'xyz', b'x11', b'za'])
        res = evaluate('contains(haystack, b"ab")')
        assert_equal(res, [True, False, False, False, False])

    def test_str_contains_basic2(self):
        haystack = array([b'abc', b'def', b'xyz', b'x11', b'za'])
        res = evaluate('contains(b"abcd", haystack)')
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
        res = evaluate('contains(b"test abc here", needles)')
        assert_equal(res, [True, False, False, False, False, False, True, True])

    def test_str_contains_basic5(self):
        needles = array(
            [b'abc', b'ab c', b' abc', b' abc ', b'\tabc', b'c h'])
        res = evaluate('contains(b"test abc here", needles)')
        assert_equal(res, [True, False, True, True, False, True])

        # Compare operation of Python 'in' operator with 'contains' using a
        # product of two lists of strings.

    def test_str_contains_listproduct(self):
        from itertools import product

        small = [
            'It w', 'as th', 'e Whit', 'e Rab', 'bit,', ' tro', 'tting',
            ' sl', 'owly', ' back ', 'again,', ' and', ' lo', 'okin', 'g a',
            'nxious', 'ly a', 'bou', 't a', 's it w', 'ent,', ' as i', 'f it',
            ' had l', 'ost', ' some', 'thi', 'ng; a', 'nd ', 'she ', 'heard ',
            'it mut', 'terin', 'g to ', 'its', 'elf ', "'The",
            ' Duch', 'ess! T', 'he ', 'Duches', 's! Oh ', 'my dea', 'r paws',
            '! Oh ', 'my f', 'ur ', 'and ', 'whiske', 'rs! ', 'She', "'ll g",
            'et me', ' ex', 'ecu', 'ted, ', 'as su', 're a', 's f', 'errets',
            ' are f', 'errets', '! Wh', 'ere ', 'CAN', ' I hav', 'e d',
            'roppe', 'd t', 'hem,', ' I wo', 'nder?', "' A", 'lice',
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
            'eard ', 'it ', 'mutteri', 'ng to', ' itself', " 'The ",
            'Duchess', '! ', 'Th', 'e ', 'Duchess', '! Oh m', 'y de',
            'ar paws', '! ', 'Oh my ', 'fu', 'r and w', 'hiskers', "! She'",
            'll ', 'get', ' me ', 'execute', 'd,', ' a', 's ', 'su', 're as ',
            'fe', 'rrets', ' are f', 'errets!', ' Wher', 'e CAN', ' I ha',
            've dro', 'pped t', 'hem', ', I ', 'won', "der?' A",
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
        res = evaluate('contains(b"abcd", withemptystr)')
        assert_equal(res, [True, False, True])

    def test_str_contains_withemptystr2(self):
        withemptystr = array([b'abc', b'def', b''])
        res = evaluate('contains(withemptystr, b"")')
        assert_equal(res, [True, True, True])


class test_numexpr2(test_numexpr):
    """Testing with 2 threads"""
    nthreads = 2


class test_evaluate(TestCase):
    def test_simple(self):
        a = array([1., 2., 3.])
        b = array([4., 5., 6.])
        c = array([7., 8., 9.])
        x = evaluate("2*a + 3*b*c")
        assert_array_equal(x, array([86., 124., 168.]))

    def test_simple_expr_small_array(self):
        x = arange(100.0)
        y = evaluate("x")
        assert_array_equal(x, y)

    def test_simple_expr(self):
        x = arange(1e6)
        y = evaluate("x")
        assert_array_equal(x, y)

    # Test for issue #37
    if sys.version_info[0] < 3:
        # In python 3 '/' perforns true division, not integer division.
        # Integer division '//' is still not suppoerted by numexpr
        def test_zero_div(self):
            x = arange(100, dtype='i4')
            y = evaluate("1/x")
            x2 = zeros(100, dtype='i4')
            x2[1] = 1
            assert_array_equal(x2, y)

    # Test for issue #22
    def test_true_div(self):
        x = arange(10, dtype='i4')
        assert_array_equal(evaluate("x/2"), x / 2)
        assert_array_equal(evaluate("x/2", truediv=False), x / 2)
        assert_array_equal(evaluate("x/2", truediv='auto'), x / 2)
        assert_array_equal(evaluate("x/2", truediv=True), x / 2.0)

    def test_left_shift(self):
        x = arange(10, dtype='i4')
        assert_array_equal(evaluate("x<<2"), x << 2)

    def test_right_shift(self):
        x = arange(10, dtype='i4')
        assert_array_equal(evaluate("x>>2"), x >> 2)

    # PyTables uses __nonzero__ among ExpressionNode objects internally
    # so this should be commented out for the moment.  See #24.
    def test_boolean_operator(self):
        x = arange(10, dtype='i4')
        try:
            evaluate("(x > 1) and (x < 9)")
        except TypeError:
            pass
        else:
            raise ValueError("should raise exception!")

    def test_rational_expr(self):
        a = arange(1e6)
        b = arange(1e6) * 0.1
        x = (a + 2 * b) / (1 + a + 4 * b * b)
        y = evaluate("(a + 2*b) / (1 + a + 4*b*b)")
        assert_array_almost_equal(x, y)
		
    def test_complex64_expr(self):
        def complex64_func(a, b):
            c = zeros(a.shape, dtype=complex64)
            c.real = a
            c.imag = b
            return c

        a = arange(1e4, dtype='float32' )
        b = (arange(1e4) ** 1e-5).astype('float32')
        z = ( a + 1j * b ).astype( 'complex64' ) # RAM this is complex128 by default numpy rules
        x = z.imag
        x = sin(complex64_func(a,b)).real + z.imag

        # RAM: this check cannot pass because we don't have a function to do this 
        # complex64(a,b) in the function list
        y = evaluate("sin(complex64(a, b)).real + z.imag")
        assert_array_almost_equal(x, y)
		
    def test_complex_expr(self):
        def complex(a, b):
            c = zeros(a.shape, dtype=complex_)
            c.real = a
            c.imag = b
            return c

        a = arange(1e4)
        b = arange(1e4) ** 1e-5
        z = a + 1j * b
        x = z.imag
        x = sin(complex(a, b)).real + z.imag
        y = evaluate("sin(complex(a, b)).real + z.imag")
        assert_array_almost_equal(x, y)

    def test_complex_strides(self):
        a = arange(100).reshape(10, 10)[::2]
        b = arange(50).reshape(5, 10)
        assert_array_equal(evaluate("a+b"), a + b)
        c = empty([10], dtype=[('c1', int32), ('c2', uint16)])
        c['c1'] = arange(10)
        c['c2'].fill(0xaaaa)
        c1 = c['c1']
        a0 = a[0]
        assert_array_equal(evaluate("c1"), c1)
        assert_array_equal(evaluate("a0+c1"), a0 + c1)

    def test_broadcasting(self):
        a = arange(100).reshape(10, 10)[::2]
        c = arange(10)
        d = arange(5).reshape(5, 1)
        assert_array_equal(evaluate("a+c"), a + c)
        assert_array_equal(evaluate("a+d"), a + d)
        expr = NumExpr("2.0*a+3.0*c", [('a', double), ('c', double)])
        assert_array_equal(expr(a, c), 2.0 * a + 3.0 * c)

    def test_all_scalar(self):
        a = 3.
        b = 4.
        assert_allclose(evaluate("a+b"), a + b)
        expr = NumExpr("2*a+3*b", [('a', double), ('b', double)])
        assert_equal(expr(a, b), 2 * a + 3 * b)

    def test_run(self):
        a = arange(100).reshape(10, 10)[::2]
        b = arange(10)
        expr = NumExpr("2*a+3*b", [('a', double), ('b', double)])
        assert_array_equal(expr(a, b), expr.run(a, b))

    def test_illegal_value(self):
        a = arange(3)
        try:
            evaluate("a < [0, 0, 0]")
        except TypeError:
            pass
        else:
            self.fail()

    if 'sparc' not in platform.machine():
        # Execution order set here so as to not use too many threads
        # during the rest of the execution.  See #33 for details.
        def test_changing_nthreads_00_inc(self):
            a = linspace(-1, 1, 1e6)
            b = ((.25 * a + .75) * a - 1.5) * a - 2
            for nthreads in range(1, 7):
                numexpr.set_num_threads(nthreads)
                c = evaluate("((.25*a + .75)*a - 1.5)*a - 2")
                assert_array_almost_equal(b, c)

        def test_changing_nthreads_01_dec(self):
            a = linspace(-1, 1, 1e6)
            b = ((.25 * a + .75) * a - 1.5) * a - 2
            for nthreads in range(6, 1, -1):
                numexpr.set_num_threads(nthreads)
                c = evaluate("((.25*a + .75)*a - 1.5)*a - 2")
                assert_array_almost_equal(b, c)


tests = [
    ('MISC', ['b*c+d*e',
              '2*a+3*b',
              '-a',
              'sinh(a)',
              '2*a + (cos(3)+5)*sinh(cos(b))',
              '2*a + arctan2(a, b)',
              'arcsin(0.5)',
              'where(a != 0.0, 2, a)',
              'where(a > 10, b < a, b > a)',
              'where((a-10).real != 0.0, a, 2)',
              '0.25 * (a < 5) + 0.33 * (a >= 5)',
              'cos(1+1)',
              '1+1',
              '1',
              'cos(a2)',
    ])]

optests = []
for op in list('+-*/%') + ['**']:
    optests.append("(a+1) %s (b+3)" % op)
    optests.append("3 %s (b+3)" % op)
    optests.append("(a+1) %s 4" % op)
    optests.append("2 %s (b+3)" % op)
    optests.append("(a+1) %s 2" % op)
    optests.append("(a+1) %s -1" % op)
    optests.append("(a+1) %s 0.5" % op)
    # Check divisions and modulus by zero (see ticket #107)
    optests.append("(a+1) %s 0" % op)
tests.append(('OPERATIONS', optests))

cmptests = []
for op in ['<', '<=', '==', '>=', '>', '!=']:
    cmptests.append("a/2+5 %s b" % op)
    cmptests.append("a/2+5 %s 7" % op)
    cmptests.append("7 %s b" % op)
    cmptests.append("7.0 %s 5" % op)
tests.append(('COMPARISONS', cmptests))

func1tests = []
for func in ['copy', 'ones_like', 'sqrt',
             'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
             'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
             'log', 'log1p', 'log10', 'exp', 'expm1', 'abs', 'conj']:
    func1tests.append("a + %s(b+c)" % func)
tests.append(('1_ARG_FUNCS', func1tests))

func2tests = []
for func in ['arctan2', 'fmod']:
    func2tests.append("a + %s(b+c, d+1)" % func)
    func2tests.append("a + %s(b+c, 1)" % func)
    func2tests.append("a + %s(1, d+1)" % func)
tests.append(('2_ARG_FUNCS', func2tests))

powtests = []
# n = -1, 0.5, 2, 4 already handled in section "OPERATIONS"
for n in (-7, -2.5, -1.5, -1.3, -.5, 0, 0.0, 1, 2.3, 2.5, 3):
    powtests.append("(a+1)**%s" % n)
tests.append(('POW_TESTS', powtests))


def equal(a, b, exact):
    if array_equal(a, b):
        return True

    if hasattr(a, 'dtype') and a.dtype in ['f4', 'f8']:
        nnans = isnan(a).sum()
        if nnans > 0:
            # For results containing NaNs, just check that the number
            # of NaNs is the same in both arrays.  This check could be
            # made more exhaustive, but checking element by element in
            # python space is very expensive in general.
            return nnans == isnan(b).sum()
        ninfs = isinf(a).sum()
        if ninfs > 0:
            # Ditto for Inf's
            return ninfs == isinf(b).sum()
    if exact:
        return (shape(a) == shape(b)) and alltrue(ravel(a) == ravel(b), axis=0)
    else:
        if hasattr(a, 'dtype') and a.dtype == 'f4':
            atol = 1e-5  # Relax precission for special opcodes, like fmod
        else:
            atol = 1e-8
        return (shape(a) == shape(b) and
                allclose(ravel(a), ravel(b), atol=atol))


class Skip(Exception): pass


def test_expressions():
    test_no = [0]

    def make_test_method(a, a2, b, c, d, e, x, expr,
                         test_scalar, dtype, optimization, exact, section):
        this_locals = locals()

        def method():
            # We don't want to listen at RuntimeWarnings like
            # "overflows" or "divide by zero" in plain eval().
            warnings.simplefilter("ignore")
            npval = eval(expr, globals(), this_locals)
            warnings.simplefilter("always")
            npval = eval(expr, globals(), this_locals)
            try:
                neval = evaluate(expr, local_dict=this_locals,
                                 optimization=optimization)
                assert equal(npval, neval, exact), """%r
(test_scalar=%r, dtype=%r, optimization=%r, exact=%r,
 npval=%r (%r - %r)\n neval=%r (%r - %r))""" % (expr, test_scalar, dtype.__name__,
                                                optimization, exact,
                                                npval, type(npval), shape(npval),
                                                neval, type(neval), shape(neval))
            except AssertionError:
                raise
            except NotImplementedError:
                print('%r not implemented for %s (scalar=%d, opt=%s)'
                      % (expr, dtype.__name__, test_scalar, optimization))
            except:
                print('numexpr error for expression %r' % (expr,))
                raise

        method.description = ('test_expressions(%s, test_scalar=%r, '
                              'dtype=%r, optimization=%r, exact=%r)') \
                             % (expr, test_scalar, dtype.__name__, optimization, exact)
        test_no[0] += 1
        method.__name__ = 'test_scalar%d_%s_%s_%s_%04d' % (test_scalar,
                                                           dtype.__name__,
                                                           optimization.encode('ascii'),
                                                           section.encode('ascii'),
                                                           test_no[0])
        return method

    x = None
    for test_scalar in (0, 1, 2):
        for dtype in (int, long, np.float32, double, complex):
            array_size = 100
            a = arange(2 * array_size, dtype=dtype)[::2]
            a2 = zeros([array_size, array_size], dtype=dtype)
            b = arange(array_size, dtype=dtype) / array_size
            c = arange(array_size, dtype=dtype)
            d = arange(array_size, dtype=dtype)
            e = arange(array_size, dtype=dtype)
            if dtype == complex:
                a = a.real
                for x in [a2, b, c, d, e]:
                    x += 1j
                    x *= 1 + 1j
            if test_scalar == 1:
                a = a[array_size // 2]
            if test_scalar == 2:
                b = b[array_size // 2]
            for optimization, exact in [
                ('none', False), ('moderate', False), ('aggressive', False)]:
                for section_name, section_tests in tests:
                    for expr in section_tests:
                        if (dtype == complex and
                            ('<' in expr or '>' in expr or '%' in expr
                             or "arctan2" in expr or "fmod" in expr)):
                            # skip complex comparisons or functions not
                            # defined in complex domain.
                            continue
                        if (dtype in (int, long) and test_scalar and
                                    expr == '(a+1) ** -1'):
                            continue

                        m = make_test_method(a, a2, b, c, d, e, x,
                                             expr, test_scalar, dtype,
                                             optimization, exact,
                                             section_name)
                        yield m


class test_int64(TestCase):
    def test_neg(self):
        a = array([2 ** 31 - 1, 2 ** 31, 2 ** 32, 2 ** 63 - 1], dtype=int64)
        res = evaluate('-a')
        assert_array_equal(res, [1 - 2 ** 31, -(2 ** 31), -(2 ** 32), 1 - 2 ** 63])
        self.assertEqual(res.dtype.name, 'int64')


class test_int32_int64(TestCase):
    if sys.version_info[0] < 2:
        # no long literals in python 3
        def test_small_long(self):
            # Small longs should not be downgraded to ints.
            res = evaluate('42L')
            assert_array_equal(res, 42)
            self.assertEqual(res.dtype.name, 'int64')

    def test_small_int(self):
        # Small ints (32-bit ones) should not be promoted to longs.
        res = evaluate('2')
        assert_array_equal(res, 2)
        self.assertEqual(res.dtype.name, 'int32')

    def test_big_int(self):
        # Big ints should be promoted to longs.
        res = evaluate('2**40')
        assert_array_equal(res, 2 ** 40)
        self.assertEqual(res.dtype.name, 'int64')

    def test_long_constant_promotion(self):
        int32array = arange(100, dtype='int32')
        itwo = np.int32(2)
        ltwo = np.int64(2)
        res = int32array * 2
        res32 = evaluate('int32array * itwo')
        res64 = evaluate('int32array * ltwo')
        assert_array_equal(res, res32)
        assert_array_equal(res, res64)
        self.assertEqual(res32.dtype.name, 'int32')
        self.assertEqual(res64.dtype.name, 'int64')

    def test_int64_array_promotion(self):
        int32array = arange(100, dtype='int32')
        int64array = arange(100, dtype='int64')
        respy = int32array * int64array
        resnx = evaluate('int32array * int64array')
        assert_array_equal(respy, resnx)
        self.assertEqual(resnx.dtype.name, 'int64')


class test_uint32_int64(TestCase):
    def test_small_uint32(self):
        # Small uint32 should not be downgraded to ints.
        a = np.uint32(42)
        res = evaluate('a')
        assert_array_equal(res, 42)
        self.assertEqual(res.dtype.name, 'int64')

    def test_uint32_constant_promotion(self):
        int32array = arange(100, dtype='int32')
        stwo = np.int32(2)
        utwo = np.uint32(2)
        res = int32array * utwo
        res32 = evaluate('int32array * stwo')
        res64 = evaluate('int32array * utwo')
        assert_array_equal(res, res32)
        assert_array_equal(res, res64)
        self.assertEqual(res32.dtype.name, 'int32')
        self.assertEqual(res64.dtype.name, 'int64')

    def test_int64_array_promotion(self):
        uint32array = arange(100, dtype='uint32')
        int64array = arange(100, dtype='int64')
        respy = uint32array * int64array
        resnx = evaluate('uint32array * int64array')
        assert_array_equal(respy, resnx)
        self.assertEqual(resnx.dtype.name, 'int64')


class test_strings(TestCase):
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
        a = np.array(["", "pepe"])
        b = np.array(["pepe2", ""])
        res = evaluate("(a == '') & (b == 'pepe2')")
        assert_array_equal(res, np.array([True, False]))
        res2 = evaluate("(a == 'pepe') & (b == '')")
        assert_array_equal(res2, np.array([False, True]))

    def test_empty_string2(self):
        a = np.array(["p", "pepe"])
        b = np.array(["pepe2", ""])
        res = evaluate("(a == '') & (b == 'pepe2')")
        assert_array_equal(res, np.array([False, False]))
        res2 = evaluate("(a == 'pepe') & (b == '')")
        assert_array_equal(res, np.array([False, False]))

    def test_add_numeric_array(self):
        sarr = self.str_array1
        narr = arange(len(sarr), dtype='int32')
        expr = 'sarr >= narr'
        self.assert_missing_op('ge_bsi', expr, locals())

    def assert_missing_op(self, op, expr, local_dict):
        msg = "expected NotImplementedError regarding '%s'" % op
        try:
            evaluate(expr, local_dict)
        except NotImplementedError, nie:
            if "'%s'" % op not in nie.args[0]:
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


# Case for testing selections in fields which are aligned but whose
# data length is not an exact multiple of the length of the record.
# The following test exposes the problem only in 32-bit machines,
# because in 64-bit machines 'c2' is unaligned.  However, this should
# check most platforms where, while not unaligned, 'len(datatype) >
# boundary_alignment' is fullfilled.
class test_irregular_stride(TestCase):
    def test_select(self):
        f0 = arange(10, dtype=int32)
        f1 = arange(10, dtype=float64)

        irregular = rec.fromarrays([f0, f1])

        f0 = irregular['f0']
        f1 = irregular['f1']

        i0 = evaluate('f0 < 5')
        i1 = evaluate('f1 < 5')

        assert_array_equal(f0[i0], arange(5, dtype=int32))
        assert_array_equal(f1[i1], arange(5, dtype=float64))


# Cases for testing arrays with dimensions that can be zero.
class test_zerodim(TestCase):
    def test_zerodim1d(self):
        a0 = array([], dtype=int32)
        a1 = array([], dtype=float64)

        r0 = evaluate('a0 + a1')
        r1 = evaluate('a0 * a1')

        assert_array_equal(r0, a1)
        assert_array_equal(r1, a1)

    def test_zerodim3d(self):
        a0 = array([], dtype=int32).reshape(0, 2, 4)
        a1 = array([], dtype=float64).reshape(0, 2, 4)

        r0 = evaluate('a0 + a1')
        r1 = evaluate('a0 * a1')

        assert_array_equal(r0, a1)
        assert_array_equal(r1, a1)


@contextmanager
def _environment(key, value):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old:
            os.environ[key] = old
        else:
            del os.environ[key]


# Test cases for the threading configuration
class test_threading_config(TestCase):
    def test_numexpr_num_threads(self):
        with _environment('OMP_NUM_THREADS', '5'):
            with _environment('NUMEXPR_NUM_THREADS', '3'):
                self.assertEquals(3, numexpr.detect_number_of_threads())

    def test_omp_num_threads(self):
        with _environment('OMP_NUM_THREADS', '5'):
            self.assertEquals(5, numexpr.detect_number_of_threads())


# Case test for threads
class test_threading(TestCase):
    def test_thread(self):
        import threading

        class ThreadTest(threading.Thread):
            def run(self):
                a = arange(3)
                assert_array_equal(evaluate('a**3'), array([0, 1, 8]))

        test = ThreadTest()
        test.start()


# The worker function for the subprocess (needs to be here because Windows
# has problems pickling nested functions with the multiprocess module :-/)
def _worker(qout=None):
    ra = np.arange(1e3)
    rows = evaluate('ra > 0')
    #print "Succeeded in evaluation!\n"
    if qout is not None:
        qout.put("Done")


# Case test for subprocesses (via multiprocessing module)
class test_subprocess(TestCase):
    def test_multiprocess(self):
        try:
            import multiprocessing as mp
        except ImportError:
            return
        # Check for two threads at least
        numexpr.set_num_threads(2)
        #print "**** Running from main process:"
        _worker()
        #print "**** Running from subprocess:"
        qout = mp.Queue()
        ps = mp.Process(target=_worker, args=(qout,))
        ps.daemon = True
        ps.start()

        result = qout.get()
        #print result


def print_versions():
    """Print the versions of software that numexpr relies on."""
    from pkg_resources import parse_version
    if parse_version(np.__version__) < parse_version(minimum_numpy_version):
        print("*Warning*: NumPy version is lower than recommended: %s < %s" % \
              (np.__version__, minimum_numpy_version))
    print('-=' * 38)
    print("Numexpr version:   %s" % numexpr.__version__)
    print("NumPy version:     %s" % np.__version__)
    print('Python version:    %s' % sys.version)
    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print('Platform:          %s-%s' % (sys.platform, machine))
    print("AMD/Intel CPU?     %s" % numexpr.is_cpu_amd_intel)
    print("VML available?     %s" % use_vml)
    if use_vml:
        print("VML/MKL version:   %s" % numexpr.get_vml_version())
    print("Number of threads used by default: %d "
          "(out of %d detected cores)" % (numexpr.nthreads, numexpr.ncores))
    print('-=' * 38)


def test():
    """
    Run all the tests in the test suite.
    """

    print_versions()
    return unittest.TextTestRunner().run(suite())


test.__test__ = False


def suite():
    import unittest
    import platform as pl

    theSuite = unittest.TestSuite()
    niter = 1

    class TestExpressions(TestCase):
        pass

    def add_method(func):
        def method(self):
            return func()

        setattr(TestExpressions, func.__name__,
                method.__get__(None, TestExpressions))

    for func in test_expressions():
        add_method(func)

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(test_numexpr))
        if 'sparc' not in platform.machine():
            theSuite.addTest(unittest.makeSuite(test_numexpr2))
        theSuite.addTest(unittest.makeSuite(test_evaluate))
        theSuite.addTest(unittest.makeSuite(TestExpressions))
        theSuite.addTest(unittest.makeSuite(test_int32_int64))
        theSuite.addTest(unittest.makeSuite(test_uint32_int64))
        theSuite.addTest(unittest.makeSuite(test_strings))
        theSuite.addTest(
            unittest.makeSuite(test_irregular_stride))
        theSuite.addTest(unittest.makeSuite(test_zerodim))
        theSuite.addTest(unittest.makeSuite(test_threading_config))

        # multiprocessing module is not supported on Hurd/kFreeBSD
        if (pl.system().lower() not in ('gnu', 'gnu/kfreebsd')):
            theSuite.addTest(unittest.makeSuite(test_subprocess))

        # I need to put this test after test_subprocess because
        # if not, the test suite locks immediately before test_subproces.
        # This only happens with Windows, so I suspect of a subtle bad
        # interaction with threads and subprocess :-/
        theSuite.addTest(unittest.makeSuite(test_threading))

    return theSuite


if __name__ == '__main__':
    print_versions()
    unittest.main(defaultTest='suite')
#    suite = suite()
#    unittest.TextTestRunner(verbosity=2).run(suite)
