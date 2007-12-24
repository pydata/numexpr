from numexpr.info import __doc__
from numexpr.expressions import E
from numexpr.compiler import numexpr, disassemble, evaluate

def test(*args, **kw):
    from numpy.testing import NumpyTest
    NumpyTest().test(*args, **kw)
