###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

from __future__ import print_function
import timeit, numpy

array_size = 1e6
iterations = 2

# Choose the type you want to benchmark
#dtype = 'int8'
#dtype = 'int16'
#dtype = 'int32'
#dtype = 'int64'
dtype = 'float32'
#dtype = 'float64'

def compare_times(setup, expr):
    print("Expression:", expr)
    namespace = {}
    exec(setup, namespace)

    numpy_timer = timeit.Timer(expr, setup)
    numpy_time = numpy_timer.timeit(number=iterations)
    print('numpy:', numpy_time / iterations)

    try:
        weave_timer = timeit.Timer('blitz("result=%s")' % expr, setup)
        weave_time = weave_timer.timeit(number=iterations)
        print("Weave:", weave_time/iterations)

        print("Speed-up of weave over numpy:", round(numpy_time/weave_time, 2))
    except:
        print("Skipping weave timing")

    numexpr_timer = timeit.Timer('evaluate("%s", optimization="aggressive")' % expr, setup)
    numexpr_time = numexpr_timer.timeit(number=iterations)
    print("numexpr:", numexpr_time/iterations)

    tratio = numpy_time/numexpr_time
    print("Speed-up of numexpr over numpy:", round(tratio, 2))
    return tratio

setup1 = """\
from numpy import arange
try: from scipy.weave import blitz
except: pass
from numexpr import evaluate
result = arange(%f, dtype='%s')
b = arange(%f, dtype='%s')
c = arange(%f, dtype='%s')
d = arange(%f, dtype='%s')
e = arange(%f, dtype='%s')
""" % ((array_size, dtype)*5)
expr1 = 'b*c+d*e'

setup2 = """\
from numpy import arange
try: from scipy.weave import blitz
except: pass
from numexpr import evaluate
a = arange(%f, dtype='%s')
b = arange(%f, dtype='%s')
result = arange(%f, dtype='%s')
""" % ((array_size, dtype)*3)
expr2 = '2*a+3*b'


setup3 = """\
from numpy import arange, sin, cos, sinh
try: from scipy.weave import blitz
except: pass
from numexpr import evaluate
a = arange(2*%f, dtype='%s')[::2]
b = arange(%f, dtype='%s')
result = arange(%f, dtype='%s')
""" % ((array_size, dtype)*3)
expr3 = '2*a + (cos(3)+5)*sinh(cos(b))'


setup4 = """\
from numpy import arange, sin, cos, sinh, arctan2
try: from scipy.weave import blitz
except: pass
from numexpr import evaluate
a = arange(2*%f, dtype='%s')[::2]
b = arange(%f, dtype='%s')
result = arange(%f, dtype='%s')
""" % ((array_size, dtype)*3)
expr4 = '2*a + arctan2(a, b)'


setup5 = """\
from numpy import arange, sin, cos, sinh, arctan2, sqrt, where
try: from scipy.weave import blitz
except: pass
from numexpr import evaluate
a = arange(2*%f, dtype='%s')[::2]
b = arange(%f, dtype='%s')
result = arange(%f, dtype='%s')
""" % ((array_size, dtype)*3)
expr5 = 'where(0.1*a > arctan2(a, b), 2*a, arctan2(a,b))'

expr6 = 'where(a != 0.0, 2, b)'

expr7 = 'where(a-10 != 0.0, a, 2)'

expr8 = 'where(a%2 != 0.0, b+5, 2)'

expr9 = 'where(a%2 != 0.0, 2, b+5)'

expr10 = 'a**2 + (b+1)**-2.5'

expr11 = '(a+1)**50'

expr12 = 'sqrt(a**2 + b**2)'

def compare(check_only=False):
    experiments = [(setup1, expr1), (setup2, expr2), (setup3, expr3),
                   (setup4, expr4), (setup5, expr5), (setup5, expr6),
                   (setup5, expr7), (setup5, expr8), (setup5, expr9),
                   (setup5, expr10), (setup5, expr11), (setup5, expr12),
                   ]
    total = 0
    for params in experiments:
        total += compare_times(*params)
        print
    average = total / len(experiments)
    print("Average =", round(average, 2))
    return average

if __name__ == '__main__':
    import numexpr
    print("Numexpr version: ", numexpr.__version__)

    averages = []
    for i in range(iterations):
        averages.append(compare())
    print("Averages:", ', '.join("%.2f" % x for x in averages))
