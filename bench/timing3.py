###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import timeit, numpy
from time import perf_counter

array_size = 2**20
iterations = 4

# Choose the type you want to benchmark
#dtype = 'int8'
#dtype = 'int16'
#dtype = 'int32'
#dtype = 'int64'
dtype = 'float32'
#dtype = 'float64'

def compare_times(setup, expr):
    print("Expression:", expr)
    # namespace = {}
    # exec(setup.format(array_size, dtype, expr), namespace)

    numpy_timer = timeit.Timer(expr, setup.format(array_size, dtype, expr ) )
    numpy_time = numpy_timer.timeit(number=iterations )
    print('numpy: {:.2e} s'.format( numpy_time / iterations) )

    numexpr_timer = timeit.Timer('neFunc()', setup.format(array_size, dtype, expr) )
    numexpr_time = numexpr_timer.timeit(number=iterations )
    print("numexpr: {:.2e} s".format(numexpr_time/iterations) )

    tratio = numpy_time/numexpr_time
    print("Speed-up of numexpr over numpy:", round(tratio, 2))
    return tratio

setup1 = """\
from numpy import arange
from numexpr3 import NumExpr
result = arange({0}, dtype='{1}')
a = arange({0}, dtype='{1}')
b = arange({0}, dtype='{1}')
c = arange({0}, dtype='{1}')
d = arange({0}, dtype='{1}')
e = arange({0}, dtype='{1}')
neFunc = NumExpr('{2}')
"""

setup3 = """\
from numpy import arange, sin, cos, sinh, arctan2, sqrt, where
from numexpr3 import NumExpr
result = arange({0}, dtype='{1}')
a = arange(2*{0}, dtype='{1}')[::2]
b = arange({0}, dtype='{1}')
neFunc = NumExpr('{2}')
"""

expr1 = 'result=b*c+d*e'
expr2 = 'result=2*a+3*b'
expr3 = 'result=2*a + (cos(3)+5)*sinh(cos(b))'
expr4 = 'result=2*a + arctan2(a, b)'
expr5 = 'result=where(0.1*a > arctan2(a, b), 2*a, arctan2(a,b))'
expr6 = 'result=where(a != 0.0, 2, b)'
expr7 = 'result=where(a-10 != 0.0, a, 2)'
expr8 = 'result=where(a%2 != 0.0, b+5, 2)'
expr9 = 'result=where(a%2 != 0.0, 2, b+5)'
expr10 = 'result=a**2 + (b+1)**-2.5'
# expr11 = '(a+1)**50'
expr12 = 'result=sqrt(a*a + b*b)'

def compare(check_only=False):
    experiments = [(setup1, expr1), (setup1, expr2), (setup3, expr3),
                   (setup3, expr4), (setup3, expr5), (setup3, expr6),
                   (setup3, expr7), (setup3, expr8), (setup3, expr9),
                   (setup3, expr10), 
                   # (setup3, expr11), 
                   (setup3, expr12),
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
