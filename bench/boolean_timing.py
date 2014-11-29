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
import sys
import timeit
import numpy

array_size = 1000*1000
iterations = 10

numpy_ttime = []
numpy_sttime = []
numpy_nttime = []
numexpr_ttime = []
numexpr_sttime = []
numexpr_nttime = []

def compare_times(expr, nexpr):
    global numpy_ttime
    global numpy_sttime
    global numpy_nttime
    global numexpr_ttime
    global numexpr_sttime
    global numexpr_nttime

    print("******************* Expression:", expr)

    setup_contiguous = setupNP_contiguous
    setup_strided = setupNP_strided
    setup_unaligned = setupNP_unaligned

    numpy_timer = timeit.Timer(expr, setup_contiguous)
    numpy_time = round(numpy_timer.timeit(number=iterations), 4)
    numpy_ttime.append(numpy_time)
    print('numpy:', numpy_time / iterations)

    numpy_timer = timeit.Timer(expr, setup_strided)
    numpy_stime = round(numpy_timer.timeit(number=iterations), 4)
    numpy_sttime.append(numpy_stime)
    print('numpy strided:', numpy_stime / iterations)

    numpy_timer = timeit.Timer(expr, setup_unaligned)
    numpy_ntime = round(numpy_timer.timeit(number=iterations), 4)
    numpy_nttime.append(numpy_ntime)
    print('numpy unaligned:', numpy_ntime / iterations)

    evalexpr = 'evaluate("%s", optimization="aggressive")' % expr
    numexpr_timer = timeit.Timer(evalexpr, setup_contiguous)
    numexpr_time = round(numexpr_timer.timeit(number=iterations), 4)
    numexpr_ttime.append(numexpr_time)
    print("numexpr:", numexpr_time/iterations, end=" ")
    print("Speed-up of numexpr over numpy:", round(numpy_time/numexpr_time, 4))

    evalexpr = 'evaluate("%s", optimization="aggressive")' % expr
    numexpr_timer = timeit.Timer(evalexpr, setup_strided)
    numexpr_stime = round(numexpr_timer.timeit(number=iterations), 4)
    numexpr_sttime.append(numexpr_stime)
    print("numexpr strided:", numexpr_stime/iterations, end=" ")
    print("Speed-up of numexpr strided over numpy:", \
          round(numpy_stime/numexpr_stime, 4))

    evalexpr = 'evaluate("%s", optimization="aggressive")' % expr
    numexpr_timer = timeit.Timer(evalexpr, setup_unaligned)
    numexpr_ntime = round(numexpr_timer.timeit(number=iterations), 4)
    numexpr_nttime.append(numexpr_ntime)
    print("numexpr unaligned:", numexpr_ntime/iterations, end=" ")
    print("Speed-up of numexpr unaligned over numpy:", \
          round(numpy_ntime/numexpr_ntime, 4))



setupNP = """\
from numpy import arange, where, arctan2, sqrt
from numpy import rec as records
from numexpr import evaluate

# Initialize a recarray of 16 MB in size
r=records.array(None, formats='a%s,i4,f8', shape=%s)
c1 = r.field('f0')%s
i2 = r.field('f1')%s
f3 = r.field('f2')%s
c1[:] = "a"
i2[:] = arange(%s)/1000
f3[:] = i2/2.
"""

setupNP_contiguous = setupNP % (4, array_size,
                                ".copy()", ".copy()", ".copy()",
                                array_size)
setupNP_strided = setupNP % (4, array_size, "", "", "", array_size)
setupNP_unaligned = setupNP % (1, array_size, "", "", "", array_size)


expressions = []
expressions.append('i2 > 0')
expressions.append('i2 < 0')
expressions.append('i2 < f3')
expressions.append('i2-10 < f3')
expressions.append('i2*f3+f3*f3 > i2')
expressions.append('0.1*i2 > arctan2(i2, f3)')
expressions.append('i2%2 > 3')
expressions.append('i2%10 < 4')
expressions.append('i2**2 + (f3+1)**-2.5 < 3')
expressions.append('(f3+1)**50 > i2')
expressions.append('sqrt(i2**2 + f3**2) > 1')
expressions.append('(i2>2) | ((f3**2>3) & ~(i2*f3<2))')

def compare(expression=False):
    if expression:
        compare_times(expression, 1)
        sys.exit(0)
    nexpr = 0
    for expr in expressions:
        nexpr += 1
        compare_times(expr, nexpr)
    print()

if __name__ == '__main__':

    import numexpr
    numexpr.print_versions()

    if len(sys.argv) > 1:
        expression = sys.argv[1]
        print("expression-->", expression)
        compare(expression)
    else:
        compare()

    tratios = numpy.array(numpy_ttime) / numpy.array(numexpr_ttime)
    stratios = numpy.array(numpy_sttime) / numpy.array(numexpr_sttime)
    ntratios = numpy.array(numpy_nttime) / numpy.array(numexpr_nttime)


    print("*************** Numexpr vs NumPy speed-ups *******************")
#     print "numpy total:", sum(numpy_ttime)/iterations
#     print "numpy strided total:", sum(numpy_sttime)/iterations
#     print "numpy unaligned total:", sum(numpy_nttime)/iterations
#     print "numexpr total:", sum(numexpr_ttime)/iterations
    print("Contiguous case:\t %s (mean), %s (min), %s (max)" % \
          (round(tratios.mean(), 2),
           round(tratios.min(), 2),
           round(tratios.max(), 2)))
#    print "numexpr strided total:", sum(numexpr_sttime)/iterations
    print("Strided case:\t\t %s (mean), %s (min), %s (max)" % \
          (round(stratios.mean(), 2),
           round(stratios.min(), 2),
           round(stratios.max(), 2)))
#    print "numexpr unaligned total:", sum(numexpr_nttime)/iterations
    print("Unaligned case:\t\t %s (mean), %s (min), %s (max)" % \
          (round(ntratios.mean(), 2),
           round(ntratios.min(), 2),
           round(ntratios.max(), 2)))
