import numpy
import numexpr

numexpr.set_num_threads(8)
x0,x1,x2,x3,x4,x5 = [0,1,2,3,4,5]
t = numpy.linspace(0,1,44100000).reshape(-1,1)
numexpr.evaluate('(x0+x1*t+x2*t**2)* cos(x3+x4*t+x5**t)')
