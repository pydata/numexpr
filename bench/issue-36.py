import numpy as np
import numexpr as ne
from numpy.testing import assert_array_equal
from time import time

ne.set_num_threads(2)
a = np.arange(2**10)

t0 = time()
for i in xrange(1000):
    ne.evaluate('a>1000')
print "numexpr--> %.3f" % (time()-t0,)

t0 = time()
for i in xrange(1000):
    eval('a>1000')
print "numpy--> %.3f" % (time()-t0,)


