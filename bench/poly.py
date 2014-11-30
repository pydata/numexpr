###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

#######################################################################
# This script compares the speed of the computation of a polynomial
# for different (numpy and numexpr) in-memory paradigms.
#
# Author: Francesc Alted
# Date: 2010-07-06
#######################################################################

from __future__ import print_function
import sys
from time import time
import numpy as np
import numexpr as ne


#expr = ".25*x**3 + .75*x**2 - 1.5*x - 2"  # the polynomial to compute
expr = "((.25*x + .75)*x - 1.5)*x - 2"  # a computer-friendly polynomial
N = 10*1000*1000               # the number of points to compute expression
x = np.linspace(-1, 1, N)   # the x in range [-1, 1]

#what = "numpy"              # uses numpy for computations
what = "numexpr"           # uses numexpr for computations

def compute():
    """Compute the polynomial."""
    if what == "numpy":
        y = eval(expr)
    else:
        y = ne.evaluate(expr)
    return len(y)


if __name__ == '__main__':
    if len(sys.argv) > 1:  # first arg is the package to use
        what = sys.argv[1]
    if len(sys.argv) > 2:  # second arg is the number of threads to use
        nthreads = int(sys.argv[2])
        if "ncores" in dir(ne):
            ne.set_num_threads(nthreads)
    if what not in ("numpy", "numexpr"):
        print("Unrecognized module:", what)
        sys.exit(0)
    print("Computing: '%s' using %s with %d points" % (expr, what, N))
    t0 = time()
    result = compute()
    ts = round(time() - t0, 3)
    print("*** Time elapsed:", ts)
