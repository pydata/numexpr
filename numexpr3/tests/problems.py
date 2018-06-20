import numpy as np
import numpy.testing as npt
import numexpr3 as ne3
import scipy.special
from matplotlib import use; use('Qt5Agg')
import matplotlib.pyplot as plt

a = np.random.uniform(size=12)
b = np.random.uniform(size=12)

# Broken functions
# factorial

a = np.random.uniform(low=-5000.0, high=1e6, size=256)
b = np.random.uniform(low=-5000.0, high=1e6, size=256)
# k = np.linspace(-5, 128, 256)
# k = np.arange(-128 + 4,  128 + 4 ).astype('uint32')
# l = np.arange(-128 - 50, 128 - 50).astype('uint32')
# FACTORIAL
# out_np = scipy.special.factorial(k)
# out_explicit = np.prod(k)
# out_ne = ne3.NumExpr('factorial(k)')()

# LOGADDEXP
# out_np = np.logaddexp(a, b)
# out_explicit = np.log(np.exp(a) + np.exp(b))
# out_ne = ne3.NumExpr('logaddexp(a, b)')()
# https://github.com/numpy/numpy/blob/d407f24cb5ad85850a60b1be3dade3305ea30c98/numpy/core/src/npymath/npy_math_internal.h.src

# print(f'Failure at {np.argwhere(np.logical_not(np.isclose(out_ne, out_np)))}')

# MOD
out_np = np.mod(a, b)
out_explicit = a % b
out_ne = ne3.NumExpr('mod(a, b)')()


plt.figure()
plt.plot(out_np, '.', label='np')
plt.plot(out_ne, '-', label='ne3')
plt.legend()
plt.show()

#
try:
    assert(np.allclose(out_np, out_ne, rtol=1e-5, equal_nan=True))
except AssertionError:
    print('ASSERTION_ERROR')
    errorIndices = np.logical_not(np.isclose(out_np, out_ne, rtol=1e-5, equal_nan=True))
    print(f'NumPy: {out_np[errorIndices]}')
    print(f'NumExpr3: {out_ne[errorIndices]}')
    print(f'Explicit: {out_explicit[errorIndices]}')
# Ah, the problem here is `decimal`, whereas we really want np.isclose...
# npt.assert_array_almost_equal(out_ne, out_np, decimal=2)