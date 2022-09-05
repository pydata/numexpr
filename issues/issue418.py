import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from time import perf_counter as pc

# geomspace seems to be very slow, just a warning about setting `n` too high.
# n = 2**28
n = 2048
x = np.geomspace(7.8e-20, 7.8e-0, n)
# x = np.zeros(n)
y = np.geomspace(2.3e-20j, 2.3e-0j, n)
z = x + y

# Check to see if we have MKL or not
ne.print_versions()

# Spin up NumExpr before benchmarking
z0 = z[:1024]
out = ne.evaluate('expm1(z0)')
outx = ne.evaluate('expm1x(z0)')

# Benchmark the options
t0 = pc()
np_out = np.expm1(z)
t1 = pc()
ne_out = ne.evaluate('expm1(z)')
t2 = pc()
nex_out = ne.evaluate('expm1x(z)')
t3 = pc()

print(f'NumPy evaluated in {t1-t0:.3f} s')
print(f'NumExpr (old) evaluated in {t2-t1:.3f} s')
print(f'NumExpr (new) evaluated in {t3-t2:.3f} s')

if n <= 4096:
    plt.figure()
    plt.plot(z, np_out.imag, label='np')
    plt.plot(z, ne_out.imag, label='ne')
    plt.plot(z, nex_out.imag, label='nex')
    plt.legend()
    if ne.use_vml:
        plt.title('im(Expm1(z)) w/ MKL')
    else:
        plt.title('im(Expm1(z)) w/o MKL')

    plt.figure()
    plt.plot(z, ne_out.real - np_out.real, label='ne')
    plt.plot(z, nex_out.real - np_out.real, label='nex')
    plt.legend()
    if ne.use_vml:
        plt.title('im(ne)-(np) w/ MKL')
    else:
        plt.title('im(ne)-imag(np) w/o MKL')

    plt.figure()
    plt.plot(z, ne_out.imag - np_out.imag, label='ne')
    plt.plot(z, nex_out.imag - np_out.imag, label='nex')
    plt.legend()
    if ne.use_vml:
        plt.title('re(ne)-re(np) w/ MKL')
    else:
        plt.title('re(ne)-re(np) w/o MKL')

    plt.figure()
    plt.plot(z, np.abs(ne_out) - np.abs(np_out), label='ne')
    plt.plot(z, np.abs(nex_out) - np.abs(np_out), label='nex')
    plt.legend()
    if ne.use_vml:
        plt.title('|ne|-|np| w/ MKL')
    else:
        plt.title('|ne|-|np| w/o MKL')


    plt.show()
