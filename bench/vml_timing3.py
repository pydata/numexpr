# -*- coding: utf-8 -*-
import numpy as np
import numexpr as ne
from timeit import default_timer as timer

x = np.ones(100000)
scaler = -1J
start = timer()
for k in range(10000):
    cexp = ne.evaluate('exp(scaler * x)')
exec_time=(timer() - start)

print("Execution took", str(round(exec_time, 3)), "seconds")
