import numpy as np
import numexpr3 as ne3

a = np.arange(2**16, dtype='float32')
out = np.zeros_like(a)
ne3.evaluate( 'out=a*a+2' )
