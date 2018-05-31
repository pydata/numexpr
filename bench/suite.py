"""
Benchmarking Suite for NumExpr3
===============================

The suite for NE3 is designed for internal comparisons of different versions of 
NE3. Previous efforts to also compare to NumExpr2.6/Numba became a little overly 
complicated and have been abandoned as a result.

What we want here is some standard that specifies the information a benchmark
should collect for inclusion in a JSON file.

Filename convention
-------------------

<machine_name>_<ne_version>

Software info
-------------

NumPy version
NumExpr version
Compiler name
Compiler version
Compiler flags
Block size (when compiled with fixed BLOCKSIZE)

Hardware info
-------------
CPU Clock speed
Virtual cores
Physical cores
Number of processors
Cache sizes
CPU flags

Timings
-------
Stored as a dict that can be saved as JSON, which requires ``str`` keys.
"""
from typing import Tuple, Union
from time import perf_counter as pc
import numpy as np
import numexpr3 as ne3
from collections import namedtuple
import json

SIZE_RANGE = np.logspace(14, 22, 25, base=2)
MAX_SIZE   = SIZE_RANGE.max()

Info   = namedtuple('Info', ['numpy_version', 
                             'numexpr3_version', 
                             'compiler_name', 
                             'compiler_version', 
                             'compiler_flags', 
                             'block_size'])
Result = namedtuple('Result', [ 'expr',
                                'time',
                                'size', 
                                'nthread', 
                                'dtypes'])

# For a store, we can generate the longest required arrays and then crop
# them for each 
# Let's have 3 store arrays for each dtype, `[A, B, C]`

def bench(expr: str, dtypes: Tuple[Union[np.dtype, str]], size: int=65536, 
          nthread: int=4, tries: int=3) -> Result:
    """
    """
    # Get views of the required store arrays

    # Generate the compiled NumExpr object
    func = ne3.NumExpr(expr)
    t0 = pc()
    for I in range(tries):
        out = func()
    t1 = pc()
    # TODO: should we embed also the fine timings from the NumExpr3 object 
    # if the `--bench` code was used?
    return Result(expr, (t1-t0)/tries, size, nthread, )

def save():
    """
    To save namedtuples, convert them to ``dict`` objects via ``namedtuple._asdict()``
    before serializing as JSON? They are saved without field names when dumping 
    into JSON.
    """
    return

