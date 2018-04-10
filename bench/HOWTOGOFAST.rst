NumExpr Benchmarking submodule
==============================

Move this into the docs...

What we want here is some standard that specifies the information a benchmark
should collet for inclusion in a JSON file.

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
Stored as a dict
keys should be named tuples: (expr, arraySize, nthreads)
Named tuples should be versioned, starting with v1