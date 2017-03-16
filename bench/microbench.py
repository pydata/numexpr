#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microbenchmarks for acceleration of NumExpr3 startup time

Created on Fri Dec 23 12:40:40 2016

@author: Robert A. McLeod
"""

import timeit
NN = 100000
cast_time = timeit.timeit( "np.empty(0,dtype='f')", 
                          'import numpy as np', number=NN )
print( "Call numpy.empty(0): %e s" % (cast_time/NN ) )

# 90 ns
itertools_count = timeit.timeit( "next(counter)", 
                                "import itertools; counter=itertools.count()", 
                                number=NN )
print( "Itertools.counter: %e s" % (itertools_count/NN) )

# About 800 ns:
build_from_uint32 = timeit.timeit( "uint32(4).tobytes()", 'from numpy import uint32', number=NN )
print( "Build byte string from np.uint32().tobytes(): %e s" % (build_from_uint32/NN ) )

# About 130 ns
# Much faster than numpy.tobytes()
build_from_struct = timeit.timeit( "pack('i',foo)", 
                                  'from struct import pack; import numpy as np; foo = np.int32(4)', 
                                  number=NN )
print( "Build byte string from struct.pack: %e s" % (build_from_struct/NN ) )

build_single_byte = timeit.timeit( "bytes( (foo,) )", 
                                  "foo=4", number=NN )
print( "Build byte singleton from bytes: %e s" % (build_single_byte/NN ) )

# About 130 ns
calling_hasattr_dtype = timeit.timeit( "hasattr(foo,'dtype')", 'import numpy as np; foo = np.int32(4)', number=NN )
print( "Ask hasattr about dtype: %e s" % (calling_hasattr_dtype/NN ) )

# About 50 ns
calling_id = timeit.timeit( "id(foo)", 'import numpy as np; foo = np.zeros(16)', number=NN )
print( "Ask id about address: %e s" % (calling_id/NN ) )

# What's the fastest way to make names for temporaries and constants?
# 200 ns
reg_assign = timeit.timeit( "b'$' + pack('b',4)", 'from struct import pack', number=NN )
print( "Merge register key: %e s" % (reg_assign/NN))

# Encoding tuples into a bytes instead of using tuples
# 630 ns
pack_tuple = timeit.timeit( "pack( 'iqcb', 0, id(foo), foo.dtype.char.encode('ascii'), 0 )",
                           "import numpy as np; from struct import pack; foo = np.int64(5)",
                           number = NN )
# 170 ns
build_tuple = timeit.timeit( "( 0, id(foo), foo.dtype.char, 0 )",
                           "import numpy as np; foo = np.int64(5)",
                           number = NN )
print( "Packing tuple into bytes: %e s" % (pack_tuple/NN))
print( "Building tuple: %e s" % (build_tuple/NN))

# What's faster, calling ioStream.write 4 times or building from a tuple?
# Ok it's fast, 260 ns
quad_iostream = timeit.timeit( "IO.write( op ); IO.write( ret ); IO.write( reg1 ); IO.write(reg2)",
"""import io; IO = io.BytesIO(); 
IO.write( b"".join( [bytes((J,)) for J in range(256)] ));
op=b'0010'; ret=b'4'; reg1=b'1'; reg2=b'3'""",
                              number = NN )
# And this is faster, 230 ns **WINNER IN GENERAL CASE**
# Maybe we should use a bytearray then instead of a stream?
one_iostream = timeit.timeit( "IO.write( b''.join( (op,ret,reg1,reg2)) )",\
"""import io; IO = io.BytesIO(); 
IO.write( b"".join( [bytes((J,)) for J in range(256)] ));
op=b'0010'; ret=b'4'; reg1=b'1'; reg2=b'3'""",
                              number = NN )
# Faster still, 162 ns (WHEN EMPTY), 250 ns (WHEN POPULATED)
# So what about when it's long already?
bytearray_join = timeit.timeit( "BA.join(  (op,ret,reg1,reg2) )",
"""BA = bytearray().join( [bytes((J,)) for J in range(256)] ); 
op=b'0010'; ret=b'4'; reg1=b'1'; reg2=b'3'""",
                              number = NN )
print( "Calling io.ByteStream x4: %e s" % (quad_iostream/NN) )
print( "Calling io.ByteStream x1 with join: %e s" % (one_iostream/NN) )
print( "Joining bytearray: %e s" % (bytearray_join/NN) )

# Inserting assignments into the stack level above the function call
stack_insert_pre = """
import sys

def insertUpStack():
    up_dict = sys._getframe( 1 ).f_locals
    up_dict['foo'] = 55
"""
insert_up_stack = timeit.timeit( "insertUpStack()", stack_insert_pre, number=NN )
print( "Getting frame and inserting assignment to calling stack level: %e s" % (insert_up_stack/NN) )


# Calling numpy.broadcast on a big array
broadcast_pre = """
import numpy as np

a = np.ones( [512,512] )
b = np.ones( [512,1] )
"""
broadcast_time = timeit.timeit( "bshape = np.broadcast(a,b).shape;", broadcast_pre, number =NN )
print( "Time to broadcast: %e s" % (broadcast_time/NN) )
# So about a microsecond.  Not horrible.


#NPY_NO_EXPORT int
#PyArray_Broadcast(PyArrayMultiIterObject *mit)
#{
#    int i, nd, k, j;
#    npy_intp tmp;
#    PyArrayIterObject *it;
#
#    /* Discover the broadcast number of dimensions */
#    for (i = 0, nd = 0; i < mit->numiter; i++) {
#        nd = PyArray_MAX(nd, PyArray_NDIM(mit->iters[i]->ao));
#    }
#    mit->nd = nd;
#
#    /* Discover the broadcast shape in each dimension */
#    for (i = 0; i < nd; i++) {
#        mit->dimensions[i] = 1;
#        for (j = 0; j < mit->numiter; j++) {
#            it = mit->iters[j];
#            /* This prepends 1 to shapes not already equal to nd */
#            k = i + PyArray_NDIM(it->ao) - nd;
#            if (k >= 0) {
#                tmp = PyArray_DIMS(it->ao)[k];
#                if (tmp == 1) {
#                    continue;
#                }
#                if (mit->dimensions[i] == 1) {
#                    mit->dimensions[i] = tmp;
#                }
#                else if (mit->dimensions[i] != tmp) {
#                    PyErr_SetString(PyExc_ValueError,
#                                    "shape mismatch: objects" \
#                                    " cannot be broadcast" \
#                                    " to a single shape");
#                    return -1;
#                }
#            }
#        }
#    }