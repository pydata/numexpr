# -*- coding: utf-8 -*-
"""
NumExpr3 interpreter function generator

Created on Thu Jan  5 12:46:25 2017
@author: Robert A. McLeod
@email: robbmcleod@gmail.com

"""
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import count
import struct

try: import cPickle
except: import pickle as cPickle

import ast
import os,sys,inspect
CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
NE_DIR = os.path.join( os.path.dirname(CURR_DIR), 'numexpr3' )
sys.path.insert(0,NE_DIR) 

NE_STRUCT = 'H'

# Insertions to be done in interpreter.hpp:
INTERP_HEADER_DEFINES = ''
# These tokens are used for finding the text block where the header and 
# body text should be replaced
INSERT_POINT = '//GENERATOR_INSERT_POINT'
PY_INSERT_POINT = '#GENERATOR_INSERT_POINT'
WARNING_EDIT = """// WARNING: THIS FILE IS AUTO-GENERATED PRIOR TO COMPILATION.
// Editing should be done on the associated stub file instead.
"""
WARNING_END = """// End of GENERATED CODE BLOCK"""

# Note: LIB_XXX and CAST_XXX could be imported  from necompiler.py, or maybe
# they should be in __init__.py?  Either way only define them in one place.
LIB_DEFAULT = 0
LIB_STD = 0
LIB_VML = 1

CAST_SAFE = 0
CAST_NO = 1
CAST_EQUIV = 2
CAST_SAME_KIND = 3
CAST_UNSAFE = 4

BLOCKSIZE = 4096 # Will be set as a global by generate()

# TODO: try different blocksizes for each itemsize
#BLOCKSIZE_1 = 32768
#BLOCKSIZE_2 = 16384
#BLOCKSIZE_4 = 8192
#BLOCKSIZE_8 = 4096
#BLOCKSIZE_16 = 2048

# If we start the operations at a value of 256 we provide space for all the
# Python op codes within the NumExpr3 operation space. We should be careful
# to make sure the C jump table is full though.
# OP_COUNT = count( start=256 )
OP_COUNT = count( start=1 ) # Leave space for NOOP=0

DCHAR_TO_CTYPE = { 
                   np.dtype('bool').char:       'npy_bool', 
                   np.dtype('uint8').char:      'npy_uint8', 
                   np.dtype('int8').char:       'npy_int8',
                   np.dtype('uint16').char:     'npy_uint16', 
                   np.dtype('int16').char:      'npy_int16',
                   np.dtype('uint32').char:     'npy_uint32', 
                   np.dtype('int32').char:      'npy_int32',
                   np.dtype('uint64').char:     'npy_uint64', 
                   np.dtype('int64').char:      'npy_int64',
                   np.dtype('float32').char:    'npy_float32', 
                   np.dtype('float64').char:    'npy_float64',
                   np.dtype('complex64').char:  'npy_complex64', 
                   np.dtype('complex128').char: 'npy_complex128', 
                   np.dtype('S1').char:         'NPY_STRING', 
                   np.dtype('U1').char:         'NPY_UNICODE', 
                  }

#def STRING_EXPR( opNum, expr, retDtype, arg1Dtype=None, arg2Dtype=None, arg3Dtype=None ):
#    # More custom operations
#    # TODO
#    pass

def DEST( index='J' ):
    # Real return is the default
    return 'dest[{0}]'.format( index )
    
#def DEST_STR( dchar ):
#    # TODO: difference between npy_string and npy_unicode?
#    return '({0} *)dest + J*memsteps[store_in]'.format( DCHAR_TO_CTYPE[dchar] )

def ARG( num, index='J' ):
    return 'x{0}[{1}]'.format( num, index )

def ARG_STRIDE( num, index='J' ):
    return 'x{0}[{1}*sb{0}]'.format( num, index )

#def ARG_STR( dchar, num ):
#    return '(({0} *)( x{1} + J*sb{1} ))'.format( DCHAR_TO_CTYPE[dchar], num )

def REDUCE( dchar, outerLoop=False ):
    if outerLoop:
        return DEST( dchar )
    # INNER LOOP
    return '*({0} *)dest'.format( DCHAR_TO_CTYPE[dchar] )

def VEC_LOOP( expr ):
    expr = expr.replace( '$ARG3', ARG(3) )
    expr = expr.replace( '$ARG2', ARG(2) )
    expr = expr.replace( '$ARG1', ARG(1) )
    return '''for(npy_intp J = 0; J < block_size; J++) { 
    EXPR; 
}'''.replace( 'EXPR', expr ) 
 
def STRIDED_LOOP( expr ):
    expr = expr.replace( '$ARG3', ARG_STRIDE(3) )
    expr = expr.replace( '$ARG2', ARG_STRIDE(2) )
    expr = expr.replace( '$ARG1', ARG_STRIDE(1) )
    retStr = ''
    if 'x3' in expr: retStr += '    sb3 /= sizeof($DTYPE3);\n';
    if 'x2' in expr: retStr += '    sb2 /= sizeof($DTYPE2);\n';
    if 'x1' in expr: retStr += '    sb1 /= sizeof($DTYPE1);\n';
    retStr += '''for(npy_intp J = 0; J < block_size; J++) { 
        EXPR; 
    }'''.replace( 'EXPR', expr ) 
    return retStr


def VEC_ARG0( expr ):
    return '''
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    $DTYPE0 *dest = ($DTYPE0 *)params->registers[store_in].mem;
    
    VEC_LOOP(expr);
    return 0;
    }
'''.replace( 'VEC_LOOP(expr);', VEC_LOOP(expr) )

# We could write a more general function suitable for any number of arguments,
# but that would make the generator code more opaque
def VEC_ARG1(expr):
    return '''
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    $DTYPE0 *dest = ($DTYPE0 *)params->registers[store_in].mem;
    $DTYPE1 *x1 = ($DTYPE1 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof($DTYPE1) ) { // Aligned
        VEC_LOOP(expr)
        return 0;
    } 
    // Strided
    STRIDED_LOOP(expr)
    return 0;
    }
'''.replace('STRIDED_LOOP(expr)', STRIDED_LOOP(expr) ).replace( 'VEC_LOOP(expr)', VEC_LOOP(expr) )

def VEC_ARG2(expr):
    return '''
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    $DTYPE0 *dest = ($DTYPE0 *)params->registers[store_in].mem;
    $DTYPE1 *x1 = ($DTYPE1 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    $DTYPE2 *x2 = ($DTYPE2 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof($DTYPE1) && sb2 == sizeof($DTYPE2) ) { // Aligned
        VEC_LOOP(expr)
        return 0;
    }
    // Strided
    STRIDED_LOOP(expr)
    return 0;
    }
'''.replace('STRIDED_LOOP(expr)', STRIDED_LOOP(expr) ).replace( 'VEC_LOOP(expr)', VEC_LOOP(expr) )


def VEC_ARG3(expr):
    return '''
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    $DTYPE0 *dest = ($DTYPE0 *)params->registers[store_in].mem;
    $DTYPE1 *x1 = ($DTYPE1 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    $DTYPE2 *x2 = ($DTYPE2 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    $DTYPE3 *x3 = ($DTYPE3 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof($DTYPE1) && sb2 == sizeof($DTYPE2) && sb3 == sizeof($DTYPE3) ) { // Aligned
        VEC_LOOP(expr)
        return 0;
    }
    // Strided
    STRIDED_LOOP(expr)
    return 0;
    }
'''.replace('STRIDED_LOOP(expr)', STRIDED_LOOP(expr) ).replace( 'VEC_LOOP(expr)', VEC_LOOP(expr) )

# This is a function lookup helper dict
VEC_ARGN = { 0: VEC_ARG0, 1: VEC_ARG1, 2: VEC_ARG2, 3: VEC_ARG3 }


#def VEC_ARG1_STRING(expr):
#    # Version with itemsize is _only_ needed for strings
#    return '''
#    {   
#        NE_REGISTER arg1 = params->program[pc].arg1;
#        BOUNDS_CHECK(store_in);
#        BOUNDS_CHECK(arg1);
#
#        char *dest = params->registers[store_in].mem;
#        char *x1 = params->registers[arg1].mem;
#        npy_intp ss1 = params->registers[arg1].itemsize;
#        npy_intp sb1 = params->registers[arg1].stride;
#        
#        VEC_LOOP(expr);
#    } break;
#'''.replace( 'VEC_LOOP(expr);', VEC_LOOP(expr) )

#def VEC_ARG2_STRING(expr):
#    # Version with itemsize is _only_ needed for strings
#    return '''
#    { 
#        NE_REGISTER arg1 = params->program[pc].arg1;
#        NE_REGISTER arg2 = params->program[pc].arg2;
#        BOUNDS_CHECK(store_in);
#        BOUNDS_CHECK(arg1);
#        BOUNDS_CHECK(arg2);
#        char *dest = params->registers[store_in].mem;
#        char *x1 = params->registers[arg1].mem;
#        npy_intp ss1 = params->registers[arg1].itemsize;
#        npy_intp sb1 = params->registers[arg1].stride;
#        char *x2 = params->registers[arg2].mem;
#        npy_intp ss2 = params->registers[arg2].itemsize;
#        npy_intp sb2 = params->registers[arg2].stride;
#        VEC_LOOP(expr);
#    } break;
#'''.replace( 'VEC_LOOP(expr);', VEC_LOOP(expr) ) 

#def VEC_ARG3_STRING(expr):
#    # Version with itemsize is _only_ needed for strings
#    return '''
#    {
#        NE_REGISTER arg1 = params->program[pc].arg1;
#        NE_REGISTER arg2 = params->program[pc].arg2;
#        NE_REGISTER arg3 = params->program[pc].arg3;
#        BOUNDS_CHECK(store_in);
#        BOUNDS_CHECK(arg1);
#        BOUNDS_CHECK(arg2);
#        BOUNDS_CHECK(arg3);
#        
#        char *dest = params->registers[store_in].mem;
#        char *x1 = params->registers[arg1].mem;
#        npy_intp ss1 = params->registers[arg1].itemsize;
#        npy_intp sb1 = params->registers[arg1].stride;
#        char *x2 = params->registers[arg2].mem;
#        npy_intp ss2 = params->registers[arg2].itemsize;
#        npy_intp sb2 = params->registers[arg2].stride;
#        char *x3 = params->registers[arg3].mem;
#        npy_intp ss3 = params->registers[arg3].itemsize;
#        npy_intp sb3 = params->registers[arg3].stride;
#        
#        VEC_LOOP(expr);
#    } break;
#'''.replace( 'VEC_LOOP(expr);', VEC_LOOP(expr) ) 



# The Intel VML calls, or any vectorized library that takes the BLOCK_SIZE as  
# an argument, such as complex_functions.hpp, and doesn't allow for a 
# stride


def VEC_ARG0_ALIGNED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        BOUNDS_CHECK(store_in);

        char *dest = params->registers[store_in].mem;
        EXPR;
        return 0;
    }
'''.replace( 'EXPR', expr )

def VEC_ARG1_ALIGNED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        EXPR;
        return 0;
    }
'''.replace( 'EXPR', expr )

def VEC_ARG2_ALIGNED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        EXPR;
        return 0;
    }
'''.replace( 'EXPR', expr )

def VEC_ARG3_ALIGNED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        NE_REGISTER arg3 = params->program[pc].arg3;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        BOUNDS_CHECK(arg3);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        char *x3 = params->registers[arg3].mem;
        EXPR;
        return 0;
    }
'''.replace( 'EXPR', expr )
VEC_ARGN_ALIGNED = { 0: VEC_ARG0_ALIGNED, 1: VEC_ARG1_ALIGNED, 2: VEC_ARG2_ALIGNED, 3: VEC_ARG3_ALIGNED }

# Strided expressions can iterate over non-unity steps, i.e. 2,4,3, 
# which is significantly slower, so in general it's best if there's a branch 
# inside each function: one for strided and one for aligned data.
def VEC_ARG0_STRIDED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        BOUNDS_CHECK(store_in);

        char *dest = params->registers[store_in].mem;
        
        EXPR;
        return 0;
    }
'''.replace( 'EXPR', expr )

def VEC_ARG1_STRIDED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        npy_intp sb1 = params->registers[arg1].stride;
        
        EXPR;
        return 0;
    }
'''.replace( 'EXPR', expr )

def VEC_ARG2_STRIDED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        npy_intp sb1 = params->registers[arg1].stride;
        char *x2 = params->registers[arg2].mem;
        npy_intp sb2 = params->registers[arg2].stride;
        
        EXPR;
        return 0;
    }
'''.replace( 'EXPR', expr )

def VEC_ARG3_STRIDED(expr):
    return '''
{
        NE_REGISTER store_in = params->program[pc].ret;
        BOUNDS_CHECK(store_in);
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        NE_REGISTER arg3 = params->program[pc].arg3;
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        BOUNDS_CHECK(arg3);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        npy_intp sb1 = params->registers[arg1].stride;
        char *x2 = params->registers[arg2].mem;
        npy_intp sb2 = params->registers[arg2].stride;
        char *x3 = params->registers[arg3].mem;
        npy_intp sb3 = params->registers[arg3].stride;
        
        EXPR;
        return 0;
    } 
'''.replace( 'EXPR', expr )
VEC_ARGN_STRIDED = { 0:VEC_ARG0_STRIDED, 1:VEC_ARG1_STRIDED, 2:VEC_ARG2_STRIDED, 3:VEC_ARG3_STRIDED }


TYPE_LOOP = 0
TYPE_STRING = 1
TYPE_ALIGNED = 2
TYPE_STRIDED = 3
def EXPR( opNum, expr, retChar, 
         arg1Dchar=None, arg2Dchar=None, arg3Dchar=None, vecType = TYPE_LOOP ):
    '''
    '''
    argChars = [item for item in (arg1Dchar, arg2Dchar, arg3Dchar) if item != None]

    # STRING, and UNICODE have special operation syntax
    # Perhaps it would be easier to make string operations _all_ functions.
    if 'S' in argChars or 'U' in argChars:
        print( 'TODO: string: %s ' % expr )
        return 

    # Replace tokens
    # TODO: this should moreso be a C-define, but perhaps BLOCK_SIZE1/2/4/8/16?
    expr = expr.replace( '$BLOCKSIZE', str(BLOCKSIZE) )

    # Build loop structure
    # docString = '// {} ({}) :: {}'.format(retChar, argChars, expr)
    if vecType == TYPE_LOOP:
        expr = VEC_ARGN[len(argChars)](expr)
        #expr = 'case {0}: {2} {1}'.format( opNum, VEC_ARGN[len(argChars)](expr), docString )
    elif vecType == TYPE_ALIGNED:
        expr = VEC_ARGN_ALIGNED[len(argChars)](expr)
        #expr =  'case {0}: {2} {1}'.format( opNum, VEC_ARGN_ALIGNED[len(argChars)](expr), docString )
    elif vecType == TYPE_STRIDED:
        expr = VEC_ARGN_STRIDED[len(argChars)](expr)
        #expr = 'case {0}: {2} {1}'.format( opNum, VEC_ARGN_STRIDED[len(argChars)](expr), docString )
    else:
        raise ValueError( "Unknown vectorization type: {}".format(vecType) )
        
    expr = expr.replace( '$DEST', DEST() )
    expr = expr.replace( '$DTYPE0', DCHAR_TO_CTYPE[retChar] )
    for I, dchar in enumerate(argChars):
        #expr = expr.replace( '$ARG{}'.format(I+1), ARG( dchar, I+1 ) )
        expr = expr.replace( '$DTYPE{}'.format(I+1), DCHAR_TO_CTYPE[dchar]  )
        
    return expr


'''
NumPy dtype.chars are used to identify the various types available for building
functions as well as the tuples used as keys to ID the opcodes.

On Linux:
bool = '?'
uint8 = 'B'
int8 = 'b'
uint16 = 'H'
int16 = 'h'
uint32 = 'I'
int32 = 'i'
uint64 = 'L'
int64 = 'l'
float32 = 'f'
float64 = 'd'
longfloat = 'g'   # Not supported
complex64 = 'F'
complex128 = 'D'
bytes = 'S'
str/unicode = 'U'

Windows has different characters.
'''
# These are some convience shortcuts for our operation hash-table below
BOOL = [ np.dtype('bool').char ]
SIGNED_INT = [ np.dtype('int8').char,np.dtype('int16').char,
              np.dtype('int32').char,np.dtype('int64').char]
UNSIGNED_INT = [ np.dtype('uint8').char,np.dtype('uint16').char,
              np.dtype('uint32').char,np.dtype('uint64').char]
DECIMAL  =  [np.dtype('float32').char,np.dtype('float64').char]
COMPLEX = [np.dtype('complex64').char,np.dtype('complex128').char]
BOOLx2 = [np.dtype('bool').char] * 2
INTx2 = [np.dtype('int32').char] * 2
LONGx2 = [np.dtype('int64').char] * 2       
STRINGS = ['S','U']
ALL_INT = SIGNED_INT + UNSIGNED_INT
REAL_NUM = BOOL + ALL_INT + DECIMAL
SIGNED_NUM = SIGNED_INT + DECIMAL
BITWISE_NUM = BOOL + ALL_INT
ALL_NUM = BOOL + ALL_INT + DECIMAL + COMPLEX    
ALL_FAM = ALL_NUM + STRINGS

# Map NumpPy functions that don't exist for the testing submodule
# Any function that doesn't have a NumPy equivalent returns None.
AUTOTEST_DICT = defaultdict( bool, {
             'add': 'A_$DTYPE1 + B_$DTYPE2',
             'sub': 'A_$DTYPE1 - B_$DTYPE2',
             'mult': 'A_$DTYPE1 * B_$DTYPE2',
             'div': 'A_$DTYPE1 / B_$DTYPE2',
             'neg': '-A_$DTYPE1',
             'mod': 'A_$DTYPE1 % B_$DTYPE2',
             'pow': 'A_$DTYPE1 ** B_$DTYPE2',
             'lshift': 'A_$DTYPE1 << B_$DTYPE2',
             'rshift': 'A_$DTYPE1 >> B_$DTYPE2',
             'bitand': 'A_$DTYPE1 & B_$DTYPE2',
             'bitor': 'A_$DTYPE1 | B_$DTYPE2',
             'bitxor': 'A_$DTYPE1 ^ B_$DTYPE2',
             'gt': 'A_$DTYPE1 > B_$DTYPE2',
             'gte':  'A_$DTYPE1 >= B_$DTYPE2',
             'lt': 'A_$DTYPE1 < B_$DTYPE2',
             'lte':  'A_$DTYPE1 <= B_$DTYPE2',
             'eq': 'A_$DTYPE1 == B_$DTYPE2',
             'noteq': 'A_$DTYPE1 != B_$DTYPE2',
             # There's no equivalent for ne's complex() array builder in NumPy
             'complex': '',
             } )


# This is the red meat of the generator, where we give the code stubs and the
# valid data type families.  Eventually the actual operation lists should be 
# in a seperate file from the language rules that build the operation hash.
class Operation(object):
    '''
    Test making Operation object-oriented as the factory functions are becoming
    grossly oversized.
    
    Takes a tokenized representation of a function/operation and builds it 
    by string replacement over the specified dtypes.  
    
    Valid replacement keywords are: 
           $DEST, $DTYPE0
           $ARG1, $DTYPE1
           $ARG2, $DTYPE
           $ARG3, $DTYPE3
    '''
    
    def __init__(self, py_Name, c_Template, libs, retFam, *argFams, 
                 vecType=TYPE_LOOP, alias=None ):
        # py_Name is either an ast.Node or a string such as 'sqrt'
        self.py_Name = py_Name
        # c_Template is the

        self.c_Template = c_Template
        self.vecType = vecType
        
        # libs is a list, valid libraries are LIB_STD and LIB_VML, possibly 
        # LIB_SIMD later.
        if type(libs) == list or type(libs) == tuple:
            self.libs = libs
        else:
            self.libs = [libs]
        
        # List of valid return character identifiers
        self.retFam = retFam
        # List of lists of valid argument character identifiers
        self.argFams = argFams
        
        self.alias = alias
        self.opNums = []
        self.c_FunctionDeclares = []
        self.py_TupleKeys = []
        self.c_FunctionImpls = []
        self.c_OpTableEntrys = []
        
        self.build()
        
    def build(self):
        global OP_COUNT
        
        for lib in self.libs:
            for I, retChar in enumerate( self.retFam ):
                opNum = next( OP_COUNT )
                
                self.opNums.append( opNum )
                passedArgs = [ arg[I] for arg in self.argFams ]
                
                funcBody = \
                        EXPR( opNum, self.c_Template, retChar, *passedArgs, 
                              vecType=self.vecType )
                        
                # The format for the Python tuple is (name,{lib},arg1type,...)
                # or for casts, ('cast',{cast_mode},cast_dtype,original_dtype )
                if self.py_Name == 'cast':
                    self.py_TupleKeys.append( 
                        tuple( [self.py_Name, lib, retChar] + passedArgs) )
                else:
                    self.py_TupleKeys.append( 
                        tuple( [self.py_Name, lib] + passedArgs) )
                    if bool(self.alias):
                        self.py_TupleKeys.append( 
                                tuple( [self.alias, lib] + passedArgs) ) 
                        
                            
                
                # Make a unique name for the function, of the form 
                # {function}_{retchar}{argchar1,...}
                if type(self.py_Name) == type:
                    funcName = self.py_Name.__name__.lower()
                else:
                    funcName = self.py_Name
                    
                # print( 'lib: %d, #%d: '%(lib,I) + str(funcName) + str(passedArgs) )
                # Bool's dchar '?' is illegal so we use '1' instead for function names
                idArgs = [ arg.replace('?','1') for arg in passedArgs ]
                
                funcNameUnique = ''.join( [funcName,'_',retChar.replace('?','1') ] + idArgs )
                
                funcDec = 'static int\n'    \
                          + funcNameUnique \
                          + '( npy_intp block_size, npy_intp pc, const NumExprObject *params )'
                          
                funcCall = funcNameUnique \
                           + '( block_size, pc, params );' 
                
                self.c_FunctionDeclares.append(  funcDec + ';' )
                self.c_OpTableEntrys.append( 'case {0}: {1} break;\n'.format( opNum, funcCall) )
                self.c_FunctionImpls.append( ''.join([funcDec, funcBody]) )
                
                # Debugging output
                #print( '##### %s #####' % opNum )
                #print( cTable[opNum] )
       
    @property
    def test_Auto(self):
        # Try to write a test function that compares our function to NumPy.
        if type(self.py_Name) == type: # Ast  node
            funcName = self.py_Name.__name__.lower()
        else: # Function name
            funcName = self.py_Name
            
        testCode = ['']
        evalFunc = None
        # There are a number of sample arrays in the autotest stub,
        # A_x, B_x, and C_x where 'x' is the dchar.
        
        for lib in self.libs:
            for I, retChar in enumerate( self.retFam ):
                passedArgs = [ arg[I] for arg in self.argFams ]
                idArgs = [ arg.replace('?','1') for arg in passedArgs ]
                
                # TODO: implement cast/copy testing
                if funcName == 'cast' or funcName == 'copy':
                    return ''
                    
                
                if AUTOTEST_DICT[funcName]:
                    evalFunc = AUTOTEST_DICT[funcName]
                elif not hasattr( np, funcName ):
                    print( "test_Auto could not find NumPy function for: {}".format(funcName) )
                    return ''
                
                funcNameUnique = ''.join( [funcName,'_',retChar.replace('?','1') ] + idArgs )
                # I wonder if it would be easier to build the program by hand
                # and not parse anything?  Or maybe I should put aliases 
                # in the OPTABLE?
                if evalFunc == None:
                    if len(idArgs) == 0:
                       evalFunc = '{0}()'.format(funcName)
                    elif len(idArgs) == 1:
                       evalFunc = '{0}( A_{1} )'.format(funcName, idArgs[0] )
                    elif len(idArgs) == 2:
                       evalFunc = '{0}( A_{1}, B_{2} )'.format(funcName, idArgs[0], idArgs[1] )     
                    elif len(idArgs) == 3:
                       evalFunc = '{0}( A_{1}, B_{2}, C_{3} )'.format(funcName, idArgs[0], idArgs[1], idArgs[2] )
                    numpyFunc = 'np.' + evalFunc
                else:
                    try: evalFunc = evalFunc.replace( '$DTYPE1', idArgs[0] )
                    except: pass
                    try: evalFunc = evalFunc.replace( '$DTYPE2', idArgs[1] )
                    except: pass
                    # Check if evalFunc is a call or a binop/boolop/comparison.
                    # In the case of a call we need to prepend a 'np.'
                    numpyFunc = evalFunc
                    #if '(' in numpyFunc:
                        # Probably it's better just to make the AUTOTEST_DICT 
                        # values into tuples, one for each syntax.  
                        # Or import the exceptions direclty into the 
                        # autotest namespace? 
                        #numpyFunc = 'np.' + numpyFunc
                
                ##### SPECIAL CASES #####
                if funcName == 'complex':
                    evalFunc = 'complex( A_{}, B_{} )'.format(idArgs[0], idArgs[1] ) 
                    numpyFunc = 'A_{} + 1j*B_{}'.format(idArgs[0], idArgs[1] ) 
                
                testCode.append( "    def test_{}(self):\n".format(
                        funcNameUnique) )
                testCode.append( "        print('Test: out={0}')\n".format( 
                        evalFunc ) )
                testCode.append( "        out = np.empty_like( A_d, dtype='{}' )\n".format(retChar) )
                testCode.append( "        ne.evaluate('out={0}')\n".format(
                        evalFunc ) )
                testCode.append( "        np.testing.assert_array_almost_equal(out,{})\n".format( 
                        numpyFunc ) )
                evalFunc = None
                
        return ''.join(testCode)
    
    def __repr__(self):
        return ''.join( [str(self.py_Name),'_',str(self.libs)] )
    
def CastFactory( opsList, casting='safe' ):
    """
    Builds all the possible cast operations. Currently all casts are based on 
    the numpy framework. I'm unaware of any potential competiting standards 
    for Python.
    
    The main issue would seem to be if we want to support anything other than
    'safe' casting?  Cast styles are 'no', 'equiv', 'safe', 'same_kind', and 
    'unsafe'.
    
    Numpy dtype.chars are used for tuple lookup, as it's faster than parsing 
    through .descr:
         bool = '?'
         uint8 = 'B'
         int8 = 'b'
         uint16 = 'H'
         int16 = 'h'
         uint32 = 'I'
         int32 = 'i'
         uint64 = 'L'
         int64 = 'l'
         float32 = 'f'
         float64 = 'd'
         longfloat = 'g'
         complex64 = 'F'
         complex128 = 'D'
         bytes = 'S'
         str/unicode = 'U'
         
    longfloat and clongfloat are disabled at present.
    """
    global OP_COUNT
    
    for dChar in ALL_FAM:
        for castChar in ALL_FAM:
            if dChar == castChar:
                continue
            if not np.can_cast( dChar, castChar, casting=casting ):
                continue

            # TEMPORARY: remove strings
            if 'S' in [dChar,castChar] or 'U' in [dChar,castChar]:
                continue
            
            # NumPy doesn't provide casts for real types to complex types so 
            # there are a number of special cases.
            elif castChar=='D' and dChar=='F':
                # 'F' -> 'D' is a special case, casting complex64 -> complex128
                opsList += [Operation( 'cast', 
                            '$DEST.real = (npy_float64)($ARG1).real; $DEST.imag=(npy_float64)($ARG1).imag', 
                            CAST_SAFE, castChar, dChar )]
            elif castChar=='F':
                # In 'safe' casting one can only cast to complex and not backward
                # and this always is allocated as = dValue + 1i*0.0
                opsList += [Operation( 'cast', 
                            '$DEST.real = (npy_float32)($ARG1); $DEST.imag=0.0', 
                            CAST_SAFE, castChar, dChar )]
            elif castChar=='D':
                opsList += [Operation( 'cast', 
                            '$DEST.real = (npy_float64)($ARG1); $DEST.imag=0.0', 
                            CAST_SAFE, castChar, dChar )]
            else:
                # castOp = Operation( 'cast', '$DEST = ($DTYPE0)($ARG1)', CAST_SAFE, castChar, dChar )
                opsList += [Operation( 'cast', 
                            '$DEST = ($DTYPE0)($ARG1)', 
                            CAST_SAFE, castChar, dChar )]
    pass


def OpsFactory( opsList ):
    """
    For anything that doesn't need special function handling.
    """
    global OP_COUNT

    # Constructor signature: 
    #    Operation( python_name template, [libs], [return_dchars], 
    #              [arg1_dchars], {[arg2_dchars], ... } )
    
    ###### Copy ######
    opsList += [ Operation( 'copy', '$DEST = $ARG1', (LIB_STD,), ALL_NUM, ALL_NUM ) ]
    # Casts are built in CastFactory().

    ###### Standard arithmatic operations ######
    opsList += [Operation( ast.Add, '$DEST = $ARG1 + $ARG2', (LIB_STD,),
                      REAL_NUM, REAL_NUM, REAL_NUM )]
            
    # For complex functions, complex_functions.hpp was vectorized.
    opsList += [Operation( ast.Sub, '$DEST = $ARG1 - $ARG2', (LIB_STD,),
                      ALL_INT+DECIMAL, ALL_INT+DECIMAL, ALL_INT+DECIMAL )]
    
    opsList += [Operation( ast.Mult ,'$DEST = $ARG1 * $ARG2', (LIB_STD,),
                      REAL_NUM, REAL_NUM, REAL_NUM )]
    
    # Division in NumPy typically returns float64 for integers.
    # floordivide() is available for integer division.
    # * In NE2 division tried to out-smart the compiler with a ternary
    #   operation but it's easier to let the compiler determine when a 
    #   INFINITY or NAN result is generated.
    opsList +=[Operation( ast.Div, '$DEST = (npy_float64)$ARG1 / (npy_float64)$ARG2',
                          (LIB_STD,), ['d']*len(BOOL+ALL_INT), BOOL+ALL_INT, BOOL+ALL_INT )]
    opsList +=[Operation( ast.Div, '$DEST = $ARG1 / $ARG2', (LIB_STD,),
                      DECIMAL, DECIMAL, DECIMAL )]
    
    ###### Mathematical functions ######
    # TODO: How to handle integer pow in a 'nice' way? We don't want to do it 
    # inside Python as with Ne2 as that's a big slow function.
    opsList += [Operation( ast.Pow, '$DEST = pow($ARG1, $ARG2)', (LIB_STD,),
                      DECIMAL, DECIMAL, DECIMAL)]
    
    # The fancy method for floating-point modulo does not work nicely for 
    # integers.  In fact even the C-standard 'x1 % x2' is faulting.
    #opsList += [Operation( ast.Mod, '$DEST = $ARG1 % $ARG2', (LIB_STD,),
    #                  ALL_INT, ALL_INT, ALL_INT )]
    opsList += [Operation( ast.Mod, '$DEST = $ARG1 - floor($ARG1/$ARG2) * $ARG2', (LIB_STD,),
                      DECIMAL, DECIMAL, DECIMAL )]

    opsList += [Operation( 'where', '$DEST = $ARG1 ? $ARG2 : $ARG3', (LIB_STD,),
                      ALL_NUM, ['?']*len(ALL_NUM), ALL_NUM, ALL_NUM )]
    
    opsList += [Operation( 'ones_like', '$DEST = 1', (LIB_STD,),
                     REAL_NUM, REAL_NUM )]
    
    opsList += [Operation( ast.USub, '$DEST = -$ARG1', (LIB_STD,),
                     SIGNED_NUM, SIGNED_NUM)]
    
    ###### Bitwise Operations ######
    opsList += [Operation( ast.LShift, '$DEST = $ARG1 << $ARG2', (LIB_STD,), 
                      ALL_INT, ALL_INT, ALL_INT )]
    opsList += [Operation( ast.RShift, '$DEST = $ARG1 >> $ARG2', (LIB_STD,), 
                      ALL_INT, ALL_INT, ALL_INT )]

    opsList += [Operation( ast.BitAnd, '$DEST = ($ARG1 & $ARG2)', (LIB_STD,),
                      BITWISE_NUM, BITWISE_NUM, BITWISE_NUM )]
    opsList += [Operation( ast.BitOr, '$DEST = ($ARG1 | $ARG2)', (LIB_STD,),
                      BITWISE_NUM, BITWISE_NUM, BITWISE_NUM  )]
    opsList += [Operation( ast.BitXor, '$DEST = ($ARG1 ^ $ARG2)', (LIB_STD,),
                      BITWISE_NUM, BITWISE_NUM, BITWISE_NUM  )]
    
    ###### Logical Operations ######
    opsList += [Operation( 'logical_and', '$DEST = ($ARG1 && $ARG2)', (LIB_STD,),
                      BOOL, BOOL, BOOL, alias=ast.And )]
    opsList += [Operation( 'logical_or', '$DEST = ($ARG1 || $ARG2)', (LIB_STD,),
                      BOOL, BOOL, BOOL, alias=ast.Or )]
    # TODO: complex and string comparisons
    opsList += [Operation( ast.Gt, '$DEST = ($ARG1 > $ARG2)', (LIB_STD,), 
                     ['?']*len(REAL_NUM), REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.GtE, '$DEST = ($ARG1 >= $ARG2)', (LIB_STD,), 
                     ['?']*len(REAL_NUM), REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.Lt, '$DEST = ($ARG1 < $ARG2)', (LIB_STD,), 
                     ['?']*len(REAL_NUM), REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.LtE, '$DEST = ($ARG1 <= $ARG2)', (LIB_STD,), 
                     ['?']*len(REAL_NUM), REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.Eq, '$DEST = ($ARG1 == $ARG2)', (LIB_STD,), 
                     ['?']*len(REAL_NUM), REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.NotEq, '$DEST = ($ARG1 != $ARG2)', (LIB_STD,), 
                     ['?']*len(REAL_NUM), REAL_NUM, REAL_NUM )]
    
    ###### Complex operations ######
    # All all in function format
    
    ###### String operations ######
    # TODO: add unicode
    #opsList += [Operation( 'contains', '$DEST = stringcontains($ARG1, $ARG2, ss1, ss2)', (LIB_STD,), 
    #                 BOOL, STRINGS, STRINGS )]
    
    ###### Reductions ######
    # TODO
    
    return


NUMPY_VML_PRE = { 'd': 'vd', 'f':'vs', 'F':'vz', 'D':'vc' }
def FunctionFactory( opsList, C11=True, mkl=False ):
    '''
    Functions are declinated from operations in cases where the name of the 
    function might change with the library and the dtype.  Therefore where 
    possible we use simple rules to build functions, but tables are required.
    
    cmath.h functions should be overloaded for modern implementations. Some 
    pre-C++/11 implementations (e.g. MSVC), have appended 'f's for the 
    single-precision version. Cmath funcs return the value, i.e. they are not 
    vectorized, but they are usually inlined.
    
    NumExpr complex funtions are prepended by: nc_{function},
        e.g. nc_conj()
    and the return is the last argument.  They are now vectorized, like VML 
    functions, so they need the number of iterators as the first argument.
    
    Intel VML functions are prepended by: v{datatype}{Function},
        e.g. vfSin()
    and the return is the last argument
    
    '''
    global OP_COUNT
    
    ####################
    # TODO: need to find a copy of cmath for MSVC compiler.
    # TODO: extend msvc_function_stubs to add all the new functions as overloads.
    opsList += [ Operation( 'abs', '$DEST = $ARG1 < 0 ? -$ARG1 : $ARG1', LIB_STD,
                   SIGNED_NUM, SIGNED_NUM, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'arccos', '$DEST = acos($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]       
    opsList += [ Operation( 'arcsin', '$DEST = asin($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]           
    opsList += [ Operation( 'arctan', '$DEST = atan($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
    opsList += [ Operation( 'arctan2', '$DEST = atan2($ARG1, $ARG2)', LIB_STD,
                   DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
    opsList += [ Operation( 'ceil', '$DEST = ceil($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]            
    opsList += [ Operation( 'cos', '$DEST = cos($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]           
    opsList += [ Operation( 'cosh', '$DEST = cosh($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]       
    opsList += [ Operation( 'exp', '$DEST = exp($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]         
    opsList += [ Operation( 'fabs', '$DEST = fabs($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
    opsList += [ Operation( 'floor', '$DEST = floor($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
    opsList += [ Operation( 'fmod', '$DEST = fmod($ARG1, $ARG2)', LIB_STD,
                   DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
    
    # These are tricky, frexp and ldexp are using ARG2 as return pointers... 
    # We don't support multiple returns at present.
    #opsList += [ Operation( 'frexp', '$DEST = frexp($ARG1, $ARG2)', LIB_STD,
    #               DECIMAL,DECIMAL, ['i','i'] ) ] 
    #opsList += [ Operation( 'ldexp', '$DEST = ldexp($ARG1, $ARG2)', LIB_STD,
    #               DECIMAL,DECIMAL, ['i','i'] ) ]     
    opsList += [ Operation( 'log', '$DEST = log($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]    
    opsList += [ Operation( 'log10', '$DEST = log10($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
    # Here ints are supported, which is something we don't have in our ast.Pow operation...
    #opsList += [ Operation( 'power', '$DEST = pow($ARG1, $ARG2)', LIB_STD,
    #                DECIMAL+DECIMAL, DECIMAL+DECIMAL, DECIMAL+INTx2, vecType=TYPE_LOOP ) ] 
    
    opsList += [ Operation( 'sin', '$DEST = sin($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'sinh', '$DEST = sinh($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'sqrt', '$DEST = sqrt($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'tan', '$DEST = tan($ARG1)', LIB_STD,
                   DECIMAL,  DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'tanh', '$DEST = tanh($ARG1)', LIB_STD,
                   DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'fpclassify', '$DEST = fpclassify($ARG1)', LIB_STD,
                    INTx2, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'isfinite', '$DEST = isfinite($ARG1)', LIB_STD,
                    BOOLx2, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'isinf', '$DEST = isinf($ARG1)', LIB_STD,
                    BOOLx2, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'isnan', '$DEST = isnan($ARG1)', LIB_STD,
                    BOOLx2, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'isnormal', '$DEST = isnormal($ARG1)', LIB_STD,
                    BOOLx2, DECIMAL, vecType=TYPE_LOOP ) ]
    opsList += [ Operation( 'signbit', '$DEST = signbit($ARG1)', LIB_STD,
                    BOOLx2, DECIMAL, vecType=TYPE_LOOP ) ]
    
    ####################
    # C++/11 overloads #
    ####################
    if bool(C11):
        opsList += [ Operation( 'arccosh', '$DEST = acosh($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'arcsinh', '$DEST = asinh($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'arctanh', '$DEST = atanh($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'cbrt', '$DEST = cbrt($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'copysign', '$DEST = copysign($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]   
        opsList += [ Operation( 'erf', '$DEST = erf($ARG1)', LIB_STD,
                       DECIMAL ,DECIMAL, vecType=TYPE_LOOP ) ]  
        opsList += [ Operation( 'erfc', '$DEST = erfc($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'exp2', '$DEST = exp2($ARG1)', LIB_STD,
                       DECIMAL ,DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'expm1', '$DEST = expm1($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'fdim', '$DEST = fdim($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
        opsList += [ Operation( 'fma', '$DEST = fma($ARG1, $ARG2, $ARG3)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]     
        opsList += [ Operation( 'fmax', '$DEST = fmax($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]    
        opsList += [ Operation( 'fmin', '$DEST = fmin($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
        opsList += [ Operation( 'hypot', '$DEST = hypot($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
        opsList += [ Operation( 'ilogb', '$DEST = ilogb($ARG1)', LIB_STD,
                        INTx2, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'lgamma', '$DEST = lgamma($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        # We don't support long long funcs at present
        opsList += [ Operation( 'log1p', '$DEST = log1p($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'log2', '$DEST = log2($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'logb', '$DEST = logb($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'lrint', '$DEST = lrint($ARG1)', LIB_STD,
                        LONGx2, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'lround', '$DEST = lround($ARG1)', LIB_STD,
                        LONGx2, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'nearbyint', '$DEST = nearbyint($ARG1)', LIB_STD,
                        LONGx2, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'nextafter', '$DEST = nextafter($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
        opsList += [ Operation( 'nexttoward', '$DEST = nexttoward($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
        # The NumPy remainder is mod(), whereas C99's remainder is fmod(), so 
        # don't include it as both are already provided and it just creates 
        # confusion.
        #opsList += [ Operation( 'remainder', '$DEST = remainder($ARG1, $ARG2)', LIB_STD,
        #               DECIMAL, DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ] 
        # Th int in remquo() is a return pointer that we don't support at present.
        #opsList += [ Operation( 'remquo', '$DEST = remquo($ARG1, $ARG2, $ARG3)', LIB_STD,
        #               DECIMAL,DECIMAL,DECIMAL, ['i','i'] ) ] 
    
        opsList += [ Operation( 'rint', '$DEST = rint($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'round', '$DEST = round($ARG1)', LIB_STD,
                        INTx2, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'scalbln', '$DEST = scalbln($ARG1, $ARG2)', LIB_STD,
                       DECIMAL, DECIMAL, LONGx2, vecType=TYPE_LOOP ) ] 
        opsList += [ Operation( 'tgamma', '$DEST = tgamma($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]
        opsList += [ Operation( 'trunc', '$DEST = trunc($ARG1)', LIB_STD,
                       DECIMAL, DECIMAL, vecType=TYPE_LOOP ) ]    
        
    ###############################################
    # Complex number functions (from complex.hpp) #
    ###############################################
    opsList += [ Operation( 'complex', 'nc_complex(block_size, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, DECIMAL, DECIMAL, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'real', 'nc_real(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, DECIMAL, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'imag', 'nc_imag(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, DECIMAL, COMPLEX, vecType=TYPE_ALIGNED ) ]
    
    opsList += [ Operation( 'abs', 'nc_abs(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, DECIMAL, COMPLEX, vecType=TYPE_ALIGNED ) ] 
    opsList += [ Operation( 'abs2', 'nc_abs2(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, DECIMAL, COMPLEX, vecType=TYPE_ALIGNED ) ] 
    opsList += [ Operation( ast.Add, 'nc_add(block_size, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( ast.Sub, 'nc_sub(block_size, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( ast.Mult, 'nc_mul(block_size, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( ast.Div, 'nc_div(block_size, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( ast.USub, 'nc_neg(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'neg', 'nc_neg(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'conj', 'nc_conj(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]

    opsList += [ Operation( 'conj', 'fconj(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, DECIMAL, DECIMAL, vecType=TYPE_ALIGNED ) ]  
         
    opsList += [ Operation( 'sqrt', 'nc_sqrt(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'log', 'nc_log(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ] 
    opsList += [ Operation( 'log1p', 'nc_log1p(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]                        
    opsList += [ Operation( 'log10', 'nc_log10(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]         
    opsList += [ Operation( 'exp', 'nc_exp(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]  
    opsList += [ Operation( 'expm1', 'nc_expm1(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ] 
    # TODO: add aliases for 'power'
    opsList += [ Operation( ast.Pow, 'nc_pow(block_size, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]  
    opsList += [ Operation( 'arccos', 'nc_acos(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'arccosh', 'nc_acosh(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]    
    opsList += [ Operation( 'arcsin', 'nc_asin(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'arcsinh', 'nc_asinh(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]   
    opsList += [ Operation( 'arctan', 'nc_atan(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'arctanh', 'nc_atanh(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]
    opsList += [ Operation( 'cos', 'nc_cos(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ] 
    opsList += [ Operation( 'cosh', 'nc_cosh(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]               
    opsList += [ Operation( 'sin', 'nc_sin(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ] 
    opsList += [ Operation( 'sinh', 'nc_sinh(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]       
    opsList += [ Operation( 'tan', 'nc_tan(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ] 
    opsList += [ Operation( 'tanh', 'nc_tanh(block_size, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD, COMPLEX, COMPLEX, vecType=TYPE_ALIGNED ) ]

               
    ##################################################################
    # Intel Vector Math Library functions (from mkl_vml_functions.h) #
    ##################################################################         
    if bool(mkl):
        # Let's try for just some hand-crafted functions to start.
        #vmlFuncs = []
        #vmlFuncs += [ Operation('abs', 'Abs( (MKL_INT)block_size, (double *)x1, (double *)dest)', LIB_VML,
        #                [ 'd',], [ 'd',] ) ]
        #vmlFuncs += [ Operation( ast.Add, 'Add( (MKL_INT)block_size, (double *)x1, (double *)x2, (double *)dest)', LIB_VML,
        #                ['d',], ['d',], ['d',] ) ]

        pass
    return

    
def generate( body_stub='interp_body_stub.cpp', header_stub='interp_header_stub.hpp', 
             blocksize=(4096,32), bounds_check=True, mkl=False, C11=True ):
    """
    generate() is called by setup.py, it generates interp_body_GENERATED.cpp, 
    which  contains all the expanded operations and the jump table used by the 
    interpreter. Similarly the opword_target is also included in the 
    interpreter and it provides the enums.
    
    We want to call generate with each build so that we do not suffer any 
    accidental gaps in the jump table.
    """
    global INTERP_HEADER_DEFINES, BLOCKSIZE
    
    # Set globals for passing blocksize to variables.
    # Not sure what blocksize[1] is actually used for in NumExpr2
    BLOCKSIZE = blocksize[0]
    
    INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
                             '#define BLOCK_SIZE1 {}\n'.format(blocksize[0]), 
                             '#define BLOCK_SIZE2 {}\n'.format(blocksize[1])] )
    
    # MKL support
    if bool(mkl):
        INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
                             '#define USE_VML\n' ] )

    ###### Insert bounds-check #######
    # BOUNDS_CHECK is used in interp_body.cpp
    if bool( bounds_check ):
        #INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
        #'#define BOUNDS_CHECK(arg) if ((arg) >= params->n_reg) { *pc_error = pc; return -2; }\n',] )
        INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
           '#define BOUNDS_CHECK(arg) if ((arg) >= params->n_reg) { return -2; }\n',] )
    else:
        INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
        '#define BOUNDS_CHECK(arg)\n',] )
    
    pythonTable = OrderedDict()
    # NO_OP
    pythonTable[('',LIB_STD,'')] = (struct.pack(NE_STRUCT, 0), '')
    cTable = OrderedDict()
    cFuncs = []
    opsList = []
    CastFactory( opsList )
    OpsFactory( opsList )
    FunctionFactory( opsList, mkl=mkl, C11=C11 )
    
    for op in opsList:
        for I, opNum in enumerate( op.opNums ):
            cTable[opNum] = op.c_OpTableEntrys[I]
            pythonTable[op.py_TupleKeys[I]] = (struct.pack( NE_STRUCT, opNum ), op.retFam[I])
            cFuncs.append( op.c_FunctionImpls[I] )
    
    # Write #define OP_END 
    OP_END = next(OP_COUNT) -1
    INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
        '#define OP_END {}\n'.format(OP_END) ] )

    ###### Write to functions_GENERATED.cpp ######        
    with open( os.path.join( NE_DIR, 'functions_GENERATED.cpp'), 'w' ) as f_body:
        f_body.write( '#include "numexpr_object.hpp"\n\n' )
        for funcBody in cFuncs:
            f_body.write( funcBody + '\n\n' )
            
    ###### Write to interp_body_stub.cpp ######
    with open( os.path.join(NE_DIR, body_stub ), 'r' ) as stub:
        bodyPrior, bodyPost = stub.read().split( INSERT_POINT )
    
    generatedBody = ''.join( [fragment for fragment in cTable.values()] )
    generatedBody = ''.join( [WARNING_EDIT, bodyPrior, generatedBody, 
                              WARNING_END, bodyPost,] )
    
    with open( os.path.join(NE_DIR, 'interp_body_GENERATED.cpp' ), 'wb' ) as body:
        body.write( generatedBody.encode('ascii') )
        
    ###### Write to interpreter_stub.hpp ######
    with open( os.path.join(NE_DIR, header_stub ), 'r' ) as stub:
        headerPrior, headerPost = stub.read().split( INSERT_POINT )
    
    generatedHeader = ''.join( [WARNING_EDIT, headerPrior, 
                                INTERP_HEADER_DEFINES, WARNING_END, 
                                headerPost, ] )
    
    with open( os.path.join(NE_DIR, 'interp_header_GENERATED.hpp' ), 'wb' ) as body:
        body.write( generatedHeader.encode('ascii') )
        
    ###### Save the lookup dict for Python ######
    with open( os.path.join(NE_DIR, 'lookup.pkl' ), 'wb' ) as lookup:
        cPickle.dump( pythonTable, lookup )
        
    ###### Write autotest_GENERATED.py ######
    with open( os.path.join( NE_DIR, 'tests/autotest_stub.py'), 'r' ) as f_test:
        autotestHead, autotestTail = f_test.read().split( PY_INSERT_POINT )
        
    with open( os.path.join( NE_DIR, 'tests/autotest_GENERATED.py'), 'w' ) as f_test:
        f_test.write( autotestHead )
        for op in opsList:
            #for I, opNum in enumerate( op.opNums ):
            f_test.write( op.test_Auto )
            
        f_test.write( autotestTail )
            
    return pythonTable, cTable, cFuncs


if __name__ == '__main__':
    pythonTable, cTable, cFuncs = generate()
    
        
        