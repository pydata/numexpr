# -*- coding: utf-8 -*-
"""
NumExpr3 interpreter function generator

Created on Thu Jan  5 12:46:25 2017
@author: Robert A. McLeod
@email: robbmcleod@gmail.com

"""
import numpy as np
from collections import OrderedDict
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

# Datatype families
D = OrderedDict()

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

def STRING_EXPR( opNum, expr, retDtype, arg1Dtype=None, arg2Dtype=None, arg3Dtype=None ):
    # More custom operations
    # TODO
    pass

def DEST( index='J' ):
    # Real return is the default
    return 'dest[{0}]'.format( index )
    
def DEST_STR( dchar ):
    # TODO: difference between npy_string and npy_unicode?
    return '({0} *)dest + J*memsteps[store_in]'.format( DCHAR_TO_CTYPE[dchar] )

def ARG_OLD( dchar, num, index='J' ):
    return '(({0} *)( x{1} + J*sb{1} ))[{2}]'.format( DCHAR_TO_CTYPE[dchar], num, index )

def ARG( num, index='J' ):
    return 'x{0}[{1}]'.format( num, index )

def ARG_STRIDE( num, index='J' ):
    return 'x{0}[{1}*sb{0}]'.format( num, index )

def ARG_STR( dchar, num ):
    return '(({0} *)( x{1} + J*sb{1} ))'.format( DCHAR_TO_CTYPE[dchar], num )

def REDUCE( dchar, outerLoop=False ):
    if outerLoop:
        return DEST( dchar )
    # INNER LOOP
    return '*({0} *)dest'.format( DCHAR_TO_CTYPE[dchar] )

def VEC_LOOP( expr ):
    expr = expr.replace( '$ARG3', ARG(3) )
    expr = expr.replace( '$ARG2', ARG(2) )
    expr = expr.replace( '$ARG1', ARG(1) )
    return '''for(J = 0; J < BLOCK_SIZE; J++) { 
            EXPR; 
        }'''.replace( 'EXPR', expr ) 
 
def STRIDED_LOOP( expr ):
    expr = expr.replace( '$ARG3', ARG_STRIDE(3) )
    expr = expr.replace( '$ARG2', ARG_STRIDE(2) )
    expr = expr.replace( '$ARG1', ARG_STRIDE(1) )
    if 'x3' in expr:
        return '''sb1 /= sizeof($DTYPE1);
        sb2 /= sizeof($DTYPE1);
        sb3 /= sizeof($DTYPE1);
        for(J = 0; J < BLOCK_SIZE; J++) { 
            EXPR; 
        }'''.replace( 'EXPR', expr ) 
    elif 'x2' in expr:
        return '''sb1 /= sizeof($DTYPE1);
        sb2 /= sizeof($DTYPE1);
        for(J = 0; J < BLOCK_SIZE; J++) { 
            EXPR; 
        }'''.replace( 'EXPR', expr ) 
    elif 'x1' in expr:
        return '''sb1 /= sizeof($DTYPE1);
        for(J = 0; J < BLOCK_SIZE; J++) { 
            EXPR; 
        }'''.replace( 'EXPR', expr ) 
    else:
        raise ValueError( "Unknown number of arguments for strided array" )

def VEC_ARG0( expr ):
    return '''
    {
        BOUNDS_CHECK(store_in);
        $DTYPE0 *dest = ($DTYPE0 *)params->registers[store_in].mem;
        VEC_LOOP(expr);
    } break;
'''.replace( 'VEC_LOOP(expr);', VEC_LOOP(expr) )

# We could write a more general function suitable for any number of arguments,
# but that would make the generator code more opaque
def VEC_ARG1(expr):
    return '''
    {   
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    $DTYPE0 *dest = ($DTYPE0 *)params->registers[store_in].mem;
    $DTYPE1 *x1 = ($DTYPE1 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof($DTYPE1) ) { // Aligned
        VEC_LOOP(expr)
        break;
    } else { // Strided
       STRIDED_LOOP(expr)
    }
    } break;
'''.replace('STRIDED_LOOP(expr)', STRIDED_LOOP(expr) ).replace( 'VEC_LOOP(expr)', VEC_LOOP(expr) )

def VEC_ARG2(expr):
    return '''
    {
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
    } else { // Strided
       STRIDED_LOOP(expr)
    }
    } break;
'''.replace('STRIDED_LOOP(expr)', STRIDED_LOOP(expr) ).replace( 'VEC_LOOP(expr)', VEC_LOOP(expr) )


def VEC_ARG3(expr):
    return '''
    {
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
    } else { // Strided
        STRIDED_LOOP(expr)
    }
    } break;
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
        BOUNDS_CHECK(store_in);

        char *dest = params->registers[store_in].mem;
        EXPR;
    } break;
'''.replace( 'EXPR', expr )

def VEC_ARG1_ALIGNED(expr):
    return '''
    {
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        EXPR;
    } break;
'''.replace( 'EXPR', expr )

def VEC_ARG2_ALIGNED(expr):
    return '''
    {
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        EXPR;
    } break;
'''.replace( 'EXPR', expr )

def VEC_ARG3_ALIGNED(expr):
    return '''
    {
        BOUNDS_CHECK(store_in);
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        NE_REGISTER arg3 = params->program[pc].arg3;
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        BOUNDS_CHECK(arg3);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        char *x3 = params->registers[arg3].mem;
        EXPR;
    } break;
'''.replace( 'EXPR', expr )
VEC_ARGN_ALIGNED = { 0: VEC_ARG0_ALIGNED, 1: VEC_ARG1_ALIGNED, 2: VEC_ARG2_ALIGNED, 3: VEC_ARG3_ALIGNED }

# Strided expressions can iterate over non-unity steps, i.e. 2,4,3, 
# which is significantly slower, so in general it's best if there's a branch 
# inside each function: one for strided and one for aligned data.
def VEC_ARG0_STRIDED(expr):
    return '''
    {
        BOUNDS_CHECK(store_in);

        char *dest = params->registers[store_in].mem;
        
        EXPR;
    } break;
'''.replace( 'EXPR', expr )

def VEC_ARG1_STRIDED(expr):
    return '''
    {
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        npy_intp sb1 = params->registers[arg1].stride;
        
        EXPR;
    } break;
'''.replace( 'EXPR', expr )

def VEC_ARG2_STRIDED(expr):
    return '''
    {
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
    } break;
'''.replace( 'EXPR', expr )

def VEC_ARG3_STRIDED(expr):
    return '''
    {
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
    } break;
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
    docString = '// {} ({}) :: {}'.format(retChar, argChars, expr)
    if vecType == TYPE_LOOP:
        expr = 'case {0}: {2} {1}'.format( opNum, VEC_ARGN[len(argChars)](expr), docString )
    elif vecType == TYPE_ALIGNED:
        expr =  'case {0}: {2} {1}'.format( opNum, VEC_ARGN_ALIGNED[len(argChars)](expr), docString )
    elif vecType == TYPE_STRIDED:
        expr = 'case {0}: {2} {1}'.format( opNum, VEC_ARGN_STRIDED[len(argChars)](expr), docString )
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
'''
# These are some convience shortcuts for our operation hash-table below
BOOL = ['?']
SIGNED_INT = ['B','H','I','L']
UNSIGNED_INT = ['B','H','I','L']
DECIMAL  =  ['f','d']
COMPLEX = ['F','D']
STRINGS = ['S','U']
ALL_INT = SIGNED_INT + UNSIGNED_INT
REAL_NUM = BOOL + ALL_INT + DECIMAL
SIGNED_NUM = SIGNED_INT + DECIMAL
BITWISE_NUM = BOOL + ALL_INT
ALL_NUM = BOOL + ALL_INT + DECIMAL + COMPLEX    
ALL_FAM = ALL_NUM + STRINGS

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
    
    def __init__(self, py_Name, c_Template, libs, retFam, *argFams, vecType=TYPE_LOOP ):
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
        
        self.opNums = []
        self.c_FunctionHandles = []
        self.py_TupleKeys = []
        self.c_FunctionCodes = []
        self.c_OpTableEntrys = []
        
        self.build()
        
    def build(self):
        for lib in self.libs:
            for I, retChar in enumerate( self.retFam ):
                opNum = next( OP_COUNT )
                
                self.opNums.append( opNum )
                passedArgs = [ arg[I] for arg in self.argFams ]
                
                self.c_OpTableEntrys.append( 
                        EXPR( opNum, self.c_Template, retChar, *passedArgs, 
                              vecType=self.vecType ) )
                # The format for the Python tuple is (name,{lib},returntype,arg1type,...)
                self.py_TupleKeys.append( 
                        tuple( [self.py_Name, lib, retChar] + passedArgs) )
                
                # Debugging output
                #print( '##### %s #####' % opNum )
                #print( cTable[opNum] )
        
    def __repr__(self):
        return ''.join( [str(self.py_Name),'_',str(self.libs)] )
    
def CastFactory( pythonTable, cTable, casting='safe' ):
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
            if not np.can_cast( dChar, castChar, casting=casting ):
                continue
            # Remove any unwanted types here
            if 'F' in [dChar,castChar] or 'D' in [dChar,castChar]:
                # TODO: we need some .real magic in the cast definitions
                continue
            # TEMPORARY: remove strings
            if 'S' in [dChar,castChar] or 'U' in [dChar,castChar]:
                continue
            
            castOp = Operation( 'cast', '$DEST = ($DTYPE0)($ARG1)', CAST_SAFE, castChar, dChar )
            cTable[ castOp.opNums[0] ] = castOp.c_OpTableEntrys[0]
            pythonTable[ castOp.py_TupleKeys[0] ] = castOp.opNums[0]
    pass


def OpsFactory( pythonTable, cTable ):
    """
    For anything that doesn't need special function handling.
    """
    global OP_COUNT
    opsList = []
    # Constructor signature: 
    #    Operation( python_name template, [libs], [return_dchars], 
    #              [arg1_dchars], {[arg2_dchars], ... } )
    
    ###### Comparison ######
    # TODO: Probably array assignment is faster than memcpy x loop in the strided case.
    opsList += [ Operation( 'copy', 'memcpy(&$DEST, ((char *)x1+J*sb1), sizeof($DTYPE1))', (LIB_STD,),
                         ALL_NUM, ALL_NUM ) ]
    # Casts are built in CastFactory().

    ###### Standard arithmatic operations ######
    opsList += [Operation( ast.Add, '$DEST = $ARG1 + $ARG2', (LIB_STD,),
                      REAL_NUM, REAL_NUM, REAL_NUM )]
            
    # For complex functions, complex_functions.hpp was vectorized.
    opsList += [Operation( ast.Sub, '$DEST = $ARG1 - $ARG2', (LIB_STD,),
                      REAL_NUM, REAL_NUM, REAL_NUM )]
    
    opsList += [Operation( ast.Mult ,'$DEST = $ARG1 * $ARG2', (LIB_STD,),
                      REAL_NUM, REAL_NUM, REAL_NUM )]
    
    opsList +=[Operation( ast.Div, '$DEST = $ARG2 ? ($ARG1 / $ARG2) : 0', (LIB_STD,),
                      REAL_NUM, REAL_NUM, REAL_NUM )]
    
    ###### Mathematical functions ######
    # TODO: How to handle integer pow in a 'nice' way? We don't want to do it 
    # inside Python as with Ne2 as that's a big slow function.
    opsList += [Operation( ast.Pow, '$DEST = pow($ARG1, $ARG2)', (LIB_STD,),
                      DECIMAL, DECIMAL, DECIMAL)]
    
    opsList += [Operation( ast.Mod, '$DEST = $ARG1 - floor($ARG1/$ARG2) * $ARG2', (LIB_STD,),
                      REAL_NUM, REAL_NUM, REAL_NUM )]

    opsList += [Operation( 'where', '$DEST = $ARG1 ? $ARG2 : $ARG3', (LIB_STD,),
                      REAL_NUM, ['?']*len(REAL_NUM), REAL_NUM, REAL_NUM  )]
    
    opsList += [Operation( 'ones_like', '$DEST = 1', (LIB_STD,),
                     REAL_NUM )]
    
    opsList += [Operation( 'neg', '$DEST = -$ARG1', (LIB_STD,),
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
    opsList += [Operation( ast.And, '$DEST = ($ARG1 && $ARG2)', (LIB_STD,),
                      BOOL, BITWISE_NUM, BITWISE_NUM )]
    opsList += [Operation( ast.Or, '$DEST = ($ARG1 || $ARG2)', (LIB_STD,),
                      BOOL, BITWISE_NUM, BITWISE_NUM )]
    # TODO: complex and string comparisons
    opsList += [Operation( ast.Gt, '$DEST = ($ARG1 > $ARG2)', (LIB_STD,), 
                     REAL_NUM, REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.GtE, '$DEST = ($ARG1 >= $ARG2)', (LIB_STD,), 
                     REAL_NUM, REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.Lt, '$DEST = ($ARG1 < $ARG2)', (LIB_STD,), 
                     REAL_NUM, REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.LtE, '$DEST = ($ARG1 <= $ARG2)', (LIB_STD,), 
                     REAL_NUM, REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.Eq, '$DEST = ($ARG1 == $ARG2)', (LIB_STD,), 
                     REAL_NUM, REAL_NUM, REAL_NUM )]
    opsList += [Operation( ast.NotEq, '$DEST = ($ARG1 != $ARG2)', (LIB_STD,), 
                     REAL_NUM, REAL_NUM, REAL_NUM )]
    
    ###### Complex operations ######
    # All all in function format
    
    ###### String operations ######
    # TODO: add unicode
    #opsList += [Operation( 'contains', '$DEST = stringcontains($ARG1, $ARG2, ss1, ss2)', (LIB_STD,), 
    #                 BOOL, STRINGS, STRINGS )]
    
    ###### Reductions ######
    # TODO
    
    # Build operations
    for operation in opsList:
        for I, opNum in enumerate( operation.opNums ):
            cTable[opNum] = operation.c_OpTableEntrys[I]
            pythonTable[operation.py_TupleKeys[I]] = opNum
                       
    return


NUMPY_VML_PRE = { 'd': 'vd', 'f':'vs', 'F':'vz', 'D':'vc' }
def FunctionFactory( pythonTable, cTable, C11=True, mkl=False ):
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

    ####################
    # TODO: need to find a copy of cmath for MSVC compiler.
    # TODO: extend msvc_function_stubs to add all the new functions as overloads.
    cmathOverloadedFuncs = []
    cmathOverloadedFuncs += [ Operation( 'abs', '$DEST = abs($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'arccos', '$DEST = acos($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]       
    cmathOverloadedFuncs += [ Operation( 'arcsin', '$DEST = asin($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]           
    cmathOverloadedFuncs += [ Operation( 'arctan', '$DEST = atan($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    cmathOverloadedFuncs += [ Operation( 'arctan2', '$DEST = atan2($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    cmathOverloadedFuncs += [ Operation( 'ceil', '$DEST = ceil($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]            
    cmathOverloadedFuncs += [ Operation( 'cos', '$DEST = cos($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]           
    cmathOverloadedFuncs += [ Operation( 'cosh', '$DEST = cosh($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]       
    cmathOverloadedFuncs += [ Operation( 'exp', '$DEST = exp($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]         
    cmathOverloadedFuncs += [ Operation( 'fabs', '$DEST = fabs($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    cmathOverloadedFuncs += [ Operation( 'floor', '$DEST = floor($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    cmathOverloadedFuncs += [ Operation( 'fmod', '$DEST = fmod($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    
    # These are tricky, frexp and ldexp are using ARG2 as return pointers... 
    # We don't support multiple returns at present.
    #cmathOverloadedFuncs += [ Operation( 'frexp', '$DEST = frexp($ARG1, $ARG2)', LIB_STD,
    #               DECIMAL,DECIMAL, ['i','i'] ) ] 
    #cmathOverloadedFuncs += [ Operation( 'ldexp', '$DEST = ldexp($ARG1, $ARG2)', LIB_STD,
    #               DECIMAL,DECIMAL, ['i','i'] ) ]     
    cmathOverloadedFuncs += [ Operation( 'log', '$DEST = log($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]    
    cmathOverloadedFuncs += [ Operation( 'log10', '$DEST = log10($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    # Here ints are supported, which is something we don't have in our ast.Pow operation...
    cmathOverloadedFuncs += [ Operation( 'power', '$DEST = pow($ARG1, $ARG2)', LIB_STD,
                    ['f','d','f','d'], ['f','d','f','d'], ['f','d','i','i'], vecType=TYPE_LOOP ) ] 
    cmathOverloadedFuncs += [ Operation( 'sin', '$DEST = sin($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'sinh', '$DEST = sinh($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'sqrt', '$DEST = sqrt($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'tan', '$DEST = tan($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'tanh', '$DEST = tanh($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'fpclassify', '$DEST = fpclassify($ARG1)', LIB_STD,
                    ['i', 'i'],DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'isfinite', '$DEST = isfinite($ARG1)', LIB_STD,
                    ['?','?'],DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'isinf', '$DEST = isinf($ARG1)', LIB_STD,
                    ['?','?'],DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'isnan', '$DEST = isnan($ARG1)', LIB_STD,
                    ['?','?'],DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'isnormal', '$DEST = isnormal($ARG1)', LIB_STD,
                    ['?','?'],DECIMAL, vecType=TYPE_LOOP ) ]
    cmathOverloadedFuncs += [ Operation( 'signbit', '$DEST = signbit($ARG1)', LIB_STD,
                    ['?','?'],DECIMAL, vecType=TYPE_LOOP ) ]
    
    ####################
    # C++/11 overloads #
    ####################
    c11Funcs = []
    c11Funcs += [ Operation( 'arccosh', '$DEST = acosh($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'arccosh', '$DEST = acosh($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'arcsinh', '$DEST = asinh($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'arctanh', '$DEST = atanh($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'cbrt', '$DEST = cbrt($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'copysign', '$DEST = copysign($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]   
    c11Funcs += [ Operation( 'erf', '$DEST = erf($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]  
    c11Funcs += [ Operation( 'erfc', '$DEST = erfc($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'exp2', '$DEST = exp2($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'expm1', '$DEST = expm1($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'fdim', '$DEST = fdim($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    c11Funcs += [ Operation( 'fma', '$DEST = fma($ARG1, $ARG2, $ARG3)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]     
    c11Funcs += [ Operation( 'fmax', '$DEST = fmax($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]    
    c11Funcs += [ Operation( 'fmin', '$DEST = fmin($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    c11Funcs += [ Operation( 'hypot', '$DEST = hypot($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    c11Funcs += [ Operation( 'ilogb', '$DEST = ilogb($ARG1)', LIB_STD,
                    ['i','i'],DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'lgamma', '$DEST = lgamma($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'expm1', '$DEST = expm1($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    # We don't support long long funcs at present
    c11Funcs += [ Operation( 'log1p', '$DEST = log1p($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'log2', '$DEST = log2($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'logb', '$DEST = logb($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'lrint', '$DEST = lrint($ARG1)', LIB_STD,
                    ['l','l'],DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'lround', '$DEST = lround($ARG1)', LIB_STD,
                    ['l','l'],DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'nearbyint', '$DEST = nearbyint($ARG1)', LIB_STD,
                    ['l','l'],DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'nextafter', '$DEST = nextafter($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    c11Funcs += [ Operation( 'nexttoward', '$DEST = nexttoward($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
    c11Funcs += [ Operation( 'remainder', '$DEST = remainder($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ] 
        
    # Th int in remquo() is a return pointer that we don't support at present.
    #c11Funcs += [ Operation( 'remquo', '$DEST = remquo($ARG1, $ARG2, $ARG3)', LIB_STD,
    #               DECIMAL,DECIMAL,DECIMAL, ['i','i'] ) ] 
    c11Funcs += [ Operation( 'rint', '$DEST = rint($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'round', '$DEST = round($ARG1)', LIB_STD,
                    ['i', 'i'],DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'scalbln', '$DEST = scalbln($ARG1, $ARG2)', LIB_STD,
                   DECIMAL,DECIMAL, ['l','l'], vecType=TYPE_LOOP ) ] 
    c11Funcs += [ Operation( 'tgamma', '$DEST = tgamma($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]
    c11Funcs += [ Operation( 'trunc', '$DEST = trunc($ARG1)', LIB_STD,
                   DECIMAL,DECIMAL, vecType=TYPE_LOOP ) ]    
        
    ##################################################################
    # Test of vectorized operations
    ##################################################################
    # I'm curious how much of a difference including the striding in an operation
    # is?  Perhaps we should have non-strided and strided versions of each
    # major operation?
    stridedFuncs = []
    stridedFuncs += [ Operation( 'mul', 'ne_mul(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest, sb1, sb2)', 
                          LIB_STD, ['d'], ['d'], ['d'], vecType=TYPE_STRIDED ) ]   
    
    # Build functions
    if bool(C11):
            cmathOverloadedFuncs.extend( c11Funcs )
            
    cmathOverloadedFuncs.extend( stridedFuncs )
        
    for func in cmathOverloadedFuncs:
        for I, opNum in enumerate( func.opNums ):
            cTable[opNum] = func.c_OpTableEntrys[I]
            pythonTable[func.py_TupleKeys[I]] = opNum
    
    ###############################################
    # Complex number functions (from complex.hpp) #
    ###############################################
    zFuncs = []
    zFuncs += [ Operation( 'abs', 'nc_abs(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,DECIMAL,COMPLEX, vecType=TYPE_ALIGNED ) ] 
    zFuncs += [ Operation( ast.Add, 'nc_add(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( ast.Sub, 'nc_sub(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( ast.Mult, 'nc_mul(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( ast.Div, 'nc_div(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'neg', 'nc_neg(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'conj', 'nc_conj(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'conj', 'fconj(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,DECIMAL,DECIMAL, vecType=TYPE_ALIGNED ) ]           
    zFuncs += [ Operation( 'sqrt', 'nc_sqrt(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'log', 'nc_log(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ] 
    zFuncs += [ Operation( 'log1p', 'nc_log1p(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]                        
    zFuncs += [ Operation( 'log10', 'nc_log10(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]         
    zFuncs += [ Operation( 'exp', 'nc_exp(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]  
    zFuncs += [ Operation( 'expm1', 'nc_expm1(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ] 
    # TODO: add aliases for 'power'
    zFuncs += [ Operation( ast.Pow, 'nc_pow(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE2 *)x2, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]  
    zFuncs += [ Operation( 'arccos', 'nc_acos(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'arccosh', 'nc_acosh(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]    
    zFuncs += [ Operation( 'arcsin', 'nc_asin(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'arcsinh', 'nc_asinh(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]   
    zFuncs += [ Operation( 'arctan', 'nc_atan(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'arctanh', 'nc_atanh(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]
    zFuncs += [ Operation( 'cos', 'nc_cos(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ] 
    zFuncs += [ Operation( 'cosh', 'nc_cosh(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]               
    zFuncs += [ Operation( 'sin', 'nc_sin(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ] 
    zFuncs += [ Operation( 'sinh', 'nc_sinh(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]       
    zFuncs += [ Operation( 'tan', 'nc_tan(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ] 
    zFuncs += [ Operation( 'tanh', 'nc_tanh(BLOCK_SIZE, ($DTYPE1 *)x1, ($DTYPE0 *)dest)', 
                          LIB_STD,COMPLEX,COMPLEX, vecType=TYPE_ALIGNED ) ]    

    for func in zFuncs:
        for I, opNum in enumerate( func.opNums ):
            cTable[opNum] = func.c_OpTableEntrys[I]
            pythonTable[func.py_TupleKeys[I]] = opNum
           
               
    ##################################################################
    # Intel Vector Math Library functions (from mkl_vml_functions.h) #
    ##################################################################         
    if bool(mkl):
        # Let's try for just some hand-crafted functions to start.
        #vmlFuncs = []
        #vmlFuncs += [ Operation('abs', 'Abs( (MKL_INT)BLOCK_SIZE, (double *)x1, (double *)dest)', LIB_VML,
        #                [ 'd',], [ 'd',] ) ]
        #vmlFuncs += [ Operation( ast.Add, 'Add( (MKL_INT)BLOCK_SIZE, (double *)x1, (double *)x2, (double *)dest)', LIB_VML,
        #                ['d',], ['d',], ['d',] ) ]
    
        # TODO: we could also make aliases so that VML funcs could be called 
        # with the LIB_STD?  Such as,
        # (sqrt_vml, LIB_STD, 'f', 'f')
        for func in vmlFuncs:
            for J in np.arange( len(func.retFam) ):
                opNum = next(OP_COUNT)
                pendedTemplate =  NUMPY_VML_PRE[func.argFams[0][J]] + func.c_Template
                
                if len( func.argFams ) == 1:
                    cTable[opNum] = EXPR( opNum, pendedTemplate, \
                                  func.retFam[J], func.argFams[0][J], 
                                  vecType=TYPE_ALIGNED )
                    pythonTable[(func.py_Name, LIB_VML, func.retFam[J], func.argFams[0][J] )] \
                              = opNum
                elif len( func.argFams ) == 2:
                    cTable[opNum] = EXPR( opNum, pendedTemplate,         \
                                          func.retFam[J],              \
                                          func.argFams[0][J], \
                                          func.argFams[1][J], \
                                           vecType=TYPE_ALIGNED )
                    pythonTable[(func.py_Name, LIB_VML, func.retFam[J],   \
                                 func.argFams[0][J], func.argFams[1][J] )] \
                              = opNum
                              
                #print( '##### %s #####' % opNum )
                #print( cTable[opNum] )

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
        INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
        '#define BOUNDS_CHECK(arg) if ((arg) >= params->n_reg) { *pc_error = pc; return -2; }\n',] )
    else:
        INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
        '#define BOUNDS_CHECK(arg)\n',] )
    
    pythonTable = OrderedDict()
    cTable = OrderedDict()
    CastFactory( pythonTable, cTable )
    OpsFactory( pythonTable, cTable )
    FunctionFactory( pythonTable, cTable, mkl=mkl, C11=C11 )
    
    # Write #define OP_END 
    OP_END = next(OP_COUNT) -1
    INTERP_HEADER_DEFINES= "".join( [INTERP_HEADER_DEFINES,
        '#define OP_END {}\n'.format(OP_END) ] )
    
    
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
    # First pre-pack all the values into bytes
    for key, value in pythonTable.items():
        pythonTable[key] = struct.pack( NE_STRUCT, value )
    with open( os.path.join(NE_DIR, 'lookup.pkl' ), 'wb' ) as lookup:
        cPickle.dump( pythonTable, lookup )
    
    return pythonTable, cTable



if __name__ == '__main__':
    pythonTable, cTable = generate()
    

        
    
        
        