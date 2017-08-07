# -*- coding: utf-8 -*-
"""
NumExpr3 expression compiler
@author: Robert A. McLeod

Compared to NumExpr2 the compiler has been rebuilt to use the CPython module
`ast` to build the Abstract Syntax Tree (AST) representation of the statements,
instead of the NE AST syntax in the old expressions.py.

AST documentation is hosted externally:
    
https://greentreesnakes.readthedocs.io

The goal is to only parse the AST tree once to reduce the amount of time spent
in pure Python compared to NE2.
"""
import __future__

import os, inspect, sys
import ast
import numpy as np
from collections import defaultdict, OrderedDict

# The operations are saved on disk as a pickled dict
try: import cPickle as pickle
except ImportError: import pickle
# For appending long strings, using a buffer is faster than ''.join()
try: from cStringIO import StringIO as BytesIO
except ImportError: from io import BytesIO
# struct.pack is the quickest way to build the program as structs
# All important format characters: https://docs.python.org/2/library/struct.html
from struct import pack, unpack, calcsize


if sys.version_info[0] >= 3:
    # Python 2 to 3 handling
    unicode = str # To suppress warnings only
else:
    pass

# Due to global state in the C-API, we need to restrict the module to running 
# one expression per Python process.  There is, unfortunately, no timeout 
# in Python 2.7 for Lock
from threading import Lock
_NE_RUN_LOCK = Lock()

# interpreter.so/pyd:
try: 
    import interpreter # For debugging, release should import from .
except ImportError:
    from . import interpreter

# Import opTable
_neDir = os.path.dirname(os.path.abspath( inspect.getfile(inspect.currentframe()) ))
with open( os.path.join( _neDir, 'lookup.pkl' ), 'rb' ) as lookup:
    OPTABLE = pickle.load( lookup )
    
# Sizes for struct.pack
_NULL_REG =  pack( 'B', 255 )
_PACK_OP =  b'H'
_PACK_REG = b'B'
_UNPACK = b''.join([_PACK_OP,_PACK_REG,_PACK_REG,_PACK_REG,_PACK_REG])
    
# Context for casting
CAST_SAFE = 0
CAST_NO = 1
CAST_EQUIV = 2
CAST_SAME_KIND = 3
CAST_UNSAFE = 4
_CAST_TRANSLATIONS = { CAST_SAFE:CAST_SAFE, 'safe':CAST_SAFE, 
                        CAST_NO:CAST_NO, 'no':CAST_NO, 
                        CAST_EQUIV:CAST_EQUIV, 'equiv':CAST_EQUIV, 
                        CAST_SAME_KIND:CAST_SAME_KIND, 'same_kind':CAST_SAME_KIND,
                        CAST_UNSAFE:CAST_UNSAFE, 'unsafe':CAST_UNSAFE }

# Context for optimization effort
OPT_MODERATE = 0
#OPT_AGGRESSIVE = 1

# Defaults to LIB_STD
LIB_STD = 0    # C++ cmath standard library
#LIB_VML = 1    # Intel Vector Math library
#LIB_SIMD = 2  # x86 SIMD extensions

CHECK_ALL = 0  # Normal operation in checking dtypes
#CHECK_NONE = 1 # Disable all checks for dtypes if expr is in cache

# Note: int(np.isscalar( np.float32(0.0) )) is used to set REGKIND, 
# so ARRAY == 0 and SCALAR == 1
_REGKIND_ARRAY = 0
_REGKIND_SCALAR = 1
_REGKIND_TEMP = 2
_REGKIND_RETURN = 3
# _REGKIND_ITER = 4 # Like a scalar, but expected to change with each run()


# TODO: implement non-default casting, optimization, library, checks
# TODO: do we need name with the object-oriented interface?
# TODO: deprecate out
def evaluate( expr, name=None, lib=LIB_STD, 
             local_dict=None, global_dict=None, out=None,
             order='K', casting=CAST_SAFE, optimization=OPT_MODERATE, 
             library=LIB_STD, checks=CHECK_ALL, stackDepth=1 ):
    """
    Evaluate a mutliline expression element-wise, using the a NumPy.iter

    expr is a string forming an expression, like 
        "c = 2*a + 3*b"
        
    The values for "a" and "b" will by default be taken from the calling 
    function's frame (through use of sys._getframe()). Alternatively, they 
    can be specifed using the 'local_dict' argument.
    
    Multi-line statements, or semi-colon seperated statements, are supported.
    
    
    Parameters
    ----------
    name : DEPRECATED
        use wisdom functions instead.
        
    local_dict : dictionary, optional
        A dictionary that replaces the local operands in current frame.

    global_dict : DEPRECATED
        A dictionary that replaces the global operands in current frame.
        Setting to {} can speed operations if you do not call globals.
        global_dict was deprecated as there is little reason for the 
        user to maintain two dictionaries of arrays.
        
    out : DEPRECATED
        use assignment in expr (i.e. 'out=a*b') instead.

    order : {'C', 'F', 'A', or 'K'}, optional
        Controls the iteration order for operands. 'C' means C order, 'F'
        means Fortran order, 'A' means 'F' order if all the arrays are
        Fortran contiguous, 'C' order otherwise, and 'K' means as close to
        the order the array elements appear in memory as possible.  For
        efficient computations, typically 'K'eep order (the default) is
        desired.

    casting : {CAST_SAFE, CAST_NO, CAST_EQUIV, CAST_SAME_KIND, CAST_UNSAFE}, 
               optional 
               (NumPy string repr also accepted)
        Controls what kind of data casting may occur when making a copy or
        buffering.  Setting this to 'unsafe' is not recommended, as it can
        adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
          
    optimization: {OPT_MODERATE, OPT_AGGRESSIVE}, optional
        Controls what level of optimization the compiler will attempt to 
        perform on the expression to speed its execution.  This optimization
        is performed both by python.compile and numexpr3
        
          * OPT_MODERATE performs simple optimizations, such as minimizing 
            the number of temporary arrays
          * OPT_AGGRESSIVE performs aggressive optimizations, such as replacing 
            powers with mutliplies.
            
    library: {LIB_STD, LIB_VML}, optional
        Indicates which library to use for calculations.  The library must 
        have been linked during the C-extension compilation, such that the 
        associated operations are found in the opTable.
          * LIB_STD is the standard C math.h / cmath.h library
          * LIB_VML is the Intel Vector Math Library
        
        Falls-back to LIB_STD if the other library is not available.  
    """
    if sys.version_info[0] < 3 and not isinstance(expr, (str,unicode) ):
        raise ValueError( "expr must be specified as a string or unicode." )
    elif not isinstance(expr, (str,bytes)):
        raise ValueError( "expr must be specified as a string or bytes." )
        
    if out is not None:
        raise ValueError( "out is deprecated, use an assignment expr such as 'out=a*x+2' instead." )
    if name is not None:
        raise ValueError( "name is deprecated, TODO: replace functionality." )    
    if global_dict is not None:
        raise ValueError( "global_dict is deprecated, please use only local_dict" )   

    casting = _CAST_TRANSLATIONS[casting]
    if casting != CAST_SAFE:
        raise NotImplementedError( "only 'safe' casting has been implemented at present." )  
        
    if lib != LIB_STD:
        raise NotImplementedError( "only 'LIB_STD casting has been implemented at present." )  

    # In order to maintain stackDepth==1 we have to pull the locals and globals
    # from here, rather than the NumExpr constructor.  Otherwise we 
    # generate many troubles in trying to 'guess' the appropriate depth of the
    # calling frame.
    if local_dict is None:
        call_frame = sys._getframe( stackDepth ) 
        local_dict = call_frame.f_locals
        _global_dict = call_frame.f_globals
    else:
        _global_dict = [None]

    # Signature from NE2:
    # numexpr_key = ('a**b', (('optimization', 'aggressive'), ('truediv', False)), 
    #           (('a', <class 'numpy.float64'>), ('b', <class 'numpy.float64'>)))
    # TODO: the signature should also include the dtypes of the arrays.  That
    # cannot change, most other things can.
    # We used to have self.dtypes but how to discover the dtypes without calling the 
    # ast parser?  Maybe this is just an advantage of using the NumExpr object.
    signature = (expr, lib, casting)
    if signature in wisdom:
        return wisdom[signature].run( local_dict=local_dict, stackDepth=stackDepth+1)
    else:
        neObj = NumExpr( expr, lib=lib, casting=casting, stackDepth=stackDepth,
                        local_dict=local_dict, _global_dict=_global_dict )
        # neObj.assemble() called on __init__()
        return neObj.run( check_arrays=False, stackDepth=stackDepth+1 )
    

########################## AST PARSING HANDLERS ################################
# Move the ast parse functions outside of class NumExpr so we can pickle it.
# Pickle cannot deal with bound methods.
# Note: these use 'self', what is passed around is an NumExpr object


def _assign(self, node, outputTup=None  ):
        # node.targets is a list; It must have a len=1 for NumExpr3
        # node.value is likely a BinOp, Call, Comparison, or BoolOp
        if len(node.targets) != 1:
            raise ValueError( 'NumExpr3 supports only singleton returns in assignments.' )
        
        # _messages.append( "Assign node: %s op assigned to %s" % ( node.value, node.targets[0]) )
        # Call function on target and value nodes
        targetTup = _ASTAssembler[type(node.targets[0])](self, node.targets[0])
        valueTup = _ASTAssembler[type(node.value)](self, node.value, targetTup )
        return valueTup
        
        
def _expression(self, node, outputTup=None  ):
    # For expressions we have to return the value instead of promoting it.
    valueTup = _ASTAssembler[type(node.value)](self, node.value)
    self.unallocatedOutput = valueTup
    return valueTup
        
           
def _name(self, node, outputTup=None ):
    #print( 'ast.Name' )
    # node.ctx is probably not something we care for.
    node_id = node.id
    

    if node_id in self.namesReg:
        regTup = self.namesReg[node_id]
        regToken = regTup[0]
    else: # Get address so we can find the dtype
        arr = None
        if node_id in self.local_dict:
            arr = self.local_dict[node_id]
        elif node_id in self._global_dict:
            arr = self._global_dict[node_id]

        if arr is not None:
            regToken = pack( _PACK_REG, next( self._regCount ) )
            # Force lists, ints, floats, and native arrays to be numpy.ndarrays
            if type(arr) != np.ndarray:
                arr = np.asarray( arr )
            # print( "Name found in locals/globals: " + node_id )
            # Build tuple and add to the namesReg
            self.namesReg[node_id] = regTup = (regToken, arr, arr.dtype.char, int(np.isscalar(arr)), node_id )
        else:
            # print( "Name not found: " + node_id )
            # It's probably supposed to be a temporary with a name, i.e. an assignment target
            regTup = self._newTemp( None, name = node_id )
            
    if outputTup is not None:
        # This is a copy operation in an assignment.
        return self._copy(regTup, outputTup)
        pass
        
    
    return regTup

def _copy(self, regTup, outputTup ):
    print( "DEBUG: outputTup = " + str(outputTup) )
    opCode, self.retChar = OPTABLE[ ('copy', self.lib,
                            regTup[2]) ]
    outputTup = self._magic_output( outputTup )
        
    self._codeStream.write( b"".join( (opCode, outputTup[0], 
                            regTup[0], _NULL_REG, _NULL_REG)  )  )
    pass
    
def _const(self, node ):
    constNo = next( self._regCount )
    regKey = pack( _PACK_REG, constNo )
    token = '${}'.format(constNo)
    
    # It's easier to just use ndim==0 numpy arrays, since PyArrayScalar 
    # isn't in the API anymore.
    # Use the _minimum _ dtype available so that we don't accidently upcast.
    # TODO: we should try and force consts to be of the correct dtype in 
    # Python, to avoid extraneous cast operations in the interpreter.

    if np.mod( node.n, 1) == 0: # int
        # Always try and used unsigned ints, but only up to half the 
        # bit-width?
        if node.n < 0:
            if node.n > 128:
                constArr = np.asarray(node.n, dtype='int8' )
            elif node.n > 32768:
                constArr = np.asarray(node.n, dtype='int16' )  
            elif node.n > 32768:
                constArr = np.asarray(node.n, dtype='int16' )  
            elif node.n > 2147483648:
                constArr = np.asarray(node.n, dtype='int32' )
            elif node.n > 2305843009213693952:
                constArr = np.asarray(node.n, dtype='int32' )
            else:
                constArr = np.asarray(node.n, dtype='float64' )
        else: # unsigned
            if node.n < 128:
                constArr = np.asarray(node.n, dtype='uint8' )
            elif node.n < 32768:
                constArr = np.asarray(node.n, dtype='uint16' )  
            elif node.n < 32768:
                constArr = np.asarray(node.n, dtype='uint16' )  
            elif node.n < 2147483648:
                constArr = np.asarray(node.n, dtype='uint32' )
            elif node.n < 2305843009213693952:
                constArr = np.asarray(node.n, dtype='uint32' )
            else:
                constArr = np.asarray(node.n, dtype='float64' )
        
    elif type(node.n) == str or type(node.n) == bytes : 
        constArr = np.asarray(node.n)
    elif np.iscomplex(node.n):
        constArr = np.complex64(node.n)
    else: # float
        constArr = np.asarray(node.n, dtype='float32' )

    self.namesReg[token] = regTup = ( regKey, constArr, 
                    constArr.dtype.char, _REGKIND_SCALAR, node.n.__str__() )
    return regTup
        
def _attribute(self, node ):
    # An attribute has a .value node which is a Name, and .value.id is the 
    # module/class reference. Then .attr is the attribute reference that 
    # we need to resolve.
    # WE CAN ONLY DEREFERENCE ONE LEVEL ('.')
    
    # TODO: .real and .imag need special handling
    
    
    if node.attr == 'real' or node.attr == 'imag':

        return self._real_imag( node, outputTup )
        
    className = node.value.id
    attrName = ''.join( [className, '.', node.attr] )
    
    if attrName in self.namesReg:
        regTup = self.namesReg[attrName]
        regToken = regTup[0]
    else:
        regToken = pack( _PACK_REG, next( self._regCount ) )
        # Get address
        arr = None
        
        # Is there any tricky way to retrieve the local_dict as is rather than
        # forcing the system to make a dict?
        if className in self.local_dict:
            classRef = self.local_dict[className]
            if node.attr in classRef.__dict__:
                arr = self.local_dict[className].__dict__[node.attr]
        # Globals is, as usual, slower than the locals, so we prefer not to 
        # search it.  
        elif className in self._global_dict:
            classRef = self.local_dict[className]
            if node.attr in classRef.__dict__:
                arr =self._global_dict[className].__dict__[node.attr]
        
        if arr is not None and not hasattr(arr,'dtype'):
            # Force lists and native arrays to be numpy.ndarrays
            arr = np.asarray( arr )

        # Build tuple and add to the namesReg
        self.namesReg[attrName] = regTup = \
                        (regToken, arr, arr.dtype.char, int(np.isscalar(arr)), attrName )
    return regTup
    

        
    
def _binop(self, node, outputTup=None ):
    #print( 'ast.Binop' )
    # (left,op,right)
    leftTup = _ASTAssembler[type(node.left)](self, node.left)
    rightTup = _ASTAssembler[type(node.right)](self, node.right)
    
    # Check to see if a cast is required
    leftTup, rightTup = self._cast2( leftTup, rightTup )
        
    # Format: (opCode, lib, left_register, right_register)
    try:
        opWord, self.retChar = OPTABLE[  (type(node.op), self.lib, leftTup[2], rightTup[2] ) ]
    except KeyError as e:
        if leftTup[2] == None or rightTup[2] == None:
            raise ValueError( 
                    'Binop did not find arrays: left: {}, right: {}.  Possibly a stack depth issue'.format(
                            leftTup[4], rightTup[4]) )
        else:
            raise e
    
    # Make/reuse a temporary for output
    outputTup = self._magic_output( outputTup )
        
    #_messages.append( 'BinOp: %s %s %s' %( node.left, type(node.op), node.right ) )
    self._codeStream.write( b"".join( (opWord, outputTup[0], leftTup[0], rightTup[0], _NULL_REG ))  )
    
    # Release the leftTup and rightTup if they are temporaries and weren't reused.
    if leftTup[3] == _REGKIND_TEMP and leftTup[0] != outputTup[0]: 
        self._releaseTemp(leftTup[0])
    if rightTup[3] == _REGKIND_TEMP and rightTup[0] != outputTup[0]: 
        self._releaseTemp(rightTup[0])
    return outputTup
           
def _call(self, node, outputTup=None ):
    # ast.Call has the following fields:
    # ('func', 'args', 'keywords', 'starargs', 'kwargs')
    argTups = [_ASTAssembler[type(arg)](self, arg) for arg in node.args]
    
    # Would be nice to have a prettier way to fill out the program 
    # than if-else block?
    if len(argTups) == 1:
        opCode, self.retChar = OPTABLE[ (node.func.id, self.lib,
                            argTups[0][2]) ]
        outputTup = self._magic_output( outputTup )
        

        self._codeStream.write( b"".join( (opCode, outputTup[0], 
                            argTups[0][0], _NULL_REG, _NULL_REG)  )  )
        
    elif len(argTups) == 2:
        argTups = self._cast2( *argTups )
        opCode, self.retChar = OPTABLE[ (node.func.id, self.lib,
                            argTups[0][2], argTups[1][2]) ]
        outputTup = self._magic_output( outputTup )
                

        self._codeStream.write( b"".join( (opCode, outputTup[0], 
                            argTups[0][0], argTups[1][0], _NULL_REG)  )  )
        
    elif len(argTups) == 3: 
        # The where() ternary operator function is currently the _only_
        # 3 argument function
        argTups[1], argTups[2] = self._cast2( argTups[1], argTups[2] )
        opCode, self.retChar = OPTABLE[ (node.func.id, self.lib,
                            argTups[0][2], argTups[1][2], argTups[2][2]) ]
        outputTup = self._magic_output( outputTup )
                
        self._codeStream.write( b"".join( (opCode, outputTup[0], 
                            argTups[0][0], argTups[1][0], argTups[2][0])  )  )
        
    else:
        raise ValueError( "call(): function calls are 1-3 arguments" )
    
    for arg in argTups:
        if arg[3] == _REGKIND_TEMP: self._releaseTemp( arg[0] )
        
    return outputTup
    
    

        
    
def _compare(self, node, outputTup=None ):
    # print( 'ast.Compare' )
    # "Awkward... this ast.Compare node is," said Yoga disparagingly.  
    # (left,ops,comparators)
    # NumExpr3 does not handle [Is, IsNot, In, NotIn]
    if len(node.ops) > 1:
        raise NotImplementedError( 
                'NumExpr3 only supports binary comparisons (between two elements); try inserting brackets' )
    # Force the node into something the _binop machinery can handle
    node.right = node.comparators[0]
    node.op = node.ops[0]
    return self._binop(node, outputTup)
   
def _boolop(self, node, outputTup=None ):
    # Functionally from the NumExpr perspective there's no difference 
    # between boolean binary operations and binary operations
    if len(node.values) != 2:
        raise ValueError( "NumExpr3 supports binary logical operations only, please seperate operations with ()." )
    node.left = node.values[0]
    node.right = node.values[1]
    self._binop( node, outputTup )
    
def _unaryop(self, node, outputTup=None ):
    # Currently only ast.USub is supported, and the node.operand is the 
    # value acted upon.
    operandTup = _ASTAssembler[type(node.operand)](self, node.operand)
    outputTup = self._magic_output( outputTup )
    
    try:
        opWord, self.retChar = OPTABLE[  (type(node.op), self.lib, operandTup[2] ) ]
    except KeyError as e:
        if operandTup[2] == None :
            raise ValueError( 
                    'Unary did not find operand array {}. Possibly a stack depth issue'.format(
                            operandTup[4]) )
        else:
            raise e
    self._codeStream.write( b"".join( (opWord, outputTup[0], operandTup[0], _NULL_REG, _NULL_REG ))  )  
        
    # Release the operandTup if it was a temporary
    if operandTup[3] == _REGKIND_TEMP and operandTup[0] != outputTup[0]: 
        self._releaseTemp(operandTup[0])
    return outputTup
                
def _unsupported(self, node, outputTuple=None ):
    raise KeyError( 'unimplmented ASTNode' + type(node) )

_ASTAssembler = defaultdict( _unsupported, 
                  { ast.Assign:_assign, ast.Expr:_expression, \
                    ast.Name:_name, ast.Num:_const, \
                    ast.Attribute:_attribute, ast.BinOp:_binop, \
                    ast.BoolOp:_boolop, ast.UnaryOp:_unaryop, \
                    ast.Call:_call, ast.Compare:_compare, \
                     } )
######################### END OF AST HANDLERS ##################################

class NumExpr(object):
    """
    
    The self.program attribute is a `bytes` object and consists of operations 
    followed by registers with the form,
    
        opcode + return_reg + arg1_reg + arg2_reg + arg3_reg
        
    Currently opcode is a uint16 and the register numbers are uint8.  However,
    note that numpy.iter limits the maximum number of arguments to 32.
    
    Register tuples have the format: 
        
        (regCode, object, dchar, regKind, name)
        
    where
        regCode: a single byte indicating the register number
        object:   the actual object reference, is `None` for temporaries.
        dchar:    the `object.dtype.char`, or for temporaries the expected value
        regKind:  indicates ndarray, scalar, temporary
        name:     the original name of the array.  It is not required for the 
                  input arguments to run have the same names, just that they
                  be in the same order and dtypes.
    """
    # The maximum arguments is 32 due to NumPy, we have space for 255
    # One can recompile NumPy after changing NPY_MAXARGS and use the full 
    # argument space.
    MAX_ARGS = 255
    
    def __init__(self, expr, lib=LIB_STD, casting=CAST_SAFE, local_dict=None, 
                 _global_dict=None, stackDepth=1 ):
        
        # Public
        self.expr = expr
        self.program = b''
        self.lib = lib
        self.casting = casting
        self.outputTarget = None
        self.namesReg = OrderedDict()
        self.unallocatedOutput = False  # sentinel
        self.retChar = ''
        
        # Protected
        self._regCount = iter(range(NumExpr.MAX_ARGS))
        self._stackDepth = stackDepth # How many frames 'up' to promote outputs
        
        self._codeStream = BytesIO()
        self._occupiedTemporaries = set()
        self._freeTemporaries = set()
        self._messages = []           # For debugging
        self._compiled_exec = None    # Handle to the C-api NumExprObject
        self._lastOp = False          # sentinel
        self._broadCast = None

        # Get references to frames
        call_frame = sys._getframe( self._stackDepth ) 
        if local_dict is None:
            self.local_dict = call_frame.f_locals
            self._global_dict = call_frame.f_globals
        else:
            self.local_dict = local_dict
            self._global_dict = _global_dict
            
        # self._ASTAssembler = defaultdict( self._unsupported, 
        #           { ast.Assign:self._assign, ast.Expr:self._expression, \
        #             ast.Name:self._name, ast.Num:self._const, \
        #             ast.Attribute:self._attribute, ast.BinOp:self._binop, \
        #             ast.BoolOp:self._boolop, ast.UnaryOp:self._unaryop, \
        #             ast.Call:self._call, ast.Compare:self._compare, \
        #              } )
    
        self.assemble()

    def __getstate__(self):
        '''
        Preserves NumExpr object via `pickledBytes = pickle.dumps(neObj)`

        For pickling, we have to remove the local_dict and _global_dict 
        attributes as they aren't pickleable.
        '''
        pickleDict = self.__dict__.copy()
        # Remove non-needed and non-pickelable attributes
        pickleDict['local_dict'] = None
        pickleDict['_global_dict'] = None
        pickleDict['_codeStream'] = b''
        #pickleDict['_compiled_exec'] = self._compiled_exec.__getstate__()

        # Strip out the array references in namesReg so we don't pickle the 
        # entire array set
        namesReg = pickleDict['namesReg']
        for key, value in namesReg.items():
            namesReg[key] = (value[0], None, *value[2:] )

        # print( 'pickleDict: {}'.format(pickleDict) )
        return pickleDict

    def __setstate__(self, state):
        '''
        Restores NumExpr object via `neObj = pickle.loads(pickledBytes)`
        '''
        self.__dict__ = state
        # print( 'setstate.state: {}'.format(state) )
        call_frame = sys._getframe( state['_stackDepth'] )
        self.local_dict = call_frame.f_locals
        self._global_dict = call_frame.f_globals
       

    def assemble(self):
        ''' 
        NumExpr.assemble() can be used in the context of having a pool of 
        NumExpr objects; it is always called by __init__().
        '''
        global wisdom
        # Here we assume the local/global_dicts have been populated with 
        # __init__.  
        # Otherwise we have issues with stackDepth being different depending 
        # on where the method is called from.
        
        forest = ast.parse( self.expr ) 
        N_forest_m1 = len(forest.body) - 1
                         
        for I, bodyItem in enumerate( forest.body ):
            bodyType = type(bodyItem)
            if I == N_forest_m1:
                # print( "Setting lastOp sentinel" )
                self._lastOp = True
                # Add the last assignment name as the outputTarget so it can be 
                # promoted up the frame later
                if bodyType == ast.Assign:
                    self.outputTarget = bodyItem.targets[0].id
                # Do we need something else for ast.Expr?
                                        
            if bodyType == ast.Assign:
                _ASTAssembler[bodyType](self, bodyItem)
            elif bodyType == ast.Expr:
                # Probably the easiest thing to do is generate magic output 
                # as with Assign but return rather than promote it.
                if not self._lastOp:
                    raise SyntaxError( "Expressions may only be single statements." )
                    
                # Just an unallocated output case
                _ASTAssembler[ast.Expr](self, bodyItem)
                
            else:
                raise NotImplementedError( 'Unknown ast body: {}'.format(bodyType) )
                                  
        if self.unallocatedOutput:
            # Discover the expected dtype and shape of the output
            self.namesReg.pop(self.unallocatedOutput[4])
            self.namesReg[self.outputTarget] = self.unallocatedOutput = \
                         (self.unallocatedOutput[0], None, self.retChar, 
                          _REGKIND_RETURN, self.outputTarget )
                                            
        # The OrderedDict for namesReg isn't needed if we sort the tuples, 
        # which we need to do such that the magic output isn't out-of-order.
        regsToInterpreter = tuple( sorted(self.namesReg.values() ) )
        
        # Collate the inputNames as well as the the required outputs
        self.program = self._codeStream.getvalue()
        
        # Add self to the wisdom
        wisdom[(self.expr, self.lib, self.casting)] = self
        # Maybe we should have a set as well?
        # print( "assemble: regsToInterpreter: " + str(regsToInterpreter))
        # self._compiled_exec = interpreter.NumExpr( program=self.program, 
        #                                      registers=regsToInterpreter )
        self._compiled_exec = interpreter.CompiledExec( program=self.program, 
                                             registers=regsToInterpreter )
        
    
    def disassemble( self ):
        global _PACK_REG, _PACK_OP, _NULL_REG
        
        blockLen = calcsize(_PACK_OP) + 4*calcsize(_PACK_REG)
        if len(self.program) % blockLen != 0:
            raise ValueError( 
                    'disassemble: len(progBytes)={} is not divisible by {}'.format(len(self.program,blockLen)) )
        # Reverse the opTable
        reverseOps = {op[0] : key for key, op in OPTABLE.items()}
        progBlocks = [self.program[I:I+blockLen] for I in range(0,len(self.program), blockLen)]
        
        print( "=======================================================" )
        print( "REGISTERS: " )
        TYPENAME = { 0:'reg', 1:'scalar', 2:'temp', 3:'return' }
        for reg in sorted(self.namesReg.values()):
            print( "{} : name {:>12} : dtype {:1} : type {:5}".format(ord(reg[0]), reg[4], reg[2], TYPENAME[reg[3]] ) )
            
            
        print( "DISASSEMBLED PROGRAM: " )
        for J, block in enumerate(progBlocks):
            opCode, ret, arg1, arg2, arg3 = unpack( _UNPACK, block )
            if arg3 == ord(_NULL_REG): arg3 = '-'
            if arg2 == ord(_NULL_REG): arg2 = '-'
            if arg1 == ord(_NULL_REG): arg1 = '-'
            
            opTuple = reverseOps[ pack(_PACK_OP, opCode) ]
    
            # For the ast.Nodes we want a 'pretty' name in the output
            # give 
            if hasattr( opTuple[0], '__name__' ):
                opString = opTuple[0].__name__ + "_" + "".join( [str(dchar) for dchar in opTuple[2:] ] ).lower()
            else:
                opString = str(opTuple[0]) + "_" + "".join( [str(dchar) for dchar in opTuple[2:] ] )
            print( '#{:2}, op: {:>12} in ret:{:3} <- args({:>3}::{:>3}::{:>3})'.format(
                    J, opString, ret, arg1, arg2, arg3 ) )
        print( "=======================================================" )
        
    
    def print_names(self):
        for val in self.namesReg.values(): print( '{}:{}'.format( val[4], val[3])  )
        
    def __call__(self, **kwargs):
        if not 'stackDepth' in kwargs:
            kwargs['stackDepth'] = self._stackDepth + 1
        else: 
            pass
        self.run( **kwargs)
        
    def run(self, stackDepth=None, local_dict=None, check_arrays=True, **kwargs):
        '''
        `run()` is called with keyword arguments as the order of 
        args is based on the Abstract Syntax Tree parse and may be 
        non-intuitive.
        
            e.g. self.run( a=a1, b=b1, out=out_new )
        
        where {a,b,out} were the original names in the expression. 
        
        Additional keyword arguments are:
            
            stackDepth {None}: Generally not needed, tells the function how 
              many stacks up it was called from.
            
            local_dict {None}: A optional dict containing all of the arrays 
              required for calculation.  Saves the Python interpreter some time 
              in looking them up from the calling namespace. 
              Note that this is somewhat superfluous as its functionality is 
              mimiced by kwargs.  
        
            check_arrays {True}: Resamples the calling frame to grab arrays. 
              There is some overhead associated with grabbing the frames so 
              if inside a loop and using run on the same arrays repeatedly 
              then try `False`. 

        '''
        # Not supporting Python 2.7 anymore, so we can mix named keywords and kw_args
        if not stackDepth:
            stackDepth = self._stackDepth

        if kwargs:
            # Match kwargs to self.namesReg[4]
            args = [kwargs[reg[4]] for reg in self.namesReg.values() if reg[3] == _REGKIND_ARRAY]

        elif local_dict: # Use local_dict, may be deprecated eventually.
            args = [local_dict[reg[4]] for reg in self.namesReg.values() if reg[3] == _REGKIND_ARRAY]

        elif check_arrays: # Renew references to frames
            call_frame = sys._getframe( stackDepth ) 
            self.local_dict = call_frame.f_locals
            self._global_dict = call_frame.f_globals
                
            args = [self.local_dict[reg[4]] for reg in self.namesReg.values() if reg[3] == _REGKIND_ARRAY]

        else: # Re-use existing arrays
            args = [reg[1] for reg in self.namesReg.values() if reg[3] == _REGKIND_ARRAY]
            if len(args) == 0:
                raise ValueError( "No input arguments found, perhaps you intended to set check_arrays=False?" )
        
        # Check if the result needs to be added to the calling frame's locals
        promoteResult = (not self.outputTarget in kwargs) \
                        and ( (not self.outputTarget in self.local_dict) and 
                              (not self.outputTarget in self._global_dict) )
        
        #print( "args2: " + str(args) )
        if bool(self.unallocatedOutput):
            # Can't *assign in Python 2.7
            # op, retN, *argNs = unpack( _UNPACK, self.program[-6:] )
            op, retN, arg1, arg2, arg3 = unpack( _UNPACK, self.program[-6:] )
            #print( "Broadcast for op: {}, ret: {}, a1: {}, a2: {}, a3: {}".format(
            #        op, retN, argNs[0], argNs[1], argNs[2] ) )
            argNs = [N for N in (arg1,arg2,arg3) if N != ord(_NULL_REG)]
            
            if retN != ord(self.namesReg[self.outputTarget][0]):
                raise ValueError( 'Last program set destination to other than output' )
                
            # Is there a better way than to iterate through all the registers
            # such as inside the AST parse with _lastOp?
            arraysOrdered = [reg[1] for reg in sorted( self.namesReg.values() )]
            # So we have broadcast problem for something like '2.0 * a + 3.0 * b * c'
            guessedBroadcast = [arraysOrdered[N] for N in argNs if arraysOrdered[N] is not None]
            if not guessedBroadcast:
                raise ValueError( "Broadcast guess failed" )
            self._broadCast = np.broadcast( *guessedBroadcast )
            unalloc = np.empty( self._broadCast.shape, dtype=self.retChar )
            
            # print( 'Unallocated output: {} broadcast shape: {}, dtype: {}'.format(self.outputTarget, unalloc.shape, unalloc.dtype) )
            
            # Sometimes the return is in the middle of the args because 
            # the set.pop() from _occupiedTemps is not deterministic.
            # TODO: re-write this args insertion mess, it can go inside 
            # the if len(args)==0 loop above.
            arrayCnt = 0
            outId = ord(self.unallocatedOutput[0])
            for reg in self.namesReg.values():
                regId = ord(reg[0])
                if regId >= outId:
                    args = args[:arrayCnt] + [unalloc] + args[arrayCnt:]
                    break
                if reg[3] == _REGKIND_ARRAY:
                    arrayCnt += 1

        else:
            unalloc = None

        
        # for I, arg in enumerate(args):
        #     print( "DB#{}: {}".format(I,arg) )
        with _NE_RUN_LOCK:
            self._compiled_exec( *args, **kwargs )
        
        if promoteResult and self.outputTarget is not None:
            # Insert result into calling frame
            if local_dict is None:
                sys._getframe( stackDepth ).f_locals[self.outputTarget] = unalloc
                return
            self.local_dict[self.outputTarget] = unalloc
            self.unallocatedOutput = False
            return 
                           
        if self.outputTarget is None:
            return unalloc
        return # end NumExpr.run()


    def _newTemp(self, dchar, name = None ):
        if len(self._freeTemporaries) > 0:
            regId = self._freeTemporaries.pop()
            #print( 'Re-using temporary: %s' % regId )
            tempTup = ( regId, None, dchar, _REGKIND_TEMP, '${}'.format(ord(regId)) )
        else:
            tempNo = next( self._regCount )
            
            #print( 'new Temp: {} assigned to reg#{}'.format(name, tempNo) )
            regId = pack( _PACK_REG, tempNo )
            if name == None:
                name = '${}'.format(tempNo)
                
            self.namesReg[name] = tempTup = ( 
                regId, None, dchar, _REGKIND_TEMP, name )
        self._occupiedTemporaries.add( regId )
        return tempTup
            

    def _releaseTemp(self, regId ):
        # Free a temporary
        #print( "Releasing temporary: %s" % regId )
        self._occupiedTemporaries.remove( regId )
        self._freeTemporaries.add( regId )


    def _real_imag( self, outputTup ):
            # Should they always be calls to real() and imag() or should we 
            # get the strided array?
            # What if the user does (x+y).real ?
            print( "TODO: handle .real and .imag" )
            pass
        

    def _magic_output( self, outputTup ):
        # 
        if outputTup is None:
            outputTup = self._newTemp( self.retChar )
        elif outputTup[2] == None: 
            self.namesReg[outputTup[4]] = outputTup = \
                                    (outputTup[0], outputTup[1], self.retChar, outputTup[3], outputTup[4] )
            if self._lastOp:
                self.unallocatedOutput = outputTup
        return outputTup


    def _cast2(self, leftTup, rightTup ): 
        leftD = leftTup[2]; rightD = rightTup[2]
        # print( "cast2: %s, %s"%(leftD,rightD) ) 
        if leftD == rightD:
            return leftTup, rightTup
        elif np.can_cast( leftD, rightD ):
            # Make a new temporary
            castTup = self._newTemp( rightD )
            
            self._codeStream.write( b"".join( 
                    (OPTABLE[('cast',self.casting,rightD,leftD)][0], castTup[0], 
                                leftTup[0], _NULL_REG, _NULL_REG)  ) )
            return castTup, rightTup
        elif np.can_cast( rightD, leftD ):
            # Make a new temporary
            castTup = self._newTemp( leftD )
                        
            self._codeStream.write( b"".join( 
                    (OPTABLE[('cast',self.casting,leftD,rightD)][0], castTup[0], 
                                rightTup[0], _NULL_REG, _NULL_REG) ) )
            return leftTup, castTup
        else:
            raise TypeError( "cast2(): Cannot cast %s to %s by rule 'safe'"
                            %(np.dtype(leftD), np.dtype(rightD) ) ) 
            
    
    def _cast3(self, leftTup, midTup, rightTup ):
        # _cast3 isn't called by where/tenary so no need for an implementation
        # at present.
        self._messages.append( 'TODO: implement 3-argument casting' )
        return leftTup, midTup, rightTup

# The wisdomBank connects strings to their NumExpr objects, so if the same 
# expression pattern is called, it will be retrieved from the bank.
# Also this permits serialization via pickle.
class _WisdomBankSingleton(dict):
    
    def __init__(self, wisdomFile="", maxEntries=256 ):
        # Call super
        super(_WisdomBankSingleton, self).__init__( self )
        # attribute dictionary breaks a lot of things in the intepreter
        # dict.__init__(self)
        self.__wisdomFile = wisdomFile
        self.maxEntries = maxEntries
        pass
    
    @property 
    def wisdomFile(self):
        if not bool(self.__wisdomFile):
            if not os.access( 'ne3_wisdom.pkl', os.W_OK ):
                raise OSError( 'insufficient permissions to write to {}'.format('ne3_wisdom.pkl') )
            self.__wisdomFile = 'ne3_wisdom.pkl'
        return self.__wisdomFile
    
    @wisdomFile.setter
    def wisdomFile(self, newName):
        '''Check to see if the user has write permisions on the file.'''
        dirName = os.path.dirname(newName)
        if not os.access( dirName, os.W_OK ):
            raise OSError('do not have write perimission for directory {}'.format(dirName))
        self.__wisdomFile = newName
    
    def __setitem__(self, key, value):
        # Protection against growing the cache too much
        if len(self) > self.maxEntries:
            # Remove a 10% of random elements from the cache
            entries_to_remove = self.maxEntries // 10
            # This code doesn't work in Python 3.
            keysView = list(self.keys())
            for I, cull in enumerate(keysView):
                # super(_WisdomBankSingleton, self).__delitem__(cull)
                self.pop(cull)
                if I >= entries_to_remove: 
                    break
                
        #self.__dict__[key] = value
        super(_WisdomBankSingleton, self).__setitem__(key, value)
         
    # Pickling support for wisdom:
    # Pickling still needs some work.  Possibly the WisdomBankSingleton also needs 
    # __getstate__ and __setstate__ magic functions.
    # def load( self, wisdomFile=None ):
    #     if wisdomFile == None:
    #         wisdomFile = self.wisdomFile
            
    #     with open( wisdomFile, 'rb' ) as fh:
    #         self = pickle.load(fh)

    # def dump( self, wisdomFile=None ):
    #     if wisdomFile == None:
    #         wisdomFile = self.wisdomFile
  
    #     with open( wisdomFile, 'wb' ) as fh:
    #         pickle.dump(self, fh)

wisdom = _WisdomBankSingleton()