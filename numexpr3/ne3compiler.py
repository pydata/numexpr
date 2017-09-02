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
from collections import defaultdict, deque
import weakref

# The operations are saved on disk as a pickled dict
try: import cPickle as pickle
except ImportError: import pickle
# For appending long strings, using a buffer is faster than ''.join()
try: from cStringIO import StringIO as BytesIO
except ImportError: from io import BytesIO
# struct.pack is the quickest way to build the program as structs
# All important format characters: https://docs.python.org/2/library/struct.html
from struct import pack, unpack, calcsize


# DEBUG:
try:
    from colorama import init, Fore, Back, Style; init()
    def info( text ):
        print( ''.join( [Fore.GREEN, text, Style.RESET_ALL] ) )  
    def warn( text ):
        print( ''.join( [Fore.YELLOW, text, Style.RESET_ALL] ) )   
    def debug( text ):
        print( ''.join( [Fore.RED, text, Style.RESET_ALL] ) )
except ImportError:
    info = print; warn = print; debug = print


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

_PACK_OP =  b'H'
_PACK_REG = b'B'
_NULL_REG =  pack( _PACK_REG, 255 )
_UNPACK = b''.join([_PACK_OP,_PACK_REG,_PACK_REG,_PACK_REG,_PACK_REG])
_RET_LOC = -4

# Map np.dtype.char to np.dtype.itemsize
_DCHAR_ITEMSIZE = {dchar:np.dtype(dchar).itemsize for dchar in np.typecodes['All']}
#    gives 0s for strings and unicode.
_DCHAR_ITEMSIZE['S'] = 1
_DCHAR_ITEMSIZE['U'] = 4
# Also we need a default value for None
_DCHAR_ITEMSIZE[None] = 0


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
LIB_STD = 0     # C++ cmath standard library
#LIB_VML = 1    # Intel Vector Math library
#LIB_YEPPP = 2  # Yeppp! math library

CHECK_ALL = 0  # Normal operation in checking dtypes
#CHECK_NONE = 1 # Disable all checks for dtypes if expr is in cache

# Note: int(np.isscalar( np.float32(0.0) )) is used to set KIND, 
# so ARRAY == 0 and SCALAR == 1
# TODO: could use bitmasks and int(1+np.isscalar( np.float32(0.0) ))
_KIND_ARRAY = 0
_KIND_SCALAR = 1
_KIND_TEMP = 2
_KIND_RETURN = 3


# TODO: implement non-default casting, optimization, library, checks
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
        A dictionary that replaces the local operands in current frame. This is 
        generally required in Cython, as Cython does not populate the calling 
        frame variables according to Python standards.

    global_dict : DEPRECATED
        A dictionary that replaces the global operands in current frame.
        Setting to {} can speed operations if you do not call globals.
        global_dict was deprecated as there is little reason for the 
        user to maintain two dictionaries of arrays.
        
    out : DEPRECATED
        use assignment in expr (i.e. 'out=a*b') instead.

    order : {'C', 'F', 'A', or 'K'}, optional
        Currently only 'K' is supported in NumExpr3.

        Controls the iteration order for operands. 'C' means C order, 'F'
        means Fortran order, 'A' means 'F' order if all the arrays are
        Fortran contiguous, 'C' order otherwise, and 'K' means as close to
        the order the array elements appear in memory as possible.  For
        efficient computations, typically 'K'eep order (the default) is
        desired.

    casting : {CAST_SAFE, CAST_NO, CAST_EQUIV, CAST_SAME_KIND, CAST_UNSAFE}, 
               optional 
               (NumPy string repr also accepted)

        Currently only 'safe' is supported in NumExpr3.

        Controls what kind of data casting may occur when making a copy or
        buffering.  Setting this to 'unsafe' is not recommended, as it can
        adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

    optimization: {OPT_MODERATE}, optional
        Controls what level of optimization the compiler will attempt to 
        perform on the expression to speed its execution.  This optimization
        is performed both by python.compile and numexpr3
        
          * OPT_MODERATE performs simple optimizations, such as minimizing 
            the number of temporary arrays
            
    library: {LIB_STD}, optional
        Indicates which library to use for calculations.  The library must 
        have been linked during the C-extension compilation, such that the 
        associated operations are found in the opTable.
          * LIB_STD is the standard C math.h / cmath.h library
        
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


    # Signature is reduced compared to NumExpr2, in that we don't discover the 
    # dtypes.  That is left for the .run() method, where in verify it does 
    # check the input dtypes and emits a TypeError if they don't match.
    signature = (expr, lib, casting)
    if signature in wisdom:
        try:
            return wisdom[signature].run( verify=True, stackDepth=stackDepth+1 )
        except TypeError as e:
            # If we get a TypeError one of the inputs dtypes is wrong, so we 
            # need to assemble a new NumExpr object
            pass

    neObj = NumExpr( expr, lib=lib, casting=casting, stackDepth=stackDepth+1 )
    return neObj.run()
    # End of ne3.evaluate()


########################## AST PARSING HANDLERS ################################
# Move the ast parse functions outside of class NumExpr so we can pickle it.
# Pickle cannot deal with bound methods.
# Note: these use 'self', which must be a NumExpr object 
# ('self' is not a reserved keyword in Python)
def _assign(self, node):
    '''
    This is for the assignment inside a code block.  The last assignment with 
    magic output is _last_assign.
    '''
    print( '$ast.Assign' )
    # node.targets is a list; It must have a len=1 for NumExpr3 (i.e. no multiple returns)
    # node.value is likely a BinOp, Call, Comparison, or BoolOp
    if len(node.targets) != 1:
        raise ValueError( 'NumExpr3 supports only singleton returns in assignments.' )
    
    valueReg = _ASTAssembler[type(node.value)](self, node.value)
    # Not populating self.assignTarget here, it's only needed for id'ing the 
    # output.
    return _mutate(self, node.targets[0], valueReg)

def _assign_last(self, node):
    info( '$ast.Assign, flow: LAST' )
    
    if len(node.targets) != 1:
        raise ValueError( 'NumExpr3 supports only singleton returns in assignments.' )
    
    valueReg = _ASTAssembler[type(node.value)](self, node.value)
    self.assignTarget = _mutate_last(self, node.targets[0], valueReg)
    return self.assignTarget

def _expression(self, node):
    raise SyntaxError( "NumExpr3 expressions can only be the last line in a statement." )        

def _expression_last(self, node):
    '''
    The statement block can end without an assignment, in which case the output 
    is implicitely allocated and returned.
    '''
    info(  '$ast.Expression, flow: LAST' )
    
    raise NotImplementedError( "We will need to duplicate all the _mutate_last logic?" )

    # DEPRATED:::::::
    valueReg = _ASTAssembler[type(node.value)](self, node.value)
    outToken = next( self._regCount )
    assignTarget = NumReg( outToken, outToken, None, self.assignDChar, _KIND_RETURN )
    self.registers[outToken] =  assignTarget
    self.assignTarget = _mutate_last(self, assignTarget, valueReg)
    return self.assignTarget
        
def _mutate(self, targetNode, valueReg):
    '''
    Used for intermediate assignment targets.  This takes the valueReg and 
    mutates targetReg into it.

    Cases:
    1.) targetReg is KIND_ARRAY or KIND_SCALAR in which case it has been
        pre-allocated and is a secondary return.
    2.) targetReg is KIND_TEMP in which case it's a _named_ temporary.

    In some cases this will require a new temporary.  For example, if the output 
    dtype is smaller than the valueReg.dchar it can't be mutated to a named 
    output.
    '''
    print( "TODO: stop targetReg from being formed if it's not needed" )

    if isinstance( targetNode, ast.Name ):
        if valueReg.kind == _KIND_ARRAY or valueReg == _KIND_SCALAR:
            # This is a copy, like NumExpr( 'y=x' )
            targetReg = _ASTAssembler[type(node.targets[0])](self, node.targets[0])
            return self._copy(targetReg, valueReg)

        nodeId = targetNode.id
        if nodeId in self.registers:
            # Pre-existing, pre-known output.
            # Often in-line operation, e.g. NumExpr('b=2*b')
            # Should we count how many times each temporary is used?
            # As this is a case where if the temp is used twice we can't 
            # get rid of it, but if it's used once we can pop it.
            targetReg = self.registers[nodeId]
            # Intermediat assignment targets keep their KIND
            program = self._codeStream.getbuffer()
            oldToken = program[_RET_LOC]
            program[_RET_LOC] = targetReg._num
            return targetReg
        
        if valueReg.itemsize <= _DCHAR_ITEMSIZE[self.assignDChar]:
            # Can mutate as the valueReg temporary's itemsize is less-than-equal to 
            # the output array's.
            self.registers.pop(valueReg.name)
            valueReg.name = nodeId
            if nodeId in self.local_dict: # Pre-allocated array found
                nodeRef = self.local_dict[nodeId]
                if type(nodeRef) != np.ndarray: nodeRef = np.asarray( nodeRef )
                valueReg.ref = nodeRef if np.isscalar(nodeRef) else weakref.ref(nodeRef)

            valueReg.kind = _KIND_RETURN
            self.assignTarget = self.registers[nodeId] = valueReg
            return valueReg

        else:
            # Else mutate valueRegister into assignTarget
            # TODO: rewind
            raise NotImplementedError( "TODO: rewind program")
        
    elif isinstance( targetNode, ast.Attribute ):
        raise NotImplementedError( "TODO: assign to attributes ")
    else:
        raise SyntaxError( "Illegal NumExpr assignment target: {}".format(targetNode) )
    pass


def _mutate_last(self, targetNode, valueReg):
    '''
    Used for logic control for final assignment targets.  Assignment targets can be 
    an ast.Name or ast.Attribute 

    There's a few-ish cases:
     1.) valueRegister is a temp, in which case see if it can mutate to a KIND_RETURN
     2.) targetNode previously named, in which case program must be rewound
     3.) valueRegister is an array or scalar, in which case we need to do a copy
     4.) valueRegister is an attribute, which is the same as name but needs one level of indirection
    
    '''
    if isinstance( targetNode, ast.Name ):
        if valueReg.kind == _KIND_ARRAY or valueReg == _KIND_SCALAR:
            # This is a copy, like NumExpr( 'y=x' )
            targetReg = _ASTAssembler[type(node.targets[0])](self, node.targets[0])
            return self._copy(targetReg, valueReg)

        nodeId = targetNode.id
        if nodeId in self.registers:
            # Pre-existing, pre-known output.
            # Often in-line operation, e.g. NumExpr('b=2*b')
            warn( "WARNING: IN-LINE CREATES AN EXTRA TEMP" )
            # Should we count how many times each temporary is used?
            # As this is a case where if the temp is used twice we can't 
            # get rid of it, but if it's used once we can pop it.
            targetReg = self.registers[nodeId]
            targetReg.kind = _KIND_RETURN
            program = self._codeStream.getbuffer()
            oldToken = program[_RET_LOC]
            program[_RET_LOC] = targetReg._num
            return targetReg
        
        
        if valueReg.itemsize <= _DCHAR_ITEMSIZE[self.assignDChar]:
            # Can mutate as the temporary's itemsize is less-than-equal to 
            # the output array's.
            self.registers.pop(valueReg.name)
            valueReg.name = nodeId
            if nodeId in self.local_dict: # Pre-allocated array found
                nodeRef = self.local_dict[nodeId]
                if type(nodeRef) != np.ndarray: nodeRef = np.asarray( nodeRef )
                valueReg.ref = nodeRef if np.isscalar(nodeRef) else weakref.ref(nodeRef)

            valueReg.kind = _KIND_RETURN
            self.assignTarget = self.registers[nodeId] = valueReg
            return valueReg

        else:
            # Else mutate valueRegister into assignTarget
            # TODO: rewind
            raise NotImplementedError( "TODO: rewind program")

    elif isinstance( targetNode, ast.Attribute ):
        raise NotImplementedError( "TODO: assign_last to attributes ")
    else:
        raise SyntaxError( "Illegal NumExpr assignment target: {}".format(targetNode) )
    pass



           
def _name(self, node):
    '''
    _name(self, node)

    This is a factory method for building new NumReg objects from names parsed
    via the AST.

    Handles three cases:  
        1. node.id is already in namesReg, in which case re-use it
        2. node.id exists in the calling frame, in which case it's KIND_ARRAY, 
        3. it doesn't, in which case it's a named temporary.

    KIND_RETURN for magic_output handled seperately.
    '''
    # node.ctx (context) is not something that needs to be tracked

    nodeId = node.id
    if nodeId in self.registers:
        info( 'ast.Name: found {} in namesReg'.format(nodeId) )
        return self.registers[nodeId]

    else: # Get address so we can find the dtype
        nodeRef = self.local_dict[nodeId] if nodeId in self.local_dict else None
        # Should we get rid of _global_dict?  It's slowing us down.  Just 
        # require people to define `global` if they want to use it.
        # elif nodeId in self._global_dict: 
        #     nodeRef = self._global_dict[nodeId]

        if nodeRef is None:
            info( 'ast.Name: named temporary {}'.format(nodeId) )
            # It's a named temporary.  
            # Named temporaries can re-use an existing temporary but they cannot be
            # re-used except explicitely by the user!
            # TODO: this could also be a return array, check self.assignTarget?
            return self._newTemp( None, nodeId )
            
        else:
            # It's an existing array we haven't seen before
            info( 'ast.Name: new existing array {}'.format(nodeId) )
            
            regToken = next( self._regCount )
            # Force lists, ints, floats, and native arrays to be numpy.ndarrays
            # so they have a dtype
            if type(nodeRef) != np.ndarray:
                nodeRef = np.asarray( nodeRef )

            # Build NumReg object and add to the registers
            if np.isscalar(nodeRef):
                self.registers[nodeId] = register = NumReg(regToken, nodeId, nodeRef, nodeRef.dtype.char, _KIND_SCALAR )
            else:
                self.registers[nodeId] = register = NumReg(regToken, nodeId, weakref.ref(nodeRef), nodeRef.dtype.char, _KIND_ARRAY )
            return register
    
def _const(self, node):
    constNo = next( self._regCount )
    # regKey = pack( _PACK_REG, constNo )
    # token = '$%d'%constNo
    
    # It's easier to just use ndim==0 numpy arrays, since PyArrayScalar 
    # isn't in the API anymore.
    # Use the _minimum _ dtype available so that we don't accidently upcast.

    # TODO: we should try and force consts to be of the correct dtype in 
    # Python, to avoid extraneous cast operations in the interpreter.
    # This should be moved into _cast2
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
        
    elif type(node.n) == str or type(node.n) == bytes: 
        constArr = np.asarray(node.n)
    elif np.iscomplex(node.n):
        constArr = np.complex64(node.n)
    else: # float
        constArr = np.asarray(node.n, dtype='float32' )

    # Const arrays shouldn't be weak references as they are part of the 
    # program and unmutable.
    self.registers[constNo] = register = NumReg( constNo, constNo, constArr, 
                    constArr.dtype.char, _KIND_SCALAR )
    return register
        

def _attribute(self, node):
    # An attribute has a .value node which is a Name, and .value.id is the 
    # module/class reference. Then .attr is the attribute reference that 
    # we need to resolve.
    # WE CAN ONLY DEREFERENCE ONE LEVEL ('.').  To go deeper we need some 
    # recursive solution.
    
    # .real and .imag need special handling because they could be sliced views
    if node.attr == 'imag' or node.attr == 'real':
        return _real_imag( self, node)

    className = node.value.id
    attrName = ''.join( [className, '.', node.attr] )
    
    if attrName in self.registers:
        register = self.registers[attrName]
        regToken = register.token
    else:
        regToken =  next( self._regCount )
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
        # elif className in self._global_dict:
        #     classRef = self.local_dict[className]
        #     if node.attr in classRef.__dict__:
        #        arr = self._global_dict[className].__dict__[node.attr]
        
        if arr is not None and not hasattr(arr,'dtype'):
            # Force lists and native arrays to be numpy.ndarrays
            arr = np.asarray( arr )

        # Build tuple and add to the namesReg
        self.registers[attrName] = register = NumReg( regToken, attrName, weakref.ref(arr), arr.dtype.char, int(np.isscalar(arr)) )
    return register
    
def _real_imag(self, node):
    viewName = node.attr
    # Having a seperate path for slicing existing arrays was considered but 
    # it is overly difficult to manage the weak reference.
    info( "Functionize ." + str(viewName)  )

    register = _ASTAssembler[type(node.value)](self, node.value)

    if isinstance( node.value, ast.Name ):
        opSig = (viewName, self.lib, register.dchar)
    else:
        opSig = (viewName, self.lib, self.assignDChar)
    opCode, self.assignDChar = OPTABLE[ opSig ]

    # Make/reuse a temporary for output
    outputRegister = self._newTemp( self.assignDChar, None )

    self._codeStream.write( b"".join( (opCode, outputRegister.token, 
                        register.token, _NULL_REG, _NULL_REG)  )  )

    self._releaseTemp(register, outputRegister)
    return outputRegister

def _binop(self, node):
    info( '$ast.Binop: %s'%node.op )
    # (left,op,right)
    leftRegister = _ASTAssembler[type(node.left)](self, node.left)
    rightRegister = _ASTAssembler[type(node.right)](self, node.right)
    
    # Check to see if a cast is required
    leftRegister, rightRegister = self._cast2( leftRegister, rightRegister )
        
    # Format: (opCode, lib, left_register, right_register)
    try:
        opWord, self.assignDChar = OPTABLE[  (type(node.op), self.lib, leftRegister.dchar, rightRegister.dchar ) ]
        warn( "binop {}_{}{} retChar: {}".format(type(node.op), leftRegister.dchar, rightRegister.dchar, self.assignDChar) )
    except KeyError as e:
        if leftRegister.dchar == None or rightRegister.dchar == None:
            raise ValueError( 
                    'Binop did not find arrays: left: {}, right: {}.  Possibly a stack depth issue'.format(
                            leftRegister.name, rightRegister.name) )
        else:
            raise e
    
    # Make/reuse a temporary for output
    outputRegister = self._transmit2(leftRegister, rightRegister)
        
    #_messages.append( 'BinOp: %s %s %s' %( node.left, type(node.op), node.right ) )
    self._codeStream.write( b"".join( (opWord, outputRegister.token, leftRegister.token, rightRegister.token, _NULL_REG ))  )
    
    # Release the leftRegister and rightRegister if they are temporaries and weren't reused.
    self._releaseTemp( leftRegister, outputRegister )
    self._releaseTemp( rightRegister, outputRegister )
    return outputRegister
           
def _call(self, node):
    # ast.Call has the following fields: (in Python <= 3.4)
    # ('func', 'args', 'keywords', 'starargs', 'kwargs')
    info( '$ast.Call: %s'%node.func.id )

    argRegisters = [_ASTAssembler[type(arg)](self, arg) for arg in node.args]
    
    # Would be nice to have a prettier way to fill out the program 
    # than if-else block?
    if len(argRegisters) == 1:
        # TODO: _cast1: We may have to do a cast here, for example "cos(<int>A)"
        opSig = (node.func.id, self.lib, argRegisters[0].dchar)
        # argRegisters.token = self._cast1( argRegisters.token, opSig )
        opCode, self.assignDChar = OPTABLE[ opSig ]
        # TODO: should try and use argRegisters instead.
        outputRegister = self._transmit1( argRegisters[0] )

        self._codeStream.write( b"".join( (opCode, outputRegister.token, 
                            argRegisters[0].token, _NULL_REG, _NULL_REG)  )  )
        
    elif len(argRegisters) == 2:
        argRegisters = self._cast2( *argRegisters )
        opCode, self.assignDChar = OPTABLE[ (node.func.id, self.lib,
                            argRegisters[0].dchar, argRegisters[1].dchar) ]
        outputRegister = self._transmit2( argRegisters[0], argRegisters[1] )
                
        self._codeStream.write( b"".join( (opCode, outputRegister.token, 
                            argRegisters[0].token, argRegisters[1].token, _NULL_REG)  )  )
        
    elif len(argRegisters) == 3: 
        # The where() ternary operator function is currently the _only_
        # 3 argument function
        argRegisters[1], argRegisters[2] = self._cast2( argRegisters[1], argRegisters[2] )

        opCode, self.assignDChar = OPTABLE[ (node.func.id, self.lib,
                            argRegisters[0].dchar, argRegisters[1].dchar, argRegisters.dchar[2]) ]
        # Because we know the first register is the bool, it's the least useful temporary to re-use
        # as it almost certainly must be promoted.
        outputRegister = self._transmit3( argRegisters[1], argRegisters[2], argRegisters[0] )
                
        self._codeStream.write( b"".join( (opCode, outputRegister.token, 
                            argRegisters[0].token, argRegisters[1].token, argRegisters[2].token)  )  )
        
    else:
        raise ValueError( "call(): function calls are 1-3 arguments" )
    
    for argReg in argRegisters:
        self._releaseTemp( argReg, outputRegister )
        
    return outputRegister
    
def _compare(self, node):
    info( 'ast.Compare' )
    # "Awkward... this ast.Compare node is," said Yoga disparagingly.  
    # (left,ops,comparators)
    # NumExpr3 does not handle [Is, IsNot, In, NotIn]
    if len(node.ops) > 1:
        raise NotImplementedError( 
                'NumExpr3 only supports binary comparisons (between two elements); try inserting brackets' )
    # Force the node into something the _binop machinery can handle
    node.right = node.comparators.token
    node.op = node.ops[0]
    return _binop(self, node)
   
def _boolop(self, node):
    # Functionally from the NumExpr perspective there's no difference 
    # between boolean binary operations and binary operations
    if len(node.values) != 2:
        raise ValueError( "NumExpr3 supports binary logical operations only, please seperate operations with ()." )
    node.left = node.values[0]
    node.right = node.values[1]
    _binop( self, node)
    
def _unaryop(self, node):
    # Currently only ast.USub is supported, and the node.operand is the 
    # value acted upon.
    operandRegister = _ASTAssembler[type(node.operand)](self, node.operand, inputRegister)
    try:
        opWord, self.assignDChar = OPTABLE[  (type(node.op), self.lib, operandRegister.dchar ) ]
    except KeyError as e:
        if operandRegister.dchar == None :
            raise ValueError( 
                    'Unary did not find operand array {}. Possibly a stack depth issue'.format(
                            operandRegister.name) )
        else:
            raise e
    outputRegister = self._transmit1( operandRegister )
    self._codeStream.write( b"".join( (opWord, outputRegister.token, operandRegister.token, _NULL_REG, _NULL_REG ))  )  
        
    # Release the operandRegister if it was a temporary
    self._releaseTemp(operandRegister, outputRegister)
    return outputRegister
                
def _unsupported(self, node, outputRegisterle=None ):
    raise KeyError( 'unimplmented ASTNode' + type(node) )

# _ASTAssembler is a function dictionary that is used for fast flow-control.
# Think of it being equivalent to a switch-case flow control in C
_ASTAssembler = defaultdict( _unsupported, 
                  { ast.Assign:_assign, 
                    (ast.Assign,-1):_assign_last, 
                    ast.Expr:_expression, 
                    (ast.Expr,-1): _expression_last,
                    ast.Name:_name, 
                    ast.Num:_const, 
                    ast.Attribute:_attribute, 
                    ast.BinOp:_binop, 
                    ast.BoolOp:_boolop, 
                    ast.UnaryOp:_unaryop,
                    ast.Call:_call, 
                    ast.Compare:_compare,
                     } )
######################### END OF AST HANDLERS ##################################

class NumReg(object):
    '''
    Previously tuples were used for registers. Tuples are faster to build but 
    they can't contain logic which becomes a problem.  Also now we can use 
    None for name for temporaries instead of building values.
    '''
    TYPENAME = { 0:'reg', 1:'scalar', 2:'temp', 3:'return' }

    def __init__(self, num, name, ref, dchar, kind, itemsize=0 ):
        self._num = num                     # The number of the register, must be unique
        self.token = pack(_PACK_REG, num)  # I.e. b'\x00' for 0, etc.
        self.name = name                    # The key, can be an int or a str
        self.ref = ref                      # A reference to the underlying array, or a weakref
        self.dchar = dchar                  # The dtype.char of the underlying array
        self.kind = kind                    # one of KIND_ARRAY, KIND_TEMP, KIND_SCALAR, or KIND_RETURN
        self.itemsize = itemsize            # For temporaries, we track the itemsize for allocation efficiency


    def pack(self):
        '''
        TODO: a faster way to get registers parsed? 

        Packs self into something that can be copied directly into a NumExprReg

        struct NumExprReg 
        {
            char          *mem;        // Pointer to array data for scalars and temps (npy_iter used for arrays)
            char           dchar;      // numpy.dtype.char
            npy_uint8      kind;       // 0 = array, 1 = scalar, 2 = temp
            npy_intp       itemsize;   // element size in bytes   (was: memsizes)
            npy_intp       stride;     // How many bytes until next element  (was: memsteps)
        };

        '''
        #  Tricky part is passing in the numpy ndarrays...
        # We have id(self.ref)?
        # Actually why is .ref needed at all?  It's replaced on the NumExpr_run call
        return pack( 'BPcBPP', 
                    self.token, 
                    id(self.ref()) if isinstance(self.ref,weakref.ref) else id(self.ref), 
                    self.kind, 
                    self.itemsize,
                    0 )


    def to_tuple(self):
        '''
        Note that numpy.dtype.itemsize is np.int32 and not recognized by the 
        Python C-api parsing routines!
        '''
        return ( self.token, 
                 self.ref() if isinstance(self.ref,weakref.ref) else self.ref,
                 self.dchar,
                 self.kind,
                 int(self.itemsize) )

    def __hash__(self):
        return self._num

    def __lt__(self, other):
        return self._num < other._num

    def __str__(self):
        return "{} | name: {:>12} | dtype: {} | kind {:7} | ref: {}".format(
            self._num, 
            self.name, 
            'N' if self.dchar is None else self.dchar, 
            NumReg.TYPENAME[self.kind], 
            self.ref )

    def __getstate__(self):
        ''' Can't pickle a weakref, and we don't want to pass full arrays. '''
        pickleDict = self.__dict__.copy()
        # Remove weakrefs
        if isinstance( pickleDict['ref'], weakref.ref):
            pickleDict['ref'] = None
        return pickleDict

    # def __setstate__(self, state):
    #     ''' We don't need a __setstate__ magic method for NumReg
    #     '''
    #     self.__dict__ = state




class NumExpr(object):
    """
    
    The self.program attribute is a `bytes` object and consists of operations 
    followed by registers with the form,
    
        opcode + return_reg + arg1_reg + arg2_reg + arg3_reg
        
    Currently opcode is a uint16 and the register numbers are uint8.  However,
    note that numpy.iter limits the maximum number of arguments to 32.
    
    Register tuples have the format: 
        
        (regCode, object, dchar, KIND, name, <max_itemsize>)
        
    where
        regCode: a single byte indicating the register number
        object:   the actual object reference, is `None` for temporaries.
        dchar:    the `object.dtype.char`, or for temporaries the expected value
        KIND:  indicates ndarray, scalar, temporary
        name:     the original name of the array.  It is not required for the 
                  input arguments to run have the same names, just that they
                  be in the same order and dtypes.
        max_itemsize: the largest dtype used in temporary arrays, optional
    """

    
    def __init__(self, expr, lib=LIB_STD, casting=CAST_SAFE, local_dict=None, 
                 stackDepth=1 ):
        """Evaluate a mutliline expression element-wise, using the a NumPy.iter.

        `expr` is a string forming an expression, like 

            neObj = NumExpr( 'c = 2*a + 3*b )    # Builds an NumExpr object
            neObj( a=a, b=b, c=c )       # Executes the calculation

        The values for 'a', 'b', and 'c' will by default be taken from the calling 
        function's frame (through use of sys._getframe()). Alternatively, they 
        can be specifed using the 'local_dict' argument.
        
        Multi-line statements, typically using triple-quote strings, or semi-colon 
        seperated statements, are supported.
        
        
        Parameters
        ----------
        name : DEPRECATED
            use wisdom functions instead.
            
        local_dict : dictionary, optional
            A dictionary that replaces the local operands in current frame. This is 
            generally required in Cython, as Cython does not populate the calling 
            frame variables according to Python standards.

        global_dict : DEPRECATED
            A dictionary that replaces the global operands in current frame.
            Setting to {} can speed operations if you do not call globals.
            global_dict was deprecated as there is little reason for the 
            user to maintain two dictionaries of arrays.
            
        out : DEPRECATED
            use assignment in expr (i.e. 'out=a*b') instead.

        order : {'C', 'F', 'A', or 'K'}, optional
            Currently only 'K' is supported in NumExpr3.

            Controls the iteration order for operands. 'C' means C order, 'F'
            means Fortran order, 'A' means 'F' order if all the arrays are
            Fortran contiguous, 'C' order otherwise, and 'K' means as close to
            the order the array elements appear in memory as possible.  For
            efficient computations, typically 'K'eep order (the default) is
            desired.

        casting : {CAST_SAFE, CAST_NO, CAST_EQUIV, CAST_SAME_KIND, CAST_UNSAFE}, 
                optional 
                (NumPy string repr also accepted)

            Currently only 'safe' is supported in NumExpr3.

            Controls what kind of data casting may occur when making a copy or
            buffering.  Setting this to 'unsafe' is not recommended, as it can
            adversely affect accumulations.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.

        optimization: {OPT_MODERATE}, optional
            Controls what level of optimization the compiler will attempt to 
            perform on the expression to speed its execution.  This optimization
            is performed both by python.compile and numexpr3
            
            * OPT_MODERATE performs simple optimizations, such as minimizing 
                the number of temporary arrays
                
        library: {LIB_STD}, optional
            Indicates which library to use for calculations.  The library must 
            have been linked during the C-extension compilation, such that the 
            associated operations are found in the opTable.
            * LIB_STD is the standard C math.h / cmath.h library
            
            Falls-back to LIB_STD if the other library is not available.  
        """
        # Public
        self.expr = expr
        self.program = None
        self.lib = lib
        self.casting = casting
        # self.outputTarget = None
        # registers is a dict of NumReg objects but it later mutates into a tuple
        self.registers = {}
        # self.unallocatedOutput = False  # sentinel
        # self.isExpression = False # normally we do assignments now.
        self.assignDChar = ''    # The assignment target's dtype.char
        self.assignTarget = None # The current assignment target
        
        # Protected
        # The maximum arguments is 32 due to NumPy, we have space for 254 in NE_REG
        # One can recompile NumPy after changing NPY_MAXARGS to use the full 
        # argument space.
        self._regCount = iter(range(interpreter.MAX_ARGS))
        self._stackDepth = stackDepth # How many frames 'up' to promote outputs
        
        self._codeStream = BytesIO()
        self._occupiedTemps = set()
        self._freeTemps = set()
        self._messages = []           # For debugging
        self._compiled_exec = None    # Handle to the C-api NumExprObject


        # Get references to frames
        call_frame = sys._getframe( self._stackDepth ) 
       
        if local_dict is None:
            self.local_dict = call_frame.f_locals
            # self._global_dict = call_frame.f_globals
        else:
            self.local_dict = local_dict
            # self._global_dict = _global_dict

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
        # pickleDict['_global_dict'] = None
        pickleDict['_codeStream'] = b''

        return pickleDict

    def __setstate__(self, state):
        '''
        Restores NumExpr object via `neObj = pickle.loads(pickledBytes)`
        '''
        self.__dict__ = state
        call_frame = sys._getframe( state['_stackDepth'] )
        self.local_dict = call_frame.f_locals
        # self._global_dict = call_frame.f_globals
       

    def assemble(self):
        ''' 
        NumExpr.assemble() can be used in the context of having a pool of 
        NumExpr objects; it is always called by __init__().
        '''
        # Here we assume the local/global_dicts have been populated with 
        # __init__.  
        # Otherwise we have issues with stackDepth being different depending 
        # on where the method is called from.
        
        forest = ast.parse( self.expr ) 
        # N_forest_m1 = len(forest.body) - 1
                         
        # Iterate over the all trees except the last one, which has magic_output
        # for I, bodyItem in enumerate( forest.body[:-1] ):
        for bodyItem in forest.body[:-1]:
            _ASTAssembler[type(bodyItem)](self, bodyItem)

        # Do the _last_ assignment/expression with magic_output
        bodyItem = forest.body[-1]
        _ASTAssembler[type(bodyItem),-1](self,bodyItem)

        # Mutate the registers into a sorted, unmutable tuple
        self.registers = tuple( [reg for reg in sorted(self.registers.values())] )
        regsToInterpreter = tuple( [reg.to_tuple() for reg in self.registers] )

        # Collate the inputNames as well as the the required outputs
        # self.program view is formed in _assign_last now
        self.program = self._codeStream.getvalue()
        
        # Add self to the wisdom
        wisdom[(self.expr, self.lib, self.casting)] = self
        # self.disassemble() # DEBUG

        # warn( "%%%%regsToInterpreter%%%%")
        # for I, reg in enumerate(regsToInterpreter):
        #     warn( '{}::{}'.format(I,reg) )
        # warn( '%%%%')

        self._compiled_exec = interpreter.CompiledExec( self.program, regsToInterpreter )
        # packedRegs = [reg.pack() for reg in self.registers]
        # self._compiled_exec = interpreter.CompiledExec( self.program, packedRegs )
        # Clean up
        self._codeStream.close()
        
    
    def disassemble( self ):
        global _PACK_REG, _PACK_OP, _NULL_REG
        
        blockLen = calcsize(_PACK_OP) + 4*calcsize(_PACK_REG)
        if len(self.program) % blockLen != 0:
            raise ValueError( 
                    'disassemble: len(progBytes)={} is not divisible by {}'.format(len(self.program,blockLen)) )
        # Reverse the opTable
        reverseOps = {op[0] : key for key, op in OPTABLE.items()}
        progBlocks = [self.program[I:I+blockLen] for I in range(0,len(self.program), blockLen)]
        
        print( "="*78 )
        print( "REGISTERS: " )  # Can be a dict or a tuple
        if isinstance( self.registers, dict ): regs = sorted(self.registers.values())
        else: regs = self.registers

        for reg in regs:
            print( reg.__str__() )


        print( "DISASSEMBLED PROGRAM: " )
        for J, block in enumerate(progBlocks):
            opCode, ret, arg1, arg2, arg3 = unpack( _UNPACK, block )
            if arg3 == ord(_NULL_REG): arg3 = '-'
            if arg2 == ord(_NULL_REG): arg2 = '-'
            if arg1 == ord(_NULL_REG): arg1 = '-'
            
            register = reverseOps[ pack(_PACK_OP, opCode) ]
    
            # For the ast.Nodes we want a 'pretty' name in the output
            # give 
            if hasattr( register[0], '__name__' ):
                opString = register[0].__name__ + "_" + "".join( [str(dchar) for dchar in register[2:] ] ).lower()
            else:
                opString = str(register[0]) + "_" + "".join( [str(dchar) for dchar in register[2:] ] )
            print( '#{:2}, op: {:>12} in ret:{:3} <- args({:>2}|{:>2}|{:>2})'.format(
                    J, opString, ret, arg1, arg2, arg3 ) )
        print( "="*78 )
        
        
    def __call__(self, stackDepth=None, verify=False, **kwargs):
        '''
        A convenience shortcut for `NumExpr.run()`.  Keyword arguments are 
        similar except `verify` defaults to False.  
        '''
        if not stackDepth:
            stackDepth = self._stackDepth + 1

        return self.run( stackDepth=stackDepth, verify=verify, **kwargs)
        
    def run(self, stackDepth=None, verify=False, **kwargs):
        '''
        `run()` is called with keyword arguments as the order of 
        args is based on the Abstract Syntax Tree parse and may be 
        non-intuitive.
        
            e.g. self.run( a=a1, b=b1, out=out_new )
        
        where {a,b,out} were the original names in the expression. The 
        `disassemble()` method can be used to see the original expression names.
        
        Additional keyword arguments are:
            
            stackDepth {None}: Tells the function how 
              many stacks up it was called from. Generally not altered unless
              one is using functional programming.
        
            verify {False}: Resamples the calling frame to grab arrays. 
              There is some overhead associated with grabbing the frames so 
              if inside a loop and using run on the same arrays repeatedly 
              then operate without arguments. 

        '''
        # Not supporting Python 2.7 anymore, so we can mix named keywords and kw_args
        if not stackDepth:
            stackDepth = self._stackDepth
        call_frame = None
        
        # self.registers must be a tuple sorted by the register tokens here
        args = []
        if kwargs:
            # Match kwargs to self.registers.name
            # args = [kwargs[reg.name] for reg in self.registers if reg.kind == _KIND_ARRAY]
            for reg in self.registers:
                if reg.name in kwargs:
                    args.append( kwargs[reg.name] )
                elif reg.kind == _KIND_RETURN:
                    # Unallocated output needs a None in the list
                    args.append(None)

        elif verify: # Renew references to frames
            call_frame = sys._getframe( stackDepth ) 
            local_dict = call_frame.f_locals
            for reg in self.registers:
                if reg.name in local_dict:
                    # Do type checking
                    arg = local_dict[reg.name]
                    if np.isscalar(arg):
                        if np.array(arg).dtype.char != reg.dchar:
                            # Formated error strings would be nice but this is a valid try-except path
                            # and we need the speed.
                            raise TypeError( "local scalar variable has different dtype than in register" )
                    elif isinstance(arg, np.ndarray):
                        if arg.dtype.char != reg.dchar:
                            raise TypeError( "local array variable has different dtype than in register" )
                    else:
                        raise TypeError( "local variable is not a np.ndarray or scalar" )    

                    args.append( arg )
                elif reg.kind == KIND_RETURN:
                    if reg.name in local_dict:
                        # Output can exist even if it didn't used to
                        args.append( local_dict[reg.name] )
                    else:
                        args.append(None)

        else: # Re-use existing arrays
            # We have to __call__ the weakrefs to get the original arrays
            # args = [reg.ref() for reg in self.registers.values() if reg.kind == _KIND_ARRAY]
            for reg in self.registers:
                if reg.kind == _KIND_ARRAY or reg.kind == _KIND_RETURN:
                    if isinstance(reg.ref, weakref.ref):
                        arg = reg.ref()
                        if arg is None: # One of our weak-refs expired.
                            debug( "Weakref expired" )
                            return self.run( verify=True, stackDepth=stackDepth+1 )
                    else:
                        arg = reg.ref
                    args.append(arg)
            
        # TODO: move global_state mutex into C-code.
        with _NE_RUN_LOCK:
            unalloc = self._compiled_exec( *args, **kwargs )

        
        # Promotion of magic output
        if self.assignTarget.ref is None and isinstance(self.assignTarget.name, str):
            # Insert result into calling frame
            if call_frame is None:
                sys._getframe( stackDepth ).f_locals[self.assignTarget.name] = unalloc
            else:
                local_dict[self.assignTarget.name] = unalloc

        return unalloc # end NumExpr.run()


    def _newTemp(self, dchar, name ):
        '''
        Either creates a new temporary register, or if possible re-uses an old 
        one. 
            'dchar' is the ndarray.dtype.char, set to `None` if unknown
            'name' is the NumReg.name, which is either a `int` or `str`
        '''

        if len(self._freeTemps) > 0:
            tempToken = self._freeTemps.pop()
            
            # Check if the previous temporary itemsize was different, and use 
            # the biggest of the two. 
            if name is None:
                # Numbered temporary
                tempRegister = self.registers[tempToken]
                info( "_newTemp: re-use case numer = {}, new dchar = {}".format(tempToken, dchar) )
                tempRegister.itemsize = np.maximum( _DCHAR_ITEMSIZE[dchar], tempRegister.itemsize)
                tempRegister.dchar = dchar
                self._occupiedTemps.add(tempToken)
                return tempRegister

            else:
                # Named temporary is taking over a previous numbered temporary, 
                # so replace the key
                info( "_newTemp: Named temporary is taking over a previous numbered temporary" )
                tempRegister = self.registers.pop(tempToken)
                tempRegister.name = name
                self.registers[name] = tempRegister
                tempRegister.itemsize = np.maximum( _DCHAR_ITEMSIZE[dchar], tempRegister.itemsize)
                tempRegister.dchar = dchar
                self._occupiedTemps.add(name)
                return tempRegister

        # Else case: no free temporaries, create a new one
        tempToken = next( self._regCount )
        if name is None:
            name = tempToken

        info( "_newTemp: creation case for name= {}, dchar = {}".format(name, dchar) )
        self.registers[name] = tempRegister = NumReg( tempToken, name,
            None, dchar, _KIND_TEMP, _DCHAR_ITEMSIZE[dchar] )

        if not isinstance(name,str):
            # Named temporaries cannot be re-used except explicitely
            # Only temporaries that have an 'int' name may be reused
            self._occupiedTemps.add( tempToken )
        return tempRegister
            

    def _releaseTemp(self, tempReg, outputReg):
        # Free a temporary
        # This should not release named temporaries, the user may re-use them 
        # at any point in the program
        if tempReg.token in self._occupiedTemps and tempReg.token != outputReg.token:
            self._occupiedTemps.remove( register )
            self._freeTemps.add( register )

    def _transmit1(self, inputReg1):
        '''
        The function checks the inputReg (the register in which the result 
        of the previous operation is held), and returns an outputReg where
        the output of the operation may be saved.
        '''
        if inputReg1.kind != _KIND_TEMP:
            return self._newTemp( self.assignDChar, None )
        # Else we may be able to re-use the temporary
        inputReg1.itemsize = np.maximum(_DCHAR_ITEMSIZE[inputReg1.dchar], _DCHAR_ITEMSIZE[self.assignDChar])
        inputReg1.dchar = self.assignDChar
        return inputReg1
    
    def _transmit2(self, inputReg1, inputReg2):
        '''
        The function checks the inputReg (the register in which the result 
        of the previous operation is held), and returns an outputReg where
        the output of the operation may be saved.
        '''

        if inputReg1.kind != _KIND_TEMP:
            if inputReg2.kind != _KIND_TEMP:
                return self._newTemp( self.assignDChar, None )
            # Else we may be able to re-use register #2
            inputReg2.itemsize = np.maximum(_DCHAR_ITEMSIZE[inputReg2.dchar], _DCHAR_ITEMSIZE[self.assignDChar])
            inputReg2.dchar = self.assignDChar
            return inputReg2    

        # Else we may be able to re-use register #1
        inputReg1.itemsize = np.maximum(_DCHAR_ITEMSIZE[inputReg1.dchar], _DCHAR_ITEMSIZE[self.assignDChar])
        inputReg1.dchar = self.assignDChar
        return inputReg1

    def _transmit3(self, inputReg1, inputReg2, inputReg3):
        '''
        The function checks the inputReg (the register in which the result 
        of the previous operation is held), and returns an outputReg where
        the output of the operation may be saved.
        '''
        if inputReg1.kind != _KIND_TEMP:
            if inputReg2.kind != _KIND_TEMP:
                if inputReg3.kind != _KIND_TEMP:
                    return self._newTemp( self.assignDChar, None )
                # Else we may be able to re-use register #3
                inputReg3.itemsize = np.maximum(_DCHAR_ITEMSIZE[inputReg3.dchar], _DCHAR_ITEMSIZE[self.assignDChar])
                inputReg3.dchar = self.assignDChar
                return inputReg3   

            # Else we may be able to re-use register #2
            inputReg2.itemsize = np.maximum(_DCHAR_ITEMSIZE[inputReg2.dchar], _DCHAR_ITEMSIZE[self.assignDChar])
            inputReg2.dchar = self.assignDChar
            return inputReg2    

        # Else we may be able to re-use register #1
        inputReg1.itemsize = np.maximum(_DCHAR_ITEMSIZE[inputReg1.dchar], _DCHAR_ITEMSIZE[self.assignDChar])
        inputReg1.dchar = self.assignDChar
        return inputReg1

    def _copy(self, register ):
        opCode, self.assignDChar = OPTABLE[ ('copy', self.lib,
                                register.dchar) ]
        # FIXME: this is silly, _copy should never be writing into a temporary
        outputRegister = self._transmit1( register )
            
        self._codeStream.write( b"".join( (opCode, outputRegister.token, 
                                register.token, _NULL_REG, _NULL_REG)  )  )
        pass

    def _cast1(self, unaryRegister, opSig):
        
        if opSig in OPTABLE:
            return unaryRegister
        
        # Else, function with appropriate dtype doesn't exist, make a new temporary
        # castRegister = self._newTemp( rightD, unaryRegister.name )
        # self._codeStream.write( b"".join( 
        #             (OPTABLE[('cast',self.casting,unaryTup.dchar)][0], castRegister.token, 
        #                         rightRegister.token, _NULL_REG, _NULL_REG) ) )

        raise TypeError( "TODO: implement unary casts: opSig = {}, unaryTup = {}".format(opSig, unaryRegister) )

    def _cast2(self, leftRegister, rightRegister ): 
        # TODO: check if one of the tups is a scalar, and change its dtype 
        # proactively.

        leftD = leftRegister.dchar; rightD = rightRegister.dchar
        # print( "cast2: %s, %s"%(leftD,rightD) ) 
        if leftD == rightD:
            return leftRegister, rightRegister
        elif np.can_cast( leftD, rightD ):

            if leftRegister.kind == _KIND_SCALAR:
                leftRegister.ref = leftRegister.ref.astype(rightD)
                leftRegister.dchar = rightD
                return leftRegister, rightRegister

            # Make a new temporary
            castRegister = self._newTemp( rightD, rightRegister.name )
            
            self._codeStream.write( b"".join( 
                    (OPTABLE[('cast',self.casting,rightD,leftD)][0], castRegister.token, 
                                leftRegister.token, _NULL_REG, _NULL_REG)  ) )
            return castRegister, rightRegister
        elif np.can_cast( rightD, leftD ):

            if rightRegister.kind == _KIND_SCALAR:
                rightRegister.ref = rightRegister.ref.astype(leftD)
                rightRegister.dchar = leftD
                return leftRegister, rightRegister

            # Make a new temporary
            castRegister = self._newTemp( leftD, leftRegister.name )
                        
            self._codeStream.write( b"".join( 
                    (OPTABLE[('cast',self.casting,leftD,rightD)][0], castRegister.token, 
                                rightRegister.token, _NULL_REG, _NULL_REG) ) )
            return leftRegister, castRegister
        else:
            raise TypeError( "cast2(): Cannot cast %s to %s by rule 'safe'"
                            %(np.dtype(leftD), np.dtype(rightD) ) ) 
            
    
    def _cast3(self, leftRegister, midRegister, rightRegister ):
        # _cast3 isn't called by where/tenary so no need for an implementation
        # at present.
        self._messages.append( 'TODO: implement 3-argument casting' )
        return leftRegister, midRegister, rightRegister

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