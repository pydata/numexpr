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
from time import time
import ast
import numpy as np
from collections import defaultdict, OrderedDict

# The operations are pickled
try: import cPickle as pickle
except: import pickle
# For appending long strings, using a buffer is faster than ''.join()
try: from cStringIO import StringIO as BytesIO
except: from io import BytesIO
# struct.pack is the quickest way to build the program as structs
# All important format characters: https://docs.python.org/2/library/struct.html
from struct import pack, unpack, calcsize

# interpreter.so/pyd:
try: 
    import interpreter # For debugging, release should import from .
except ImportError:
    from . import interpreter

# Import opTable
__neDir = os.path.dirname(os.path.abspath( inspect.getfile(inspect.currentframe()) ))
with open( os.path.join( __neDir, 'lookup.pkl' ), 'rb' ) as lookup:
    OPTABLE = pickle.load( lookup )
    
# Sizes for struct.pack
_NULL_REG =  pack( 'B', 255 )
_PACK_REG = 'B'
_PACK_OP =  'H'
    
# Context for casting
CAST_SAFE = 0
CAST_NO = 1
CAST_EQUIV = 2
CAST_SAME_KIND = 3
CAST_UNSAFE = 4
__CAST_TRANSLATIONS = { CAST_SAFE:CAST_SAFE, 'safe':CAST_SAFE, 
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

# The wisdomBank connects strings to their NumExpr objects, so if the same 
# expression pattern is called, it  will be retrieved from the bank.
# Also this permits serialization.
class __WisdomBankSingleton(dict):
    
    def __init__(self, wisdomName="default_wisdom.pkl" ):
        pass
    
# TODO: add load and dump functionality.
wisdomBank = __WisdomBankSingleton()



# TODO: implement non-default casting, optimization, library, checks
# TODO: do we need name with the object-oriented interface?
# TODO: deprecate out
def evaluate( expr, name=None, lib=LIB_STD, 
             local_dict=None, global_dict=None, out=None,
             order='K', casting=CAST_SAFE, optimization=OPT_MODERATE, 
             library=LIB_STD, checks=CHECK_ALL, stackDepth=3 ):
    """
    Evaluate a mutliline expression element-wise, using the a NumPy.iter

    expr is a string forming an expression, like 
        "c = 2*a + 3*b"
        
    The values for "a" and "b" will by default be taken from the calling 
    function's frame (through use of sys._getframe()). Alternatively, they 
    can be specifed using the 'local_dict' or 'global_dict' arguments.
    
    Multi-line statements, or semi-colon seperated statements, are supported.
    
    
    Parameters
    ----------
    name : DEPRECATED
        use wisdom functions instead.
        
    local_dict : dictionary, optional
        A dictionary that replaces the local operands in current frame.

    global_dict : dictionary, optional
        A dictionary that replaces the global operands in current frame.
        Setting to {} can speed operations if you do not call globals.
        
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
        
    if out != None:
        raise ValueError( "out is depricated, use an assignment expr such as 'out=a*x+2' instead." )
    if name != None:
        raise ValueError( "name is depricated, TODO: replace functionality." )    
        

    casting = __CAST_TRANSLATIONS[casting]
    if casting != CAST_SAFE:
        raise NotImplementedError( "only 'safe' casting has been implemented at present." )  
        
    if lib != LIB_STD:
        raise NotImplementedError( "only 'LIB_STD casting has been implemented at present." )  

    # Signature from NE2:
    # numexpr_key = ('a**b', (('optimization', 'aggressive'), ('truediv', False)), 
    #           (('a', <class 'numpy.float64'>), ('b', <class 'numpy.float64'>)))
    signature = (expr, order, casting, optimization)
    if signature in wisdomBank:
        raise NotImplementedError('TODO: wisdom bank')
    else:
        # stackDepth is 2 here because the NumExpr object involves another 
        # layer.
        neObj = NumExpr( expr, lib=lib, casting=casting, stackDepth=stackDepth,
                        local_dict=local_dict, global_dict=global_dict )
        # neObj.assemble() called on __init__()
        
        # Hrm, frame promotion not working in this context
        return neObj.run( check_arrays=False )
    


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

    
    def __init__(self, expr, lib=LIB_STD, casting=CAST_SAFE, local_dict = None, global_dict = None, 
                 stackDepth=2 ):
        
        self.expr = expr
        self._regCount = np.nditer( np.arange(NumExpr.MAX_ARGS) )
        
        self.program = b''
        self.lib = lib
        self.casting = casting
        
        self._stackDepth = stackDepth # How many frames 'up' to promote outputs
        
        self._occupiedTemporaries = set()
        self._freeTemporaries = set()
        self._messages = []            # For debugging
        
        self.__compiled_exec = None    # Handle to the C-api NumExprObject
        self.__outputs = []  # Not used at present
        
        self.__lastOp = False             # sentinel
        self.__unallocatedOutput = False  # sentinel
    
        # Public
        self.inputNames = []  # Not used at present
        self.outputTarget = None
        self.namesReg = OrderedDict()
        self.codeStream = BytesIO()
        
        self.local_dict = None
        self.global_dict = None
        
        # TODO: Is it faster to have a global lookup dict versus building 
        # the function dict per-object?
        self._ASTAssembler = defaultdict( self.__unsupported, 
                  { ast.Assign:self.__assign, ast.Expr:self.__expression, \
                    ast.Name:self.__name, ast.Num:self.__const, \
                    ast.Attribute:self.__attribute, ast.BinOp:self.__binop, \
                    ast.BoolOp:self.__binop,
                    ast.Call:self.__call, ast.Compare:self.__compare, \
                     } )
    
        self.assemble()

    
    
    def assemble(self, local_dict=None, global_dict=None ):
        ''' 
        NumExpr.assemble() can be used in the context of having a pool of 
        NumExpr objects; it is also always called by __init__().
        '''
        # Get references to frames
        call_frame = sys._getframe( self._stackDepth ) 
        if local_dict is None:
            local_dict = call_frame.f_locals
        self.local_dict = local_dict 
        if global_dict is None:
            global_dict = call_frame.f_globals
        self.global_dict = global_dict
        
        forest = ast.parse( self.expr ) 
        N_forest_m1 = len(forest.body) - 1
                         
        for I, bodyItem in enumerate( forest.body ):
            bodyType = type(bodyItem)
            if I == N_forest_m1:
                # print( "Setting lastOp sentinel" )
                self.__lastOp = True
                # Add the last assignment name as the outputTarget so it can be 
                # promoted up the frame later
                if bodyType == ast.Assign:
                    self.outputTarget = bodyItem.targets[0].id
                # Do we need something else for ast.Expr?
                                        
            if bodyType == ast.Assign:
                self._ASTAssembler[bodyType]( bodyItem )
            elif bodyType == ast.Expr:
                # Probably the easiest thing to do is generate magic output 
                # as with Assign but return rather than promote it.
                if not self.__lastOp:
                    raise SyntaxError( "Expressions may only be single statements." )
                    
                # Just an unallocated output case
                self._ASTAssembler[ast.Expr]( bodyItem )
                
            else:
                raise NotImplementedError( 'Unknown ast body: {}'.format(bodyType) )

                                  
        if self.__unallocatedOutput:
            # We need either output buffering or we need to know the size of the 
            # array to allocate.
            self.namesReg.pop(self.__unallocatedOutput[4])
            self.namesReg[self.outputTarget] = self.__unallocatedOutput = \
                         (self.__unallocatedOutput[0], None, self.__unallocatedOutput[2], 
                          _REGKIND_ARRAY, self.outputTarget )
                                            
        # The OrderedDict for namesReg isn't needed if we sort the tuples, 
        # which we need to do such that the magic output isn't out-of-order.
        regsToInterpreter = tuple( sorted(self.namesReg.values() ) )
        
        # Collate the inputNames as well as the the required outputs
        self.program = self.codeStream.getvalue()
        self.__compiled_exec = interpreter.NumExpr( program=self.program, 
                                             registers=regsToInterpreter )

        
    
    def disassemble( self ):
        global _PACK_REG, _PACK_OP, _NULL_REG
        
        structFormat = "".join( (_PACK_OP, _PACK_REG, _PACK_REG, _PACK_REG, _PACK_REG) ).encode('ascii')
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
            print( "{} : name {:6} : dtype {:1} : type {:5}".format(ord(reg[0]), reg[4], reg[2], TYPENAME[reg[3]] ) )
            
            
        print( "DISASSEMBLED PROGRAM: " )
        for J, block in enumerate(progBlocks):
            opCode, ret, arg1, arg2, arg3 = unpack( structFormat, block )
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
        
        
    def run(self, *args, local_dict=None, global_dict=None,
            check_arrays=True, ex_uses_vml=False, **kwargs ):
        '''
        It is preferred to call run() with keyword arguments as the order of 
        args is based on the Abstract Syntax Tree parse and may be 
        non-intuitive.
        
            e.g. self.run( a=a1, b=b1, out=out_new )
        
        where {a,b,out} were the original names in the expression.
        
            check_arrays: {True} Resamples the calling frame to grab arrays. 
              There is some overhead associated with grabbing the frames so 
              if inside a loop and using run on the same arrays repeatedly 
              then try `False`. 
        '''
        

        if check_arrays:
            # Renew references to frames
            call_frame = sys._getframe( self._stackDepth ) 
            if local_dict is None:
                local_dict = call_frame.f_locals
            self.local_dict = local_dict
            if global_dict is None:
                global_dict = call_frame.f_globals
            self.global_dict = global_dict
            
            # Build args if it was passed in as names
            # TODO: a smoother function interface.
            if len(args) == 0:
                # Build args from kwargs and the list of known names
                args = []
                for regKey in self.namesReg:     
                    if regKey in kwargs:
                        args.append( kwargs[regKey] )
                        kwargs.pop( regKey )
        else:
            args = [reg[1] for reg in self.namesReg.values() if reg[3] == _REGKIND_ARRAY]
        
        # Check if the result needs to be added to the calling frame's locals
        promoteResult = (not self.outputTarget in kwargs) \
                        and ( (not self.outputTarget in self.local_dict) and 
                              (not self.outputTarget in self.global_dict) )
        
        if bool(self.__unallocatedOutput):
            print( "ALLOCATE -> {}".format(self.outputTarget) )
            # Get the last operation from the program, and use np.broadcast 
            # to determine the output size?  What if there's a temp?  
            
            # If we have to go down the tree until we get to 
            # Maybe we should track the broadcast all the way through?
            # But what if the user changes the shape of the input arrays?
            unalloc = np.zeros_like(args[-1])
            
            # Sometimes the return is in the middle of the args because 
            # the set.pop() from __occupiedTemps is not deterministic.

            # TODO: re-write this args insertion mess, it can go inside 
            # the if len(args)==0 loop above.
            arrayCnt = 0
            outId = ord(self.__unallocatedOutput[0])
            for reg in self.namesReg.values():
                regId = ord(reg[0])
                if regId >= outId:
                    args = args[:arrayCnt] + [unalloc] + args[arrayCnt:]
                    break
                if reg[3] == _REGKIND_ARRAY:
                    arrayCnt += 1

        else:
            unalloc = None
            
        self.__compiled_exec( *args, ex_uses_vml=ex_uses_vml, **kwargs )
        
        if promoteResult and self.outputTarget is not None:
            # Insert result into calling frame
            if local_dict is None:
                sys._getframe( self._stackDepth ).f_locals[self.outputTarget] = unalloc
                return
            self.local_dict[self.outputTarget] = unalloc
            return 
                           
        if self.outputTarget is None:
            return unalloc
        return
    

    def __assign(self, node ):
        # node.targets is a list; It must have a len=1 for NumExpr3
        # node.value is likely a BinOp, Call, Comparison, or BoolOp
        if len(node.targets) != 1:
            raise ValueError( 'NumExpr3 supports only singleton returns in assignments.' )
        
        # _messages.append( "Assign node: %s op assigned to %s" % ( node.value, node.targets[0]) )
        # Call function on target and value nodes
        targetTup = self._ASTAssembler[type(node.targets[0])](node.targets[0])
        valueTup = self._ASTAssembler[type(node.value)]( node.value, targetTup )
        return valueTup
        
        
    def __expression(self, node ):
        # For expressions we have to return the value instead of promoting it.
        valueTup = self._ASTAssembler[type(node.value)](node.value)
        self.__unallocatedOutput = valueTup
        return valueTup
        
           
    def __name(self, node ):
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
            elif node_id in self.global_dict:
                arr = self.global_dict[node_id]
    
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
                regTup = self.__newTemp( None, name = node_id )
                
        return regTup

            
    def __const(self, node ):
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
        
    def __attribute(self, node ):
        # An attribute has a .value node which is a Name, and .value.id is the 
        # module/class reference. Then .attr is the attribute reference that 
        # we need to resolve.
        # WE CAN ONLY DEREFERENCE ONE LEVEL ('.')
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
            elif className in self.global_dict:
                classRef = self.local_dict[className]
                if node.attr in classRef.__dict__:
                    arr =self.global_dict[className].__dict__[node.attr]
            
            if arr is not None and not hasattr(arr,'dtype'):
                # Force lists and native arrays to be numpy.ndarrays
                arr = np.asarray( arr )

            # Build tuple and add to the namesReg
            self.namesReg[attrName] = regTup = \
                         (regToken, arr, arr.dtype.char, int(np.isscalar(arr)), attrName )
        return regTup
        
    def __magic_output( self, retChar, outputTup ):
        # 
        if outputTup is None:
            outputTup = self.__newTemp( retChar )
        elif outputTup[2] == None: 
            self.namesReg[outputTup[4]] = outputTup = \
                                 (outputTup[0], outputTup[1], retChar, outputTup[3], outputTup[4] )
            if self.__lastOp:
                self.__unallocatedOutput = outputTup
        return outputTup
        
    
    def __binop(self, node, outputTup=None ):
        #print( 'ast.Binop' )
        # (left,op,right)
        leftTup = self._ASTAssembler[type(node.left)](node.left)
        rightTup = self._ASTAssembler[type(node.right)](node.right)
        
        # Check to see if a cast is required
        leftTup, rightTup = self.__cast2( leftTup, rightTup )
          
        # Format: (opCode, lib, left_register, right_register)
        opWord, retChar = OPTABLE[  (type(node.op), self.lib, leftTup[2], rightTup[2] ) ] 
        
        # Make/reuse a temporary for output
        outputTup = self.__magic_output( retChar, outputTup )
            
        #_messages.append( 'BinOp: %s %s %s' %( node.left, type(node.op), node.right ) )
        self.codeStream.write( b"".join( (opWord, outputTup[0], leftTup[0], rightTup[0], _NULL_REG ))  )
        
        # Release the leftTup and rightTup if they are temporaries and weren't reused.
        if leftTup[3] == _REGKIND_TEMP and leftTup[0] != outputTup[0]: 
            self.__releaseTemp(leftTup[0])
        if rightTup[3] == _REGKIND_TEMP and rightTup[0] != outputTup[0]: 
            self.__releaseTemp(rightTup[0])
        return outputTup
        

        
    def __call(self, node, outputTup=None ):
        # ast.Call has the following fields:
        # ('func', 'args', 'keywords', 'starargs', 'kwargs')
        argTups = [self._ASTAssembler[type(arg)](arg) for arg in node.args]
        
        # Would be nice to have a prettier way to fill out the program 
        # than if-else block?
        if len(argTups) == 1:
            opCode, retChar = OPTABLE[ (node.func.id, self.lib,
                               argTups[0][2]) ]
            outputTup = self.__magic_output( retChar, outputTup )
            

            self.codeStream.write( b"".join( (opCode, outputTup[0], 
                                argTups[0][0], _NULL_REG, _NULL_REG)  )  )
            
        elif len(argTups) == 2:
            argTups = self.__cast2( *argTups )
            opCode, retChar = OPTABLE[ (node.func.id, self.lib,
                               argTups[0][2], argTups[1][2]) ]
            outputTup = self.__magic_output( retChar, outputTup )
                    

            self.codeStream.write( b"".join( (opCode, outputTup[0], 
                               argTups[0][0], argTups[1][0], _NULL_REG)  )  )
            
        elif len(argTups) == 3: 
            # The where() ternary operator function is currently the _only_
            # 3 argument function
            argTups[1], argTups[2] = self.__cast2( argTups[1], argTups[2] )
            opCode, retChar = OPTABLE[ (node.func.id, self.lib,
                               argTups[0][2], argTups[1][2], argTups[2][2]) ]
            outputTup = self.__magic_output( retChar, outputTup )
                    
            self.codeStream.write( b"".join( (opCode, outputTup[0], 
                               argTups[0][0], argTups[1][0], argTups[2][0])  )  )
            
        else:
            raise ValueError( "call(): function calls are 1-3 arguments" )
        
        for arg in argTups:
            if arg[3] == _REGKIND_TEMP: self.__releaseTemp( arg[0] )
            
        return outputTup
    
    
    def __newTemp(self, dchar, name = None ):
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
        
    
    # Free a temporary
    def __releaseTemp(self, regId ):
        #print( "Releasing temporary: %s" % regId )
        self._occupiedTemporaries.remove( regId )
        self._freeTemporaries.add( regId )
            
        
    def __compare(self, node, outputTup=None ):
        # print( 'ast.Compare' )
        # "Awkward... this ast.Compare node is," said Yoga disparagingly.  
        # (left,ops,comparators)
        # NumExpr3 does not handle [Is, IsNot, In, NotIn]
        if len(node.ops) > 1:
            raise NotImplementedError( 
                    'NumExpr3 only supports binary comparisons (between two elements); try inserting brackets' )
        # Force the node into something the __binop machinery can handle
        node.right = node.comparators[0]
        node.op = node.ops[0]
        return self.__binop(node, outputTup)
   
    def __boolop(self, node, outputTup=None ):
        # Functionally from the NumExpr perspective there's no difference 
        # between boolean binary operations and binary operations
        # Note that we never get here, binop is called instead by the lookup
        # dict.
        self.binop( node, outputTup )
        
    def __cast2(self, leftTup, rightTup ): 
        leftD = leftTup[2]; rightD = rightTup[2]
        if leftD == rightD:
            return leftTup, rightTup
        elif np.can_cast( leftD, rightD ):
            # Make a new temporary
            castTup = self.__newTemp( rightD )
            
            self.codeStream.write( b"".join( 
                    (OPTABLE[('cast',self.casting,rightD,leftD)][0], castTup[0], 
                             leftTup[0], _NULL_REG, _NULL_REG)  ) )
            return castTup, rightTup
        elif np.can_cast( rightD, leftD ):
            # Make a new temporary
            castTup = self.__newTemp( leftD )
                        
            self.codeStream.write( b"".join( 
                    (OPTABLE[('cast',self.casting,leftD,rightD)][0], castTup[0], 
                             rightTup[0], _NULL_REG, _NULL_REG) ) )
            return leftTup, castTup
        else:
            raise TypeError( 'cast2(): Cannot cast %s to %s by rule <TODO>' 
                            %(np.dtype(leftD), np.dtype(rightD) ) ) 
                
        
    def __cast3(self, leftTup, midTup, rightTup ):
        # This isn't called by where/tenary so no need for an implementation
        # at present.
        self._messages.append( 'TODO: implement 3-argument casting' )
        return leftTup, midTup, rightTup

    def __unsupported(self, node, outputTuple=None ):
        raise KeyError( 'unimplmented ASTNode' + type(node) )
        


if __name__ == "__main__":

    ######################
    ###### TESTING  ######
    ######################
    import numexpr as ne2

    # Simple operation, comparison with Ne2 and NumPy for break-even point
    interpreter._set_num_threads(12)
    ne2.set_num_threads(12)
    
    arrSize = int(2**20-42) # The minus is to make the last block a different size
    
    print( "Array size: {:.2f}k".format(arrSize/1024 ))
    
    
    a = np.pi*np.ones( arrSize )
    b = 0.5*np.ones( arrSize )
    c = 42*np.ones( arrSize )
    yesno = np.random.uniform( size=arrSize ) > 0.5
    out = np.zeros( arrSize )
    out_ne2 = np.zeros( arrSize )
    out_int = np.zeros( arrSize, dtype='int32' )
    
    # Initialize threads with un-tracked calls
    ne2.evaluate( 'a+b+1' )
    neObj = NumExpr( 'out = a + b + 1', stackDepth=1 )
    neObj.run( b=b, a=a, out=out )
    
    # Try a std::cmath function
    # Not very fast for NE3 here, evidently the GNU cmath isn't being 
    # vectorized.
    t60 = time()
    evaluate( 'out_magic1 = sqrt(b)', stackDepth=1 )
    t61 = time()
    ne2.evaluate( 'sqrt(b)' )
    t62 = time()
    np.testing.assert_array_almost_equal( np.sqrt(b), out_magic1 )
    print( "---------------------" )
    print( "Ne3 completed single call: %.2e s"%(t61-t60) )
    print( "Ne2 completed single call: %.2e s"%(t62-t61) )
    
    # Old-school expression with no assignment, just a return.
    neExpr = NumExpr( "a*b", stackDepth=1 )
    expr_out = neExpr.run( a=a, b=b )
    np.testing.assert_array_almost_equal( a*b, expr_out )

    # For some reason NE3 is significantly faster if we do not call NE2 
    # in-between.  I wonder if there's something funny with Python handling 
    # of the two modules.
    t0 = time()
    neObj = NumExpr( 'out=a*b', stackDepth=1 )
    neObj.run( b=b, a=a, out=out )
    t1 = time()
    
    t2 = time()
    ne2.evaluate( 'a*b', out=out_ne2 )
    t3 = time()
    out_np = a*b
    t4 = time()
    print( "---------------------" )
    print( "Ne3 completed simple-op a*b: %.2e s"%(t1-t0) )
    print( "Ne2 completed simple-op a*b: %.2e s"%(t3-t2) )
    print( "numpy completed simple-op a*b: %.2e s"%(t4-t3) )
    np.testing.assert_array_almost_equal( out_np, out )


    # In-place op
    # Not such a huge gap, only ~ 200 %
    neObj = NumExpr( 'out=a+b', stackDepth=1 )
    neObj.run( b=b, a=a, out=out )
    
    t50 = time()
    inplace = NumExpr( 'a = a*a' )
    inplace.run( check_arrays=False )
    t51 = time()
    inplace_ne2 = ne2.evaluate( 'a*a', out=a )
    t52 = time()
    print( "---------------------" )
    print( "Ne3 in-place op: %.2e s"%(t51-t50) )
    print( "Ne2 in-place op: %.2e s"%(t52-t51) )
    del inplace
    
    
    ##############################################
    ###### Multi-line with named temporary  ######
    ##############################################
    # Are there fewer temporaries in the ne2 program?
    # 
    #    # Run once for each of NE2 and NE3 to start interpreters

    t40 = time()
    neObj = NumExpr( 'temp = a*a + b*c - a; out_magic = c / sqrt(temp)', stackDepth=1 )
    result = neObj.run( b=b, a=a, c=c )
    t41 = time()
    temp = ne2.evaluate( 'a*a + b*c - a' )
    out_ne2 = ne2.evaluate( 'c / sqrt(temp)' )
    t42 = time()
    print( "---------------------" )
    print( "Ne3 completed multiline: %.2e s"%(t41-t40) )
    print( "Ne2 completed multiline: %.2e s"%(t42-t41) )
    np.testing.assert_array_almost_equal( c / np.sqrt(a*a + b*c - a), out_magic )

    # Magic output on __call() rather than __binop
    test = evaluate( 'out_magic1=sqrt(a)', stackDepth=1 )
    np.testing.assert_array_almost_equal( np.sqrt(a), out_magic1 )
    
   
    
    '''
     Comparing NE3 versus NE2 optimizations:
     1.) So we have extra casts, we should make sure scalars are the right 
          dtype in Python or in C?  
     2.) ne2 has a power optimization and I don't yet.  I would prefer the 
         pow optimization be at the function level in the C-interpreter.
     3.) We have more temporaries sometimes still?
     4.) Can we do in-place operations with temporaries to further optimize?
         In-place operations are faster in NE2 than binops, and about the same 
         speed in NE3.
     5.) ne2 drops the first b*b into the zeroth (return) register. We would 
         need more logic for when the output is valid as a temporary.
    '''
    
    # Where/Ternary
    expr = 'out = where(yesno,a,b)'
    neObj = NumExpr( expr )
    neObj.run( out=out, yesno=yesno, a=a, b=b )
    #neObj.print_names()
    np.testing.assert_array_almost_equal( np.where(yesno,a,b), out )
    
    # Try a C++/11 cmath function
    # Note behavoir here is different from NumPy... which returns double.
    expr = "out_int = round(a)"
    neObj = NumExpr( expr )
    neObj.run( out_int=out_int, a=a )
    # round() doesn't work on Windows?
    try:
        np.testing.assert_array_almost_equal( np.round(a).astype('int32'), out_int )
    except AssertionError as e:
        print( e )
    
    # Try C++/11 FMA function
    t10 = time()
    expr = "out = fma(a,b,c)"
    neObj = NumExpr( expr )
    neObj.run( out=out, a=a, b=b, c=c  )
    t11 = time()
    ne2.evaluate( "a*b+c", out=out_ne2 )
    t12 = time()
    out_np = a*b+c
    t13 = time()
    # FMA doesn't scale well with large arrays.
    np.testing.assert_array_almost_equal( out_np, out )
    print( "---------------------" )
    print( "Ne3 completed fused multiply-add: %.2e s"%(t11-t10) )
    print( "Ne2 completed multiply-add: %.2e s"%(t12-t11) )
    print( "numpy completed multiply-add: %.2e s"%(t13-t12) )
    
    
    #######################################
    ###### TESTING COMPLEX NUMBERS  #######
    #######################################
    pi2 = np.pi/2.0
    ncx = np.random.uniform( -pi2, pi2, arrSize ).astype('complex64') + 1j
    ncy = np.complex64(1) + 1j*np.random.uniform( -pi2, pi2, arrSize ).astype('complex64')
    out_c = np.zeros( arrSize, dtype='complex64' )
    out_c_ne2 = np.zeros( arrSize, dtype='complex128' )
    
    t20 = time()
    neObj = NumExpr( 'out_c = ncx*ncy'  )
    neObj.run( out_c=out_c, ncx=ncx, ncy=ncy )
    t21 = time()
    ne2.evaluate( 'ncx*ncy', out=out_c_ne2 )
    t22 = time()
    out_c_np = ncx*ncy
    t23 = time()
    
    np.testing.assert_array_almost_equal( out_c_np, out_c )
    
    print( "---------------------" )
    print( "Ne3 completed complex64 ncx*ncy: %.2e s"%(t21-t20) )
    print( "Ne2 completed complex128 ncx*ncy: %.2e s"%(t22-t21) )
    print( "numpy completed complex64 ncx*ncy: %.2e s"%(t23-t22) )
    
    neObj = NumExpr( 'out_c = sqrt(ncx)'  )
    neObj.run( out_c=out_c, ncx=ncx )
    np.testing.assert_array_almost_equal( np.sqrt(ncx), out_c )
    neObj = NumExpr( 'out_c = exp(ncx)'  )
    neObj.run( out_c=out_c, ncx=ncx )
    np.testing.assert_array_almost_equal( np.exp(ncx), out_c )
    neObj = NumExpr( 'out_c = cosh(ncx)'  )
    neObj.run( out_c=out_c, ncx=ncx )
    np.testing.assert_array_almost_equal( np.cosh(ncx), out_c )
    

    ########################################
    ###### TESTING COMPLEX Intel VML  ######
    ########################################
    # distutils...
    #interpreter._set_vml_num_threads(4)
    #expr = 'out = a + b'
    #neObj = NumExpr( expr, lib=LIB_VML )
    #neObj.run( out, a, b )

    #################################
    ###### TEST with striding  ######
    #################################
    # DEBUG: why do calls to NE2 slow down NE3?  Maybe the interpreter 
    # is doing some extra work in the background?
    neObj = NumExpr( 'out=a+b' )
    neObj.run( out, a, b )
    
    da = a[::4]
    db = b[::4]
    out_stride1 = np.empty_like(da)
    out_stride2 = np.empty_like(da)
    t30 = time()
    neObj = NumExpr( 'out_stride1 = da*db' )
    neObj.run( out_stride1, da, db )
    t31 = time()
    ne2.evaluate( 'da*db', out=out_stride2 )
    t32 = time()
    print( "---------------------" )
    print( "Strided computation:" )
    print( "Ne3 completed (strided) a*b: %.2e s"%(t31-t30) )
    print( "Ne2 completed (strided) a*b: %.2e s"%(t32-t31) )
   
    np.testing.assert_array_almost_equal( out_stride1, da*db )


    ####################
    #### COMPARISON ####
    ####################
    out_bool = np.zeros_like(a, dtype='bool')
    neObj = NumExpr( 'out_bool = b > a' )
    # No check of inputs
    neObj.run( check_arrays=False )
    
    
    ###################################################
    # Uint8                                           #
    # A case where NumExpr2 returns the wrong result. #
    ###################################################
    
    lena = np.random.randint( 0, 255, [3600,3200] ).astype('uint8')
    paul = np.random.randint( 0, 255, [3600,3200] ).astype('uint8')
    
    t70 = time()
    evaluate( 'sum_image = lena - paul' )
    t71 = time()
    sum_ne2 = ne2.evaluate( 'lena - paul' )
    t72 = time()
    
    np.testing.assert_array_almost_equal( sum_image, lena-paul )
    print( "uint8 image processing:" )
    print( "Ne3 completed lena - paul: %.2e s"%(t71-t70) )
    print( "Ne2 completed lena - paul: %.2e s"%(t72-t71) )