# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:15:20 2016

@author: Robert A. McLeod
"""
import __future__
import os, inspect, sys, time
import ast
import numpy as np
from collections import defaultdict, OrderedDict
from operator import itemgetter # Fastest way to pull from names from Python stackspace?
from itertools import count

# Cythonization
# Initial try of Cython was slower than Python
#import pyximport; pyximport.install()
# It might be useful to look at cytoolz as a template on how to properly 
# Cythonize the Python-side code.

# We pickle the operation hash table. 
try: import cPickle as pickle
except: import pickle
# Fast bytes construction for passing to virtual machine:
# https://waymoot.org/home/python_string/
try: from cStringIO import StringIO as BytesIO
except: from io import BytesIO
# struct.pack is the quickest way to build 4-byte op words
# All important format characters: https://docs.python.org/2/library/struct.html
from struct import pack, unpack, calcsize

# interpreter.so:
import interpreter
# Import opTable
__neDir = os.path.dirname(os.path.abspath( inspect.getfile(inspect.currentframe()) ))
with open( os.path.join( __neDir, 'lookup.pkl' ), 'rb' ) as lookup:
    OPTABLE = pickle.load( lookup )
    
# Sizes for struct.pack
_NULL_REG =  pack( 'B', 255 )
_PACK_REG = 'B'
_PACK_OP =  'H'
    
# Context for casting
# TODO: maybe we should stick with numpy here, otherwise each np.can_cast() 
# needs a conversion.
CAST_SAFE = 0
CAST_NO = 1
CAST_EQUIV = 2
CAST_SAME_KIND = 3
CAST_UNSAFE = 4
_NE_TO_NP_CAST = { CAST_NO:'no', CAST_EQUIV:'equiv', CAST_SAFE:'safe', 
                   CAST_SAME_KIND:'same_kind', CAST_UNSAFE:'unsafe'}

# Context for optimization effort
OPT_NONE = 0
OPT_MODERATE = 1
OPT_AGGRESSIVE = 2

# Defaults to LIB_STD
LIB_STD = 0    # C++ standard library
LIB_VML = 1    # Intel Vector Math library
#LIB_SIMD = 2  # x86 SIMD extensions

CHECK_ALL = 0  # Normal operation in checking dtypes
CHECK_NONE = 1 # Disable all checks for dtypes if expr is in cache

# Note: int(np.isscalar( np.float32(0.0) )) is used to set REGKIND, so ARRAY == 0 and SCALAR == 1
_REGKIND_ARRAY = 0
_REGKIND_SCALAR = 1
_REGKIND_TEMP = 2
_REGKIND_RETURN = 3
#_REG_LOOKUP = { np.ndarray: REGKIND_ARRAY, \
#             np.int8: REGKIND_SCALAR, np.int16: REGKIND_SCALAR, \
#             np.int32: REGKIND_SCALAR, np.int64: REGKIND_SCALAR, \
#             np.float32: REGKIND_SCALAR, np.float64: REGKIND_SCALAR, \
#             np.complex64: REGKIND_SCALAR, np.complex128: REGKIND_SCALAR, \
#             np.str_: REGKIND_ARRAY, }

# The wisdomBank connects strings to their NumExpr objects, so if the same 
# expression pattern is called, it  will be retrieved from the bank.
# Also this permits serialization.
class __WisdomBankSingleton(dict):
    
    def __init__(self, wisdomName="default_wisdom.pkl" ):
        pass
    
# TODO: add load and dump functionality.
wisdomBank = __WisdomBankSingleton()


def disassemble( progBytes ):
    global _PACK_REG, _PACK_OP, _NULL_REG
    
    structFormat = "".join( (_PACK_OP, _PACK_REG, _PACK_REG, _PACK_REG, _PACK_REG) ).encode('ascii')
    blockLen = calcsize(_PACK_OP) + 4*calcsize(_PACK_REG)
    if len(progBytes) % blockLen != 0:
        raise ValueError( 
                'disassemble: len(progBytes)={} is not divisible by {}'.format(len(progBytes,blockLen)) )
        
    # Reverse the opTable
    reverseOps = {op: key for key, op in OPTABLE.items()}
    
    progBlocks = [progBytes[I:I+blockLen] for I in range(0,len(progBytes), blockLen)]
    
    # TODO: print out namesReg as well?
    
    #       0, op:      cast_di in ret:  3 <- args(  1::  -::  -)
    print( "=======================================================" )
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
            opString = opTuple[0].__name__ + "_" + "".join( [str(dchar) for dchar in opTuple[2:] ] )
        else:
            opString = str(opTuple[0]) + "_" + "".join( [str(dchar) for dchar in opTuple[2:] ] )
        print( '#{:2}, op: {:>12} in ret:{:3} <- args({:>3}::{:>3}::{:>3})'.format(
                J, opString, ret, arg1, arg2, arg3 ) )
    print( "=======================================================" )


# TODO: implement non-default casting, optimization, library, checks
# TODO: do we need name with the object-oriented interface?
# TODO: deprecate out
def evaluate( expr, name=None, local_dict=None, global_dict=None, out=None,
             order='K', casting=CAST_SAFE, optimization=OPT_AGGRESSIVE, 
             library=LIB_STD, checks=CHECK_ALL ):
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
        
    out : DEPRECATED
        use assignment in expr instead.

    order : {'C', 'F', 'A', or 'K'}, optional
        Controls the iteration order for operands. 'C' means C order, 'F'
        means Fortran order, 'A' means 'F' order if all the arrays are
        Fortran contiguous, 'C' order otherwise, and 'K' means as close to
        the order the array elements appear in memory as possible.  For
        efficient computations, typically 'K'eep order (the default) is
        desired.

    casting : {CAST_NO, CAST_EQUIV, CAST_SAFE, CAST_SAME_KIND, CAST_UNSAFE}, 
               optional
        Controls what kind of data casting may occur when making a copy or
        buffering.  Setting this to 'unsafe' is not recommended, as it can
        adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
          
    optimization: {OPT_NONE, OPT_MODERATE, OPT_AGGRESSIVE}, optional
        Controls what level of optimization the compiler will attempt to 
        perform on the expression to speed its execution.  This optimization
        is performed both by python.compile and numexpr3
        
          * OPT_NONE means no optimizations will be performed.
          * OPT_MODERATE performs simple optimizations, such as switching 
            simple divides to multiplies.
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
        raise ValueError( "expr must be specified as a string or unicode" )
    elif not isinstance(expr, (str,bytes)):
        raise ValueError( "expr must be specified as a string or bytes" )
        
    if out != None:
        raise ValueError( "out is depricated, use an assignment expr such as 'out=a*x+2' instead" )
    
    # Context is simply all the keyword arguments in a standard order.
    # context = (order, casting, optimization)
    # Signature is the union of the string expression and the context
    # In NumExpr2 the key has additional features
    # Maybe we should use the AST tree instead then...
    # We could use cForest.co_code
    # 
    # numexpr_key = ('a**b', (('optimization', 'aggressive'), ('truediv', False)), 
    #           (('a', <class 'numpy.float64'>), ('b', <class 'numpy.float64'>)))
    signature = (expr, order, casting, optimization)
    if signature in wisdomBank:
        reassemble( wisdomBank[signature], local_dict, global_dict, out )
    else:
        assemble( signature, local_dict, global_dict, out )
    


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
    # The maximum arguments is due to NumPy, we have space for 255
    MAX_ARGS = 32 

    
    def __init__(self, expr, lib=LIB_STD, local_dict = None, global_dict = None, 
                 stackDepth = 1 ):
        
        self.expr = expr
        self._regCount = np.nditer( np.arange(NumExpr.MAX_ARGS) )
        self._output_buffering = False
        
        self.program = b''
        self.lib = lib
        self.bench = {} # for minimization of time inside 
        
        self._stackDepth = stackDepth # How many frames 'up' to promote outputs
        
        self._occupiedTemporaries = set()
        self._freeTemporaries = set()
        self._messages = []            # For debugging
        
        self.__compiled_exec = None    # Handle to the C-api NumExprObject
        self.__outputs = []
    
        # Public
        self.inputNames = []
        self.outputTarget = None
        self.namesReg = OrderedDict()
        self.codeStream = BytesIO()
        
        self.local_dict = None
        self.global_dict = None
        
        # TODO: Is it faster to have a global versus building the function 
        # dict?
        self._ASTAssembler = defaultdict( self.unsupported, 
                  { ast.Assign:self.assign, ast.Expr:self.expression, \
                    ast.Name:self.name, ast.Num:self.const, \
                    ast.Attribute:self.attribute, ast.BinOp:self.binop, \
                    ast.BoolOp:self.binop,
                    ast.Call:self.call, ast.Compare:self.compare, \
                     } )
    
        self.assemble()

    
    
    def assemble(self, stackDepth=1, local_dict=None, global_dict=None ):
        ''' 
        NumExpr.assemble() can be used in the context of having a pool of 
        NumExpr objects; it is also always called by __init__().
        '''
        # Get references to frames
        call_frame = sys._getframe( stackDepth ) 
        if local_dict is None:
            local_dict = call_frame.f_locals
        self.local_dict = local_dict 
        if global_dict is None:
            global_dict = call_frame.f_globals
        self.global_dict = global_dict
        
        forest = ast.parse( self.expr ) 
        # next( self._regCount )  # Reserve register #0 for the output/return value.  
        
            
        for I, bodyItem in enumerate( forest.body ):
            bodyType = type(bodyItem)
            if bodyType == ast.Assign:
                self._ASTAssembler[bodyType]( bodyItem, terminalAssign=True )
            elif bodyType == ast.Expr:
                # Probably the easiest thing to do is generate the array in the exact
                # same way as Assign but return it rather than promotting it up
                # the frame.
                raise NotImplementedError( 'Backwards support for Expr not in yet.' )
            else:
                raise NotImplementedError( 'Unknown ast body: {}'.format(bodyType) )
                
        # Add the last assignment name as the outputTarget so it can be 
        # promoted up the frame later
        self.outputTarget = bodyItem.targets[0].id
                                            
        # Collate the inputNames as well as the the required outputs
        self.program = self.codeStream.getvalue()
        self.__compiled_exec = interpreter.NumExpr( program=self.program, 
                                             registers=tuple(self.namesReg.values()) )
        
        #print( 'Messages/warnings: ')
        #for message in self._messages:
        #    print( '  ' + message )
        #print( '   Occupied temps: %d, Free temps: %d' %(len(self._occupiedTemporaries), len(self._freeTemporaries)) )
        #print( "======== NumExpr3 benchmarks ========" )
        #print( "Frame call time: %e s" %(t2-t0) )
        #print( "AST construction time: %e s" % (t3-t2) )
        #print( "Function setup time: %e s" % (t_funcsetup1-t_funcsetup0) )
        #print( "Time to build NumExpr3 program from AST: %e s" % (t7-t6) )
        #print( "Total NumExpr3 time: %e s" % (t7-t0) )
        
        #disassemble( self.program )
        

    def print_names(self):
        for val in neObj.namesReg.values(): print( '{}:{}'.format( val[4], val[3])  )
        
    def run(self, *args, stackDepth=1, local_dict=None, global_dict=None,
            ex_uses_vml=False, **kwargs ):
        '''
        It is preferred to call run() with keyword arguments as the order of 
        args is based on the Abstract Syntax Tree parse and may be 
        non-intuitive.
        
            e.g. self.run( a=a1, b=b1, out=out_new )
        
        where {a,b,out} were the original names in the expression.
        '''
        # Get references to frames
        # This is fairly expensive...
        call_frame = sys._getframe( stackDepth ) 
        if local_dict is None:
            local_dict = call_frame.f_locals
        self.local_dict = local_dict
        if global_dict is None:
            global_dict = call_frame.f_globals
        self.global_dict = global_dict
        
        # Build args if it was passed in as names
        # TODO: make NumExpr an attrdict instead.  Much easier that all this 
        # tomfoolery.
        if len(args) == 0:
            # Build args from kwargs and the list of known names
            args = []
            for I, regKey in enumerate(self.namesReg):     
                if regKey in kwargs:
                    args.append( kwargs[regKey] )
                    kwargs.pop( regKey )
                    
        # Add outputs?
        promoteResult = (not self.outputTarget in kwargs) \
                        and ( (not self.outputTarget in self.local_dict) and 
                              (not self.outputTarget in self.global_dict) )
                        
        result = self.__compiled_exec( *args, ex_uses_vml=ex_uses_vml, **kwargs )
        
        if promoteResult:
            # Insert result into higher frames
            # print( "run(): Promoting result {} to calling frame".format(self.outputTarget) )
            self.local_dict[ self.outputTarget ] = result
        
        return result
    

    
    def assign(self, node, terminalAssign=False ):
        #print( 'ast.Assign' )
        
        # node.targets is a list;  It must have a len=1 for NumExpr3
        if len(node.targets) != 1:
            raise ValueError( 'NumExpr3 supports only singleton returns in assignments.' )
        # node.value is likely a BinOp
        
        # _messages.append( "Assign node: %s op assigned to %s" % ( node.value, node.targets[0]) )
        # Call function on target and value
        
        targetTup = self._ASTAssembler[type(node.targets[0])](node.targets[0])
        valueTup = self._ASTAssembler[type(node.value)]( node.value, targetTup )
        
        # Now we know the dtype of valueTup, so we can check for special cases
        # for the assignment target
        if not bool(targetTup) and terminalAssign:
            # Make a return array, OR we can order output buffering.
            # Here set a flag to self.use_output_buffering in the future
            self.need_output_buffering = True
            regToken = pack( _PACK_REG, next( self._regCount ) )
            targetTup = ( regToken, None, valueTup[2], _REGKIND_RETURN, '$ret' )
        if not bool(targetTup):
            # It's an intermediate temporary if terminalAssign == False
            targetTup = self.newTemp_INLINE( valueTup[2] )

        
        #TODO: assign needs to know if it's the last operation (in which case it should make an output array)
        # This 'a*b'program should be writing to register 00...

        return valueTup
        
        
    def expression(self, node ):
        #print( 'ast.Expr' )
        #_messages.append( "Expression node: %s op expression" % ( node.value ) )
        self._ASTAssembler[type(node.value)](node.value)
        # TODO: how to apply to 
        
                    
    def name(self, node ):
        #print( 'ast.Name' )
        # node.ctx is probably not something we care for.
        global local_dict, global_dict
        
        node_id = node.id
        if node_id in self.namesReg:
            regTup = self.namesReg[node_id]
            regToken = regTup[0]
        else:
            
            # Get address
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
                regTup = self.newTemp_INLINE( None, name = node_id )
            
        # codeView.append( "reg%d_%s_%s|" % (ord(regToken),node.id,regTup[1].dtype.descr[0][1] ) )

        return regTup

            
    
    def const(self, node ):
        #print( 'ast.Const' )
        # Do we need a seperate constants registry or can we just use one and
        # assign it a random illegal name like $1 or something?
        constNo = next( self._regCount )
        regKey = pack( _PACK_REG, constNo )
        token = '${}'.format(constNo)
        
        # It's easier to just use ndim==0 numpy arrays, since PyArrayScalar isn't in the API anymore.
        # Use the _minimum _ dtype available so that we don't accidently upcast.
        # Later we should try and force consts to be of the correct dtype in 
        # Python.
    
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
            
        #print( "DEBUG: const pointer for {} @ {}".format(node.n, hex(constArr.__array_interface__['data'][0])) )
            
        self.namesReg[token] = regTup = ( regKey, constArr, 
                     constArr.dtype.char, _REGKIND_SCALAR, node.n.__str__() )
        #constReg.append( regTup )
        
        #codeView.append( "const%d_%s|" % (ord(regKey), constArr.dtype.descr[0][1])  )
        return regTup
        
    def attribute(self, node ):
        #print( 'ast.Attr' )
        # An attribute has a .value node which is a Name, and .value.id is the 
        # module/class reference.  Then .attr is the attribute reference that 
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
            if className in local_dict:
                classRef = local_dict[className]
                if node.attr in classRef.__dict__:
                    arr = local_dict[className].__dict__[node.attr]
            # Globals is, as usual, slower than the locals, so we prefer not to 
            # search it.  
            elif className in global_dict:
                classRef = local_dict[className]
                if node.attr in classRef.__dict__:
                    arr = global_dict[className].__dict__[node.attr]
            #print( "Found attribute = %s" % arg )
            
            if arr is not None and not hasattr(arr,'dtype'):
                # Force lists and native arrays to be numpy.ndarrays
                arr = np.asarray( arr )

            # Build tuple and add to the namesReg
            self.namesReg[attrName] = regTup = \
                         (regToken, arr, arr.dtype.char, int(np.isscalar(arr)), attrName )
            
        #print( "Found attribute: %s, registry location %d, dtype %s" % 
        #      (attrName, ord(regToken), regTup[2].descr[0][1]) )
        return regTup
        

        
    def binop(self, node, outputTup=None ):
        #print( 'ast.Binop' )
        # (left,op,right)
        # How to get precidence ordering right here?
        # Probably I have to return the dtype.

        leftTup = self._ASTAssembler[type(node.left)](node.left)
        rightTup = self._ASTAssembler[type(node.right)](node.right)
        
        # Check to see if a cast is required
        leftTup, rightTup = self.cast2( leftTup, rightTup )
          
        # Make/reuse a temporary for output
        if not outputTup:
            outputTup = self.newTemp_INLINE( leftTup[2] )
            #outputKey = lockTemp_INLINE()
            #namesReg[b'$'+outputKey] = outputTup = ( outputKey, None, leftTup[2], REGTYPE_TEMP )
            
        # Named intermediate with unknown dtype
        if outputTup[2] == None: 
            # Ergh how to find the name...
            #for key, value in namesReg.items():
            #    print( str(key) + " :: " + str(value) )
            #raise NotImplementedError( 'Urgh...' )
            # List instead of tuple?
            # Slow work-around of searching dict by value?
            for testKey, testTup in self.namesReg.items():
                if testTup == outputTup:
                    #print( "Assigning dtype to named temporary" )
                    self.namesReg[testKey] = outputTup = \
                                 (outputTup[0], outputTup[1], leftTup[2], outputTup[3], outputTup[4] )
                    #print( "namesReg[{}] = {}".format(testKey, self.namesReg[testKey]) )
                    break
            pass
                
        # Format: op(4-bytes), return_register, left_register, right_register
        opWord = OPTABLE[  (type(node.op), self.lib, outputTup[2], leftTup[2], rightTup[2] ) ] 
     
        #_messages.append( 'BinOp: %s %s %s' %( node.left, type(node.op), node.right ) )
        self.codeStream.write( b"".join( (opWord, outputTup[0], leftTup[0], rightTup[0], _NULL_REG ))  )
        
        # Pop off the leftTup and rightTup if they are temporaries
        if leftTup[3] == _REGKIND_TEMP: self.releaseTemp_INLINE(leftTup[0])
        if rightTup[3] == _REGKIND_TEMP: self.releaseTemp_INLINE(rightTup[0])
        return outputTup
        
    def boolop(self, node, outputTup=None ):
        # Functionally from the NumExpr perspective there's no difference 
        # between boolean binary operations and binary operations
        # Note that we never get here, binop is called instead by the lookup
        # dict.
        self.binop( node, outputTup )
        
    def newTemp_INLINE(self, dtype, name = None ):
        if len(self._freeTemporaries) > 0:
            # The maximum dtype.itemsize that goes into a temporary array is 
            # tracked in the C-extension
            regId = self._freeTemporaries.pop()
            #print( 'Re-using temporary: %s' % regId )
            tempTup = ( regId, None, dtype, _REGKIND_TEMP, '${}'.format(ord(regId)) )
        else:
            #print( 'new Temp: ' + str(name) )
            tempNo = next( self._regCount )
            
            regId = pack( _PACK_REG, tempNo )
            if name == None:
                name = regId
            self.namesReg[name] = tempTup = ( 
                    regId, None, dtype, _REGKIND_TEMP, '${}'.format(tempNo) )
        #tempReg.append( newTempTup )
        #print( "Creating new temporary: %s" % newTempKey )
        self._occupiedTemporaries.add( regId )
        
        return tempTup
        


    # Free a temporary
    def releaseTemp_INLINE(self, regId ):
        #print( "Releasing temporary: %s" % regId )
        self._occupiedTemporaries.remove( regId )
        self._freeTemporaries.add( regId )
            
        
    def compare(self, node, outputTup=None ):
        # print( 'ast.Compare' )
        # "Awkward... this ast.Compare node is," said Yoga disparagingly.  
        # (left,ops,comparators)
        # NumExpr3 does not handle [Is, IsNot, In, NotIn]
        
        if len(node.ops) > 1:
            raise NotImplementedError( 
                    'compare(): NumExpr3 only supports binary comparisons (between two elements)' )
        leftTup = self._ASTAssembler[type(node.left)](node.left)
        rightTup = self._ASTAssembler[type(node.comparators[0])](node.comparators[0])
        
        # Check to see if we need to cast
        leftTup,rightTup = self.cast2( leftTup, rightTup )
          
        # Make a new temporary
        if not outputTup:
            outputTup = self.newTemp_INLINE( leftTup[2] )
        
        # Format: op(4-bytes), return_register, left_register, right_register
        self.codeStream.write( b"".join( (OPTABLE[type(node.ops[0])], outputTup[0], 
                                             leftTup[0], rightTup[0], _NULL_REG ))  )
        
        # Pop off the leftTup and rightTup if they are temporaries
        if leftTup[3] == _REGKIND_TEMP: self.releaseTemp_INLINE(leftTup[0])
        if rightTup[3] == _REGKIND_TEMP: self.releaseTemp_INLINE(rightTup[0])

        return outputTup
        
    def cast2(self, leftTup, rightTup ):

            
        leftD = leftTup[2]; rightD = rightTup[2]
        if leftD == rightD:
            return leftTup, rightTup
        elif np.can_cast( leftD, rightD ):
            # Make a new temporary
            castTup = self.newTemp_INLINE( rightD )
            
            self.codeStream.write( b"".join( (OPTABLE[('cast',self.lib,rightD,leftD)], castTup[0], leftTup[0], _NULL_REG, _NULL_REG)  ) )
            return castTup, rightTup
        elif np.can_cast( rightD, leftD ):
            # Make a new temporary
            castTup = self.newTemp_INLINE( leftD )
                        
            self.codeStream.write( b"".join( 
                    (OPTABLE[('cast',self.lib,leftD,rightD)], castTup[0], rightTup[0], _NULL_REG, _NULL_REG) ) )
            return leftTup, castTup
        else:
            raise TypeError( 'cast2(): Cannot cast %s to %s by rule <TODO>' 
                            %(np.dtype(leftD), np.dtype(rightD) ) ) 
                
        
    def cast3(self, leftTup, midTup, rightTup ):
        # This isn't called by where/tenary so why do we need three element 
        # casts at present?
        self._messages.append( 'TODO: implement 3-argument casting' )
        return leftTup, midTup, rightTup


    def call(self, node, outputTup=None ):
        #print( 'ast.Call: {}'.format(node.func.id) )
        # ast.Call has the following fields:
        # ('func', 'args', 'keywords', 'starargs', 'kwargs')
        
        argTups = [self._ASTAssembler[type(arg)](arg) for arg in node.args]
        # Make a new temporary
        if not outputTup:
            # For where(bool,value1,value2) the dtype is not encapsulated in 
            # first argument's dchar (which is bool), so pick the last one.
            outputTup = self.newTemp_INLINE( argTups[-1][2] )
        
        if len(argTups) == 1:
            opCode = OPTABLE[ (node.func.id, self.lib, outputTup[2],
                               argTups[0][2]) ]
            self.codeStream.write( b"".join( (opCode, outputTup[0], 
                                argTups[0][0], _NULL_REG, _NULL_REG)  )  )
            
        elif len(argTups) == 2:
            argTups = self.cast2( *argTups )
            opCode = OPTABLE[ (node.func.id, self.lib, outputTup[2],
                               argTups[0][2], argTups[1][2]) ]
            self.codeStream.write( b"".join( (opCode, outputTup[0], 
                               argTups[0][0], argTups[1][0], _NULL_REG)  )  )
            
        elif len(argTups) == 3: 
            # The where() ternary operator function
            argTups[1], argTups[2] = self.cast2( argTups[1], argTups[2] )
            opCode = OPTABLE[ (node.func.id, self.lib, outputTup[2],
                               argTups[0][2], argTups[1][2], argTups[2][2]) ]
            self.codeStream.write( b"".join( (opCode, outputTup[0], 
                               argTups[0][0], argTups[1][0], argTups[2][0])  )  )
            
        else:
            raise ValueError( "call(): function calls are 1-3 arguments" )
        
        for arg in argTups:
            if arg[3] == _REGKIND_TEMP: self.releaseTemp_INLINE( arg[0] )
            
        return outputTup

    def unsupported(self, node, outputTuple=None ):
        raise KeyError( 'unimplmented ASTNode' + type(node) )
        


if __name__ == "__main__":
    

    ######################
    ###### TESTING  ######
    ######################
    import numexpr as ne2

    # Simple operation, comparison with Ne2 and NumPy for break-even point
    interpreter._set_num_threads(4)
    ne2.set_num_threads(4)
    
    arrSize = int(2**17 - 42) # The minus is to make the last block a different size
    
    print( "Array size: {:.2f}k".format(arrSize/1024 ))
    
    a = np.pi*np.ones( arrSize )
    b = 0.5*np.ones( arrSize )
    c = 42*np.ones( arrSize )
    yesno = np.random.uniform( size=arrSize ) > 0.5
    out = np.empty( arrSize )
    out_ne2 = np.zeros( arrSize )
    out_int = np.zeros( arrSize, dtype='int32' )
    
    t0 = time.time()
    neObj = NumExpr( 'out=a*b' )
    neObj.run( b=b, a=a, out=out )
    t3 = time.time()
    ne2.evaluate( 'a*b', out=out_ne2 )
    t4 = time.time()
    out_np = a*b
    t5 = time.time()
    print( "---------------------" )
    print( "Ne3 completed simple-op a*b: %.2e s"%(t3-t0) )
    print( "Ne2 completed simple-op a*b: %.2e s"%(t4-t3) )
    print( "numpy completed simple-op a*b: %.2e s"%(t5-t4) )
    
    np.testing.assert_array_almost_equal( out_np, out )
    
    # Multi-line with named temporary
    neObj = NumExpr( """mid_result = c + a
out = mid_result*b""" )
    #neObj.print_names()
    neObj.run( c=c, a=a, out=out, b=b )
    np.testing.assert_array_almost_equal( (c+a)*b, out )

    # Where/Ternary
    expr = 'out = where(yesno,a,b)'
    neObj = NumExpr( expr )
    neObj.run( out=out, yesno=yesno, a=a, b=b )
    #neObj.print_names()
    np.testing.assert_array_almost_equal( np.where(yesno,a,b), out )
    
    # Try a std::cmath function
    expr = "out = arccos(b)"
    neObj = NumExpr( expr )
    neObj.run( b=b, out=out )
    np.testing.assert_array_almost_equal( np.arccos(b), out )
    
    # Try a C++/11 cmath function
    # Note behavoir here is different from NumPy... which returns double.
    expr = "out_int = round(a)"
    neObj = NumExpr( expr )
    neObj.run( out_int=out_int, a=a )
    np.testing.assert_array_almost_equal( np.round(a).astype('int32'), out_int )
    
    # Try C++/11 FMA function
    t10 = time.time()
    expr = "out = fma(a,b,c)"
    neObj = NumExpr( expr )
    neObj.run( out=out, a=a, b=b, c=c  )
    t11 = time.time()
    ne2.evaluate( "a*b+c", out=out_ne2 )
    t12 = time.time()
    out_np = a*b+c
    t13 = time.time()
    # FMA doesn't scale well with large arrays, probably because of function 
    # call overhead.
    
    np.testing.assert_array_almost_equal( out_np, out )
    print( "---------------------" )
    print( "Ne3 completed fused multiply-add: %.2e s"%(t11-t10) )
    print( "Ne2 completed multiply-add: %.2e s"%(t12-t11) )
    print( "numpy completed multiply-add: %.2e s"%(t13-t12) )
    
    
    #######################################
    ###### TESTING COMPLEX NUMBERS  #######
    #######################################
    arrSize = 2**17-17
    pi2 = np.pi/2.0
    ncx = np.random.uniform( -pi2, pi2, arrSize ).astype('complex64') + 1j
    ncy = np.complex64(1) + 1j*np.random.uniform( -pi2, pi2, arrSize ).astype('complex64')
    out_c = np.zeros( arrSize, dtype='complex64' )
    out_c_ne2 = np.zeros( arrSize, dtype='complex128' )
    
    t20 = time.time()
    neObj = NumExpr( 'out_c = ncx*ncy'  )
    neObj.run( out_c=out_c, ncx=ncx, ncy=ncy )
    t21 = time.time()
    ne2.evaluate( 'ncx*ncy', out=out_c_ne2 )
    t22 = time.time()
    out_c_np = ncx*ncy
    t23 = time.time()
    
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
    
    #############################################
    ###### TESTING Unallocated Assignment  ######
    #############################################
    print( "---------------------" )
    neObj = NumExpr( 'out_unalloc=2*a + b' )
    test = neObj.run( b=b, a=a )
    # So the promotion works but the zeroth array is not the output 
    # but 'a' in this case.  So you need to reserve register #0 as was 
    # done originally (in assemble() ). Then turn on 
    # output_buffering, or figure out how to pre-allocate with np.broadcast().
    # However in the 'magic' preallocation case you also need to account for 
    # future reductions.
    try:
        np.testing.assert_array_almost_equal( out_unalloc, 2*a+b )
    except AssertionError:
        print( "Magic output returned incorrect array" )
    except NameError:
        print( 'Did not successfully promote output array to calling frame' )
        raise
    
    ########################################
    ###### TESTING COMPLEX Intel VML  ######
    ########################################
    # distutils...
    #interpreter._set_vml_num_threads(4)
    #expr = 'out = a + b'
    #neObj = NumExpr( expr, lib=LIB_VML )
    #neObj.run( out, a, b )

    
    

