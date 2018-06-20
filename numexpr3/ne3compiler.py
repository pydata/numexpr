# -*- coding: utf-8 -*-
'''
NumExpr3 expression compiler
@author: Robert A. McLeod

Compared to NumExpr2 the compiler has been rebuilt to use the CPython module
`ast` to build the Abstract Syntax Tree (AST) representation of the statements,
instead of the NE AST syntax in the old expressions.py.

AST documentation is hosted externally:
    
https://greentreesnakes.readthedocs.io

The goal is to only parse the AST tree once to reduce the amount of time spent
in pure Python compared to NE2.
'''
from typing import Optional, Tuple, Dict

import os, inspect, sys
import ast
import numpy as np
from collections import defaultdict
import weakref
import pickle
from time import perf_counter

# For appending long strings, using a buffer is faster than ''.join()
from io import BytesIO
# struct.pack is the quickest way to build the program as structs
# All important format characters: https://docs.python.org/3/library/struct.html
from struct import pack, unpack, calcsize


# DEBUG:
try:
    from colorama import init, Fore, Back, Style; init()
    def info(text):
        print( ''.join([Fore.GREEN, text, Style.RESET_ALL]))  
    def warn(text):
        print( ''.join([Fore.YELLOW, text, Style.RESET_ALL]))   
    def debug(text):
        print( ''.join([Fore.RED, text, Style.RESET_ALL]))
except ImportError:
    info = print; warn = print; debug = print


# interpreter.so/pyd:
try: 
    import interpreter # For debugging, release should import from .
except ImportError:
    from . import interpreter

# Import opTable
_neDir = os.path.dirname(os.path.abspath( inspect.getfile(inspect.currentframe())))
with open(os.path.join(_neDir, 'lookup.pkl'), 'rb') as lookup:
    OPTABLE = pickle.load(lookup)
    
# Sizes for struct.pack

_PACK_OP =  b'H'
_PACK_REG = b'B'
_NULL_REG =  pack(_PACK_REG, 255)
_UNPACK = b''.join([_PACK_OP,_PACK_REG,_PACK_REG,_PACK_REG,_PACK_REG])
_RET_LOC = -4

# Map np.dtype.char to np.dtype.itemsize
_DCHAR_ITEMSIZE = {dchar:np.dtype(dchar).itemsize for dchar in np.typecodes['All']}
#    gives 0s for strings and unicode, so set them by hand:
_DCHAR_ITEMSIZE['S'] = 1
_DCHAR_ITEMSIZE['U'] = 4
# Also we need a default value for None:
_DCHAR_ITEMSIZE[None] = 0


# Context for casting
CAST_SAFE = 0
CAST_NO = 1
CAST_EQUIV = 2
CAST_SAME_KIND = 3
CAST_UNSAFE = 4
_CAST_TRANSLATIONS = {CAST_SAFE:CAST_SAFE, 'safe':CAST_SAFE, 
                      CAST_NO:CAST_NO, 'no':CAST_NO, 
                      CAST_EQUIV:CAST_EQUIV, 'equiv':CAST_EQUIV, 
                      CAST_SAME_KIND:CAST_SAME_KIND, 'same_kind':CAST_SAME_KIND,
                      CAST_UNSAFE:CAST_UNSAFE, 'unsafe':CAST_UNSAFE}

# Casting suggestions for functions that don't support integers, such as
# all the transcendentals. NumPy actually returns float-16 for u/int8 but 
# we don't support that...
if os.name == 'nt':
    _CAST1_SUGGEST = {'b':'f', 'B':'f', 
                      'h':'f', 'H':'f',
                      'l':'d', 'L':'d',
                      'q':'d', 'Q':'d'}
else: # posix
    _CAST1_SUGGEST = {'b':'f', 'B':'f', 
                      'h':'f', 'H':'f',
                      'i':'d', 'I':'d',
                      'l':'d', 'L':'d'}

# Context for optimization effort
OPT_MODERATE = 0
#OPT_AGGRESSIVE = 1

# Defaults to LIB_STD
LIB_STD = 0     # C++ cmath standard library
#LIB_VML = 1    # Intel Vector Math library
#LIB_YEP = 2  # Yeppp! math library

CHECK_ALL = 0  # Normal operation in checking dtypes
#CHECK_NONE = 1 # Disable all checks for dtypes if expr is in cache

_KIND_ARRAY  = 1
_KIND_RETURN = 2
_KIND_SCALAR = 4
_KIND_TEMP   = 8
_KIND_NAMED  = 16


# CLASS DEFINITIONS
# =================
class NumReg(object):
    '''
    Previously tuples were used for registers. Tuples are faster to build but 
    they can't contain logic which becomes a problem.  Also now we can use 
    None for name for temporaries instead of building values.

    Attributes
    ----------
    * ``num``: The register number, an ``int``
    * ``token``: The register number encoded as ``bytes``
    * ``name``: The unicode representation of the variable name. For named   
      variables this is a ``str`` and for temporaries a ``int``
    * ``dchar``: The ``numpy.dtype.char`` representation of the underlying 
      array or const datatype.  Cannot be mutated.
      Note this changes from Linux to Windows.  Code generation must always be 
    run on the appropriate platform.
    * ``ref``: The reference to the passed array. Normally this is a 
      ``weakref.ref`` so that the ``NumExpr`` object does not stop 
      garbage collection.
    * ``kind``: a bitmask used for flow-control.  One of:
      - ``_KIND_ARRAY`` : a passed-in ``ndarray``
      - ``_KIND_TEMP``  : a named or unnamed temporary allocated by the 
        virtual machine
      - ``_KIND_SCALAR``: a literal constant, such as ``1``
      - ``_KIND_RETURN``: the last assignment target in a statement block. 
      - ``_KIND_NAMED`` : a 
        Can be pre-allocated or magically promoted to the calling frame.
    '''
    TYPENAME = {_KIND_ARRAY:'array', _KIND_SCALAR:'scalar', 
                _KIND_TEMP:'temp', _KIND_RETURN:'return', _KIND_NAMED:'named'}

    __slots__ = ('_num', 'token', 'name', 'ref', 'dchar', 'kind', 'itemsize')

    def __init__(self, num, name, ref, dchar, kind, itemsize=0):
        self._num = num                     # The number of the register, must be unique
        self.token = pack(_PACK_REG, num)  # I.e. b'\x00' for 0, etc.
        self.name = name                    # The key, can be an int or a str
        self.ref = ref                      # A reference to the underlying scalar, or a weakref
        self.dchar = dchar                  # The dtype.char of the underlying array
        self.kind = kind                    # one of {KIND_ARRAY, KIND_TEMP, KIND_SCALAR, KIND_RETURN, KIND_NAMED}
        self.itemsize = itemsize            # For temporaries, we track the itemsize for allocation efficiency

    # def pack(self):
    #     '''
    #     This is a potential prototype for more quickly building NumReg objects.
    #     Most likely a pure C-api solution will be used instead.
    #     '''
    #     return pack('BPcBPP', self.token, id(self.ref()) if isinstance(self.ref,weakref.ref) else id(self.ref), 
    #                 self.kind, self.itemsize, 0)

    def to_tuple(self) -> Tuple:
        '''
        Packs a NumReg into a tuple capable of being parsed by the virtual machine
        function :code:`NumExpr_init()`.

        Note that numpy.dtype.itemsize is np.int32 and not recognized by the 
        Python C-api parsing routines!
        '''
        return (self.token, 
                 self.ref() if isinstance(self.ref,weakref.ref) else self.ref,
                 self.dchar,
                 self.kind,
                 int(self.itemsize))

    def __hash__(self) -> int:
        return self._num

    def __lt__(self, other: object) -> bool:
        return self._num < other._num

    def __str__(self) -> str:
        return '{} | name: {:>12} | dtype: {} | kind {:7} | ref: {}'.format(
            self._num, 
            self.name, 
            'N' if self.dchar is None else self.dchar, 
            NumReg.TYPENAME[self.kind], 
            self.ref)

    def __getstate__(self) -> Tuple:
        ''' Pickling magic method. Array references are removed as one can't 
        pickle a weakref, and we don't want to pass full arrays.'''
        return (self._num, self.token, self.name, 
                None if isinstance( self.ref, weakref.ref) else self.ref, 
                self.dchar, self.kind, self.itemsize)

    def __setstate__(self, state: Tuple) -> None:
        self._num, self.token, self.name, self.ref, self.dchar, self.kind, self.itemsize = state


# TODO: implement non-default casting, optimization, library, checks
def evaluate(expr: str, name: str=None, lib: int=LIB_STD, 
             local_dict: Dict=None, global_dict: Dict=None, out: np.ndarray=None,
             order: str='K', casting: int=CAST_SAFE, optimization: int=OPT_MODERATE, 
             library: int=LIB_STD, checks: int=CHECK_ALL, stackDepth: int=1):
    '''
    Evaluate a mutli-line expression element-wise, using NumPy broadcasting 
    rules. This function is provided as a convience for porting code from 
    `numexpr` 2.6 to the new release. In general new users are advised to use 
    the :code:`NumExpr` class instead.  

    Arguments
    ^^^^^^^^^

    * :code:`expr`: expr is a string forming an expression, like :code:`c = 2*a + 3*b`.
        
      The values for :code:`a` and :code:`b` will by default be taken from the calling 
      function's frame (through use of :code:`sys._getframe()`). Alternatively, they 
      can be specifed using the :code:`local_dict` argument.
    
      Multi-line statements, or semi-colon seperated statements, are supported.

    Keyword Arguments
    ^^^^^^^^^^^^^^^^^

    :code:`name` : DEPRECATED
        use wisdom functions instead.
        
    :code:`local_dict` : dictionary, optional
        A dictionary that replaces the local operands in current frame. This is 
        generally required in Cython, as Cython does not populate the calling 
        frame variables according to Python standards.

    :code:`global_dict` : DEPRECATED
        A dictionary that replaces the global operands in current frame.
        Setting to {} can speed operations if you do not call globals.
        global_dict was deprecated as there is little reason for the 
        user to maintain two dictionaries of arrays.
        
    :code:`out` : DEPRECATED
        use assignment in expr (i.e. :code:`'out=a*b'`) instead.

    :code:`order:code:`: { :code:`'K'`}, optional
        Controls the iteration order for operands. Currently only :code:`'K'` 
        (default NumPy array ordering) is supported in NumExpr3.

    :code:`casting:code:` : {:code:`CAST_SAFE`|:code:`'safe'`}, optional 

        Controls what kind of data casting may occur when making a copy or
        buffering. Explicit cast functions are also supported, see the user guide.

    :code:`optimization`: {:code:`OPT_MODERATE`}, optional
        Controls what level of optimization the compiler will attempt to 
        perform on the expression to speed its execution. :code:`OPT_MODERATE` 
        performs simple optimizations, such as minimizing the number of temporary arrays
            
    :code:`library`: {:code:`LIB_STD`}, optional
        Indicates which library to use for calculations.  The library must 
        have been linked during the C-extension compilation, such that the 
        associated operations are found in the opTable.
          * :code:`LIB_STD` is the standard C math.h / cmath.h library
    '''
    if not isinstance(expr, (str,bytes)):
        raise ValueError('expr must be specified as a string or bytes.')
        
    if out is not None:
        raise ValueError('out is deprecated, use an assignment expr such as "out=a*x+2" instead.')
    if name is not None:
        raise ValueError('name is deprecated, TODO: replace functionality.')    
    if global_dict is not None:
        raise ValueError('global_dict is deprecated, please use only local_dict')   

    casting = _CAST_TRANSLATIONS[casting]
    if casting != CAST_SAFE:
        raise NotImplementedError('only "safe" casting has been implemented at present.')  
        
    if lib != LIB_STD:
        raise NotImplementedError('only LIB_STD casting has been implemented at present.')  

    # Signature is reduced compared to NumExpr2, in that we don't discover the 
    # dtypes.  That is left for the .run() method, where in verify it does 
    # check the input dtypes and emits a TypeError if they don't match.
    signature = (expr, lib, casting)
    if signature in wisdom:
        try:
            return wisdom[signature].run(verify=True, stackDepth=stackDepth+1)
        except TypeError as e:
            # If we get a TypeError one of the inputs dtypes is wrong, so we 
            # need to assemble a new NumExpr object
            pass

    neObj = NumExpr(expr, lib=lib, casting=casting, stackDepth=stackDepth+1)
    return neObj.run(verify=False)
    # End of ne3.evaluate()

class NumExpr(object):
    '''
    The :code:`NumExpr` class is the core encapsulation of the :code:`numexpr3` 
    module. It encapsulates a `CompiledExec` object which is the virtual 
    machine object from the C-api. It also handles all of the parsing of 
    string statements via a functional dictionary that uses :code:`ast` module 
    Nodes as keys.  This approach results in the Abstract Syntax Tree being 
    parsed in a single-pass, whereas NumExpr 2 required multiple passes for 
    different tasks.  See the :code:`_ASTAssembler` dictionary and its 
    associated functions for implementation details.

    Attributes
    ^^^^^^^^^^

    * :code:`program`: The self.program attribute is a `bytes` object and consists of operations 
      followed by registers with the form::
    
        opcode + return_reg + arg1_reg + arg2_reg + arg3_reg
        
      Currently opcode is a uint16 and the register numbers are uint8. This means
      there can be up to 64k operations in the virtual machine and up to 254 
      registers/arguments (255 is reserved) in a code block. However,
      note that the NumPy code :code:`#define NPY_MAXARGS 32` limits the maximum 
      number of arguments to 32.

    * :code:`_codeStream`: a :code:`BytesIO` buffer that is used to build the 
      program. This was found to be the fastest way to construct the large byte
      strings formed.

    TODO: other NumExpr attribs
    '''

    
    def __init__(self, expr, lib=LIB_STD, casting=CAST_SAFE, local_dict=None, 
                 stackDepth=1):
        '''
        Evaluate a mutli-line expression element-wise, using NumPy broadcasting 
        rules::

            neObj = NumExpr('c=2*a+3*b') # Builds an NumExpr object
            neObj()                      # Executes with original arrays
            neObj(verify=True)           # Checks the calling frame for the array names
            neObj(a=foo, b=bar, c=moo)   # Executes the calculation with new arrays
            neObj(**local_dict)          # Unpack a dictionary with variable names as keys

        Multi-line statements, typically using triple-quote strings, or semi-colon 
        seperated statements, are supported. If an intermediate assignment target 
        exists in the calling frame, it is treated by convention as a second 
        (or third, ...) output.  Otherwise it is a named temporary, and it will 
        never be a full-size array::

            big = NumExpr('')
            
        TODO: example with 
        
        Arguments
        ^^^^^^^^^

        * :code:`expr`: expr is a string forming an expression, like :code:`c = 2*a + 3*b`.
            
        The values for :code:`a` and :code:`b` will by default be taken from the calling 
        function's frame (through use of :code:`sys._getframe()`). Alternatively, they 
        can be specifed using the :code:`local_dict` argument.
        
        Multi-line statements, or semi-colon seperated statements, are supported.

        Keyword Arguments
        ^^^^^^^^^^^^^^^^^

        * :code:`local_dict` : dictionary, optional
        A dictionary that replaces the local operands in current frame. This is 
        generally required in Cython, as Cython does not populate the calling 
        frame variables according to Python standards.

        * :code:`order:code:`: { :code:`'K'`}, optional
        Controls the iteration order for operands. Currently only :code:`'K'` 
        (default NumPy array ordering) is supported in NumExpr3.

        * :code:`casting` : {:code:`CAST_SAFE`|:code:`'safe'`}, optional 
        Controls what kind of data casting may occur when making a copy or
        buffering. Explicit cast functions are also supported, see the user guide.

        * :code:`optimization`: {:code:`OPT_MODERATE`}, optional
        Controls what level of optimization the compiler will attempt to 
        perform on the expression to speed its execution. :code:`OPT_MODERATE` 
        performs simple optimizations, such as minimizing the number of temporary arrays
                
        * :code:`library`: {:code:`LIB_STD`}, optional
        Indicates which library to use for calculations.  The library must 
        have been linked during the C-extension compilation, such that the 
        associated operations are found in the opTable.
        - :code:`LIB_STD` is the standard C math.h / cmath.h library

        * :code:`stackDepth`: How many frames up to promote any magic outputs. 
        This should only be relevant if you are doing fancy tricks with 
        :code:`functools.partial` or other functional programming approachs.
        '''
        # Public
        t0 = perf_counter()
        self.expr = expr
        self.program = None
        self.lib = lib
        self.casting = casting
        # registers is a dict of NumReg objects but it later mutates into a tuple
        self.registers = {}
        self.assignDChar = ''    # The assignment target's dtype.char
        self.assignTarget = None # The current assignment target
        
        # Protected
        # The maximum arguments is 32 due to NumPy, we have space for 254 in NE_REG
        # One can recompile NumPy after changing NPY_MAXARGS to use the full 
        # argument space.
        # TODO: some of these could become class-level and be re-used?
        self._regCount = iter(range(interpreter.MAX_ARGS))
        self._stackDepth = stackDepth # How many frames 'up' to promote outputs
        
        self._codeStream = BytesIO()
        self._occupiedTemps = set()
        self._freeTemps = set()
        self._compiled_exec = None    # Handle to the C-api NumExprObject

        self.timings = {}               # For benchmarking

        # Get references to frames
        t1 = perf_counter()
        if local_dict is None:
            call_frame = sys._getframe(self._stackDepth) 
            self.local_dict = call_frame.f_locals
            self._global_dict = call_frame.f_globals
        else:
            self.local_dict = local_dict

        self.timings['frame_call'] = perf_counter() - t1
        self.timings['__init__']   = t1-t0
        self._assemble()

    def __getstate__(self):
        '''
        Preserves NumExpr object via :code:`pickledBytes = pickle.dumps(neObj)`

        For pickling, we have to remove the local_dict attribute as it is not 
        pickleable.  Weak-references to arrays are handled appropriately by
        the NumReg class.
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
        call_frame = sys._getframe(state['_stackDepth'])
        self.local_dict = call_frame.f_locals
        self._global_dict = call_frame.f_globals
       

    def _assemble(self):
        ''' 
        NumExpr._assemble() is always called by __init__().
        '''
        # Here we assume the local_dict has been populated with 
        # __init__. Otherwise we have issues with stackDepth being different 
        # depending on where the method is called from.
        t0 = perf_counter()
        forest = ast.parse(self.expr) 

        t1 = perf_counter()
        # Iterate over the all trees except the last one, which has magic_output
        for bodyItem in forest.body[:-1]:
            _ASTAssembler[type(bodyItem)](self, bodyItem)

        # Do the _last_ assignment/expression with magic_output
        bodyItem = forest.body[-1]
        _ASTAssembler[type(bodyItem),-1](self,bodyItem)

        t2 = perf_counter()
        # Mutate the registers into a sorted, unmutable tuple that the C-api understands
        self.registers = tuple( [reg for reg in sorted(self.registers.values())] )
        regsToInterpreter = tuple( [reg.to_tuple() for reg in self.registers] )

        # Collate the inputNames as well as the the required outputs
        self.program = self._codeStream.getvalue()
        
        # Add self to the wisdom
        wisdom[(self.expr, self.lib, self.casting)] = self
        # If crashing in the C-api calling disassemble here gives good clues
        # self.disassemble() # DEBUG
        t3 = perf_counter()
        self._compiled_exec = interpreter.CompiledExec(self.program, regsToInterpreter)

        # Clean up unneeded attributes and resources
        self._codeStream.close()
        t4 = perf_counter()
        
        self.timings['ast_parse']    = t1-t0
        self.timings['ast_walk']     = t2-t1
        self.timings['to_tuple']     = t3-t2
        self.timings['obj_construct']= t4-t3
        
    
    def disassemble( self ):
        '''
        Prints a formatted representation of the program and registers for 
        debugging purposes.
        '''
        global _PACK_REG, _PACK_OP, _NULL_REG
        
        blockLen = calcsize(_PACK_OP) + 4*calcsize(_PACK_REG)
        if len(self.program) % blockLen != 0:
            raise ValueError( 
                    'disassemble: len(progBytes)={} is not divisible by {}'.format(len(self.program,blockLen)))
        # Reverse the opTable
        reverseOps = {op[0] : key for key, op in OPTABLE.items()}
        progBlocks = [self.program[I:I+blockLen] for I in range(0,len(self.program), blockLen)]
        
        print('='*78)
        print('REGISTERS: ')  # Can be a dict or a tuple
        if isinstance(self.registers, dict): regs = sorted(self.registers.values())
        else: regs = self.registers

        for reg in regs:
            print(reg.__str__())

        print('DISASSEMBLED PROGRAM: ')
        for J, block in enumerate(progBlocks):
            opCode, ret, arg1, arg2, arg3 = unpack(_UNPACK, block)
            if arg3 == ord(_NULL_REG): arg3 = '-'
            if arg2 == ord(_NULL_REG): arg2 = '-'
            if arg1 == ord(_NULL_REG): arg1 = '-'
            
            register = reverseOps[pack(_PACK_OP, opCode)]
    
            # For the ast.Nodes we want a 'pretty' name in the output
            if hasattr(register[0], '__name__'):
                opString = register[0].__name__ + '_' + ''.join( [str(dchar) for dchar in register[2:] ] ).lower()
            else:
                opString = str(register[0]) + '_' + ''.join( [str(dchar) for dchar in register[2:] ] )
            print('#{:2}, op: {:>12} in ret:{:3} <- args({:>2}|{:>2}|{:>2})'.format(
                    J, opString, ret, arg1, arg2, arg3))
        print( '='*78 )
        
        
    def __call__(self, stackDepth=None, verify=False, **kwargs):
        '''
        A convenience shortcut for :code:`NumExpr.run()`. Keyword arguments are 
        identical.
        '''
        if not stackDepth:
            stackDepth = self._stackDepth + 1

        return self.run(stackDepth=stackDepth, verify=verify, **kwargs)
        
    def run(self, stackDepth=None, verify=False, **kwargs):
        '''
        :code:`run()` is typicall called with keyword arguments as the order of 
        args is based on the Abstract Syntax Tree parse and may be 
        non-intuitive. See the class definition for calling semantics.
        
        Keyword Arguments
        ^^^^^^^^^^^^^^^^^
            
        * :code:`stackDepth` {None}: Tells the function how 
              many stacks up it was called from. Generally not altered unless
              one is using functional programming.
        * :code:`verify` {False}: Resamples the calling frame to grab arrays. 
              There is some overhead associated with grabbing the frames so 
              if inside a loop and using :code:`run()` on the same arrays repeatedly 
              then operate without arguments. 

        '''
        t0 = perf_counter()
        # Not supporting Python 2.7 anymore, so we can mix named keywords and kw_args
        if not stackDepth:
            stackDepth = self._stackDepth
        call_frame = None
        
        # self.registers must be a tuple sorted by the register tokens here
        args = []
        if kwargs:
            # info('run case kwargs')
            # Match kwargs to self.registers.name
            # args = [kwargs[reg.name] for reg in self.registers if reg.kind == _KIND_ARRAY]
            for reg in self.registers:
                if reg.name in kwargs:
                    args.append(kwargs[reg.name])
                elif reg.kind == _KIND_RETURN and reg.ref is None:
                    # Unallocated output needs a None in the list
                    args.append(None)

        elif verify: # Renew references from frames
            # info('run case renew from frames')
            call_frame = sys._getframe(stackDepth) 
            local_dict = call_frame.f_locals
            for reg in self.registers:
                if reg.name in local_dict:
                    # Do type checking
                    arg = local_dict[reg.name]
                    if np.isscalar(arg):
                        if np.array(arg).dtype.char != reg.dchar:
                            # Formated error strings would be nice but this is a valid try-except path
                            # and we need the speed.
                            raise TypeError('local scalar variable has different dtype than in register')
                    elif isinstance(arg, np.ndarray):
                        if arg.dtype.char != reg.dchar:
                            raise TypeError('local array variable has different dtype than in register')
                    else:
                        raise TypeError('local variable is not a np.ndarray or scalar')    

                    args.append( arg )
                elif reg.kind == _KIND_RETURN:
                    if reg.name in local_dict:
                        # Output can exist even if it didn't used to
                        args.append(local_dict[reg.name])
                    else:
                        args.append(None)

        else: # Grab arrays from existing weak references
            # We have to __call__ the weakrefs to get the original arrays
            # args = [reg.ref() for reg in self.registers.values() if reg.kind == _KIND_ARRAY]
            for reg in self.registers:
                if reg.kind & (_KIND_ARRAY|_KIND_RETURN):
                    if isinstance(reg.ref, weakref.ref):
                        arg = reg.ref()
                        if arg is None: # One of our weak-refs expired.
                            # debug('Weakref expired')
                            return self.run(verify=True, stackDepth=stackDepth+1)
                    else: # Likely implies a named scalar.
                        arg = reg.ref
                    args.append(arg)
            
        self.timings['run_pre'] = perf_counter() - t0
        unalloc = self._compiled_exec(*args, **kwargs)
        t0 = perf_counter()

        # Promotion of magic output
        if self.assignTarget.ref is None and isinstance(self.assignTarget.name, str):
            # Insert result into calling frame
            if call_frame is None:
                sys._getframe( stackDepth ).f_locals[self.assignTarget.name] = unalloc
            else:
                local_dict[self.assignTarget.name] = unalloc

        self.timings['run_post'] = perf_counter() - t0
        return unalloc # end NumExpr.run()


    def _newTemp(self, dchar, name ):
        '''
        Either creates a new temporary register, or if possible re-uses an old 
        one.

        * :code:`'dchar'` is the :code:`ndarray.dtype.char`, set to :code:`None` if unknown
        * :code:`'name'` is the :code:`NumReg.name`, which is either a :code:`int` or :code:`str`
        '''

        if len(self._freeTemps) > 0:
            tempToken = self._freeTemps.pop()
            
            # Check if the previous temporary itemsize was different, and use 
            # the biggest of the two. 
            if name is None:
                # Numbered temporary
                tempRegister = self.registers[tempToken]
                # info('_newTemp: re-use case numer = {}, new dchar = {}'.format(tempToken, dchar))
                tempRegister.itemsize = np.maximum(_DCHAR_ITEMSIZE[dchar], tempRegister.itemsize)
                tempRegister.dchar = dchar
                self._occupiedTemps.add(tempToken)
                return tempRegister

            else:
                # Named temporary is taking over a previous numbered temporary, 
                # so replace the key
                # info('_newTemp: Named temporary is taking over a previous numbered temporary')
                tempRegister = self.registers.pop(tempToken)
                tempRegister.name = name
                self.registers[name] = tempRegister
                tempRegister.itemsize = np.maximum(_DCHAR_ITEMSIZE[dchar], tempRegister.itemsize)
                tempRegister.dchar = dchar
                self._occupiedTemps.add(name)
                return tempRegister

        # Else case: no free temporaries, create a new one
        tempToken = next( self._regCount )
        if name is None:
            name = tempToken
        
        # info('_newTemp: creation case for name= {}, dchar = {}'.format(name, dchar))
        self.registers[name] = tempRegister = NumReg(tempToken, name,
            None, dchar, _KIND_TEMP, _DCHAR_ITEMSIZE[dchar])

        if not isinstance(name,str):
            # Named temporaries cannot be re-used except explicitely
            # Only temporaries that have an 'int' name may be reused
            self._occupiedTemps.add(tempToken)
        return tempRegister
            

    def _releaseTemp(self, tempReg, outputReg):
        '''Free a temporary. This should not release named temporaries, the 
        user may re-use them at any point in the program.'''
        if tempReg.token in self._occupiedTemps and tempReg.token != outputReg.token:
            self._occupiedTemps.remove(tempReg)
            self._freeTemps.add(tempReg)

    def _transmit1(self, inputReg1):
        '''
        The function checks the inputReg (the register in which the result 
        of the previous operation is held), and returns an outputReg where
        the output of the operation may be saved.
        '''
        if inputReg1.kind != _KIND_TEMP:
            return self._newTemp(self.assignDChar, None)
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
                return self._newTemp(self.assignDChar, None)
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
                    return self._newTemp(self.assignDChar, None)
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

    def _copy(self, targetReg, valueReg ):
        '''
        A copy operation. Can happen with trival ops such as :code:`NumExpr('x=a')`.
        '''
        # debug( 'copying valueReg (%s) to targetReg (%s)'%(valueReg.dchar, targetReg.dchar) )
        opCode, self.assignDChar = OPTABLE[('copy', self.lib,
                                valueReg.dchar)]
       
        self._codeStream.write(b''.join( (opCode, targetReg.token, 
                                valueReg.token, _NULL_REG, _NULL_REG)))
        return targetReg

    def _func_cast(self, unaryRegister, opSig):
        '''
        A helper cast for functions called with integer arguments but where only 
        floating-point versions exist.  Example::

            a = np.arange(32)
            ne3.NumExpr('sqrt(a)')()

        will cast :code:`a` to :code:`np.float64`, the same as NumPy. Generally 
        most functions are available as float32 and float64, occassionally
        complex64 and complex128
        '''
        if opSig in OPTABLE:
            return unaryRegister, opSig

        funcName, castConvention, dchar = opSig
        castDchar = _CAST1_SUGGEST[dchar]
        castSig = (funcName, castConvention, castDchar)
        if not castSig in OPTABLE:
            raise NotImplementedError( 'Could not find match or suitable cast for function {} with dchar {}'.format(funcName, dchar) )

        if unaryRegister.kind != _KIND_TEMP:
            castRegister = self._newTemp(castDchar, None)
        else: # Else we can re-use the temporary
            castRegister = unaryRegister
            castRegister.itemsize = np.maximum(_DCHAR_ITEMSIZE[dchar], _DCHAR_ITEMSIZE[castDchar])
            castRegister.dchar = castDchar
            
        self._codeStream.write( b''.join(
                    (OPTABLE[('cast',castConvention,castDchar,dchar)][0], castRegister.token, 
                                unaryRegister.token, _NULL_REG, _NULL_REG)))
        return castRegister, castSig

        

    def _cast2(self, leftRegister, rightRegister ): 
        '''
        Search for a valid 'safe' cast between two registers, such as in a 
        binomial operation.  If one of the registers is a constant scalar, it
        will be forced to be the pairwise array's type, even if the cast is unsafe.
        '''
        leftD = leftRegister.dchar; rightD = rightRegister.dchar
        # print( '_cast2: %s dtype :%s, \n %s dtype: %s'%(leftRegister.name,leftD, rightRegister.name,rightD) ) 
        
        if leftD == rightD:
            return leftRegister, rightRegister
        elif np.can_cast( leftD, rightD ):  # This has problems if one register is a temporary and the other isn't.

            if leftRegister.kind == _KIND_SCALAR:
                leftRegister.ref = leftRegister.ref.astype(rightD)
                leftRegister.dchar = rightD
                return leftRegister, rightRegister

            # Make a new temporary
            # debug( '_cast2: rightD: {}, name: {}'.format(rightD, rightRegister.name) )
            # castRegister = self._newTemp( rightD, rightRegister.name )

            if leftRegister.kind != _KIND_TEMP:
                castRegister = self._newTemp(rightD, None)
            else: # Else we can re-use the temporary
                castRegister = leftRegister
                castRegister.itemsize = np.maximum(_DCHAR_ITEMSIZE[rightD], _DCHAR_ITEMSIZE[leftD])
                castRegister.dchar = rightD
            
            self._codeStream.write( b''.join(
                    (OPTABLE[('cast',self.casting,rightD,leftD)][0], castRegister.token, 
                                leftRegister.token, _NULL_REG, _NULL_REG)))
            return castRegister, rightRegister
        elif np.can_cast(rightD, leftD):

            if rightRegister.kind == _KIND_SCALAR:
                rightRegister.ref = rightRegister.ref.astype(leftD)
                rightRegister.dchar = leftD
                return leftRegister, rightRegister

            # Make a new temporary
            # debug( '_cast2: leftD: {}, name: {}'.format(leftD, leftRegister.name) )
            # castRegister = self._newTemp( leftD, leftRegister.name )
            if rightRegister.kind != _KIND_TEMP:
                castRegister = self._newTemp(leftD, None)
            else: # Else we can re-use the temporary
                castRegister = rightRegister
                castRegister.itemsize = np.maximum(_DCHAR_ITEMSIZE[rightD], _DCHAR_ITEMSIZE[leftD])
                castRegister.dchar = leftD
                        
            self._codeStream.write( b''.join(
                    (OPTABLE[('cast',self.casting,leftD,rightD)][0], castRegister.token, 
                                rightRegister.token, _NULL_REG, _NULL_REG)))
            return leftRegister, castRegister
        else:
            raise TypeError('cast2(): Cannot cast {} (dchar {}) and {} (dchar {}) by rule "safe"'.format(
                        leftRegister.name, leftRegister.dchar,
                        rightRegister.name, rightRegister.dchar))
            
    
    def _cast3(self, leftRegister, midRegister, rightRegister):
        '''_cast3 isn't called by where/tenary so no need for an implementation
        at present.'''
        warn('TODO: implement 3-argument casting')
        return leftRegister, midRegister, rightRegister

# AST PARSING HANDLERS
# ==============================================================================
# Move the ast parse functions outside of class NumExpr so we can pickle it.
# Pickle cannot deal with bound methods.
# Note: these use 'self', which must be a NumExpr object 
# ('self' is not a reserved keyword in Python)
def _assign(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    AST function handler for an intermediate assignment in a statement block. 
    The convention is that intermediate targets are named targets if they do not 
    exist in the calling namespace and secondary outputs if they have been 
    pre-allocated.
    '''
    # print( '$ast.Assign' )
    # node.targets is a list; It must have a len=1 for NumExpr3 (i.e. no multiple returns)
    # node.value is likely a BinOp, Call, Comparison, or BoolOp
    if len(node.targets) != 1:
        raise ValueError('NumExpr3 supports only singleton returns in assignments.')
    
    valueReg = _ASTAssembler[type(node.value)](self, node.value)
    # Not populating self.assignTarget here, it's only needed for id'ing the 
    # output.
    return _mutate(self, node.targets[0], valueReg)

def _assign_last(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    AST function handler for the last assignment in a statement block. Promotes 
    output magically if it has not been pre-allocated.
    '''
    # info( '$ast.Assign, flow: LAST' )
    
    if len(node.targets) != 1:
        raise ValueError('NumExpr3 supports only singleton returns in assignments.')
    
    valueReg = _ASTAssembler[type(node.value)](self, node.value)
    self.assignTarget = _mutate_last(self, node.targets[0], valueReg)
    return self.assignTarget

def _expression(self: NumExpr, node: ast.AST) -> None:
    raise SyntaxError('NumExpr3 expressions can only be the last line in a statement.')        

def _expression_last(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    The statement block can end without an assignment, in which case the output 
    is implicitly allocated and returned.
    '''
    # info(  '$ast.Expression, flow: LAST' )
    
    valueReg = _ASTAssembler[type(node.value)](self, node.value)
    # Make a substitute output node
    targetNode = ast.Name('$out', None)
    self.assignTarget = _mutate_last(self, targetNode, valueReg)
    self.assignTarget.name = self.assignTarget._num
    return self.assignTarget
        
def _mutate(self: NumExpr, targetNode: ast.AST, valueReg: NumReg) -> NumReg:
    '''
    Used for intermediate assignment targets. This takes the valueReg and 
    mutates targetReg into it.

    Cases:
      1. targetReg is KIND_ARRAY or KIND_SCALAR in which case it has been
         pre-allocated and is a secondary return.
      2. targetReg is KIND_NAMED in which case it's a _named_ temporary. I.e. 
         a temporary that we cannot re-use because it was assigned as an 
         intermediate assignment target.

    In some cases this will require a new temporary.  For example, if the output 
    dtype is smaller than the valueReg.dchar it can't be mutated to a named 
    output.
    '''
    # print(f'Mutating: {targetNode} and {valueReg}')

    if isinstance(targetNode, ast.Name):
        if valueReg.kind & (_KIND_ARRAY|_KIND_SCALAR):
            # This is a copy, like NumExpr( 'y=x' )
            targetReg = _ASTAssembler[type(targetNode)](self, targetNode)
            targetReg.dchar = valueReg.dchar
            return self._copy(targetReg, valueReg)
        # else KIND_TEMP|KIND_NAMED
        nodeId = targetNode.id
        if nodeId in self.registers:
            # Pre-existing, pre-known output.
            # Often in-line operation, e.g. NumExpr('b=2*b')
            # Should we count how many times each temporary is used?
            # As this is a case where if the temp is used twice we can't 
            # get rid of it, but if it's used once we can pop it.
            targetReg = self.registers[nodeId]
            # Intermediate assignment targets keep their KIND
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
            raise NotImplementedError('TODO: rewind program')
        
    elif isinstance(targetNode, ast.Attribute):
        raise NotImplementedError('TODO: assign to attributes ')
    else:
        raise SyntaxError('Illegal NumExpr assignment target: {}'.format(targetNode))
    pass


def _mutate_last(self: NumExpr, targetNode: ast.AST, valueReg: NumReg) -> NumReg:
    '''
    Used for logic control for final assignment targets.  Assignment targets can be 
    an ast.Name or ast.Attribute 

    There's a few-ish cases:
      1. valueRegister is a temp, in which case see if it can mutate to a KIND_RETURN
      2. targetNode previously named, in which case program must be rewound
      3. valueRegister is an array or scalar, in which case we need to do a copy
      4. valueRegister is an attribute, which is the same as name but needs one level of indirection
    
    '''
    if isinstance(targetNode, ast.Name):
        if valueReg.kind & (_KIND_ARRAY|_KIND_SCALAR):
            # This is a copy, like NumExpr( 'y=x' )
            targetReg = _ASTAssembler[type(targetNode)](self, targetNode)
            targetReg.dchar = valueReg.dchar
            targetReg.kind = _KIND_RETURN
            return self._copy(targetReg, valueReg)

        nodeId = targetNode.id
        if nodeId in self.registers:
            # Pre-existing, pre-known output.
            # Often in-line operation, e.g. NumExpr('b=2*b')
            # warn('WARNING: IN-LINE CREATES AN EXTRA TEMP')
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
                if type(nodeRef) != np.ndarray: nodeRef = np.asarray(nodeRef)
                valueReg.ref = nodeRef if np.isscalar(nodeRef) else weakref.ref(nodeRef)

            valueReg.kind = _KIND_RETURN
            self.assignTarget = self.registers[nodeId] = valueReg
            return valueReg

        else:
            # Else mutate valueRegister into assignTarget
            # TODO: rewind
            raise NotImplementedError('TODO: rewind program')

    elif isinstance(targetNode, ast.Attribute):
        raise NotImplementedError('TODO: assign_last to attributes ')
    else:
        raise SyntaxError('Illegal NumExpr assignment target: {}'.format(targetNode))
    pass



           
def _name(self: NumExpr, node: ast.AST) -> NumReg:
    '''AST factory function for NumReg registers from variable names parsed
    by the AST.

    Handles three cases:  
      1. node.id is already in namesReg, in which case re-use it
      2. node.id exists in the calling frame, in which case it's KIND_ARRAY, 
      3. it doesn't, in which case it's a named temporary.
    '''
    # node.ctx (context) is not something that needs to be tracked

    nodeId = node.id
    if nodeId in self.registers:
        # info( 'ast.Name: found {} in namesReg'.format(nodeId) )
        return self.registers[nodeId]

    else: # Get address so we can find the dtype
        if nodeId in self.local_dict:
            nodeRef = self.local_dict[nodeId]
        # Should we get rid of _global_dict?  It's slowing us down. 
        elif nodeId in self._global_dict: 
            nodeRef = self._global_dict[nodeId]
        else:
            nodeRef = None

        if nodeRef is None:
            # info( 'ast.Name: named temporary {}'.format(nodeId) )
            # It's a named temporary.  
            # Named temporaries can re-use an existing temporary but they cannot be
            # re-used except explicitely by the user!
            # TODO: this could also be a return array, check self.assignTarget?
            return self._newTemp(None, nodeId)
            
        else: 
            # It's an existing array or scalar we haven't seen before
            # info( 'ast.Name: new existing array {}'.format(nodeId) )
            regToken = next(self._regCount)

            # We have to make temporary versions of the node reference object
            # if it's not an ndarray in order ot coerce out the dtype.
            # (NumPy scalars have dtypes but not Python floats, ints)
            if np.isscalar(nodeRef):
                dchar = np.asarray(nodeRef).dtype.char
                # scalars cannot be weak-referenced, but it's not a significant memory-leak.
                self.registers[nodeId] = register = NumReg(regToken, nodeId, nodeRef, dchar, _KIND_ARRAY)
            else:
                self.registers[nodeId] = register = NumReg(regToken, nodeId, weakref.ref(nodeRef), nodeRef.dtype.char, _KIND_ARRAY)
            return register
    
def _const(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    AST factory function for building scalar constants from numbers or string
    literals parsed by the AST.  

    The constants are stored and cast such that they do not impose another cast
    operation on the virtual machine. This is the opposite behavoir to NumExpr2 
    where a float64 const could upcast an enormous array.
    '''
    
    # It's easier to just use ndim==0 numpy arrays, since PyArrayScalar 
    # isn't in the API anymore.
    node_n = node.n
    if np.iscomplex(node_n):
        constArr = np.complex64(node_n)
    elif isinstance(node_n, int): 
        constArr = np.asarray(node_n, dtype=int)
    elif isinstance(node_n, float):
        constArr = np.asarray(node_n, dtype=float)
    elif type(node_n) == str or type(node_n) == bytes: 
        constArr = np.asarray(node_n)
    else: 
        raise TypeError( 
            'Unknown constant type for NE3 literal: {} of type {}'.format( 
                node_n, type(node_n)))

    # Const arrays shouldn't be weak references as they are part of the 
    # program and unmutable.
    constNo = next(self._regCount)
    self.registers[constNo] = register = NumReg(constNo, constNo, constArr, 
                    constArr.dtype.char, _KIND_SCALAR)
    return register
        

def _attribute(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    AST factory function for attributes, such as :code:`numpy.pi`. Other 
    than the attribute handle these are always treated similar to 
    :code:`_name()`.

    An attribute has a .value node which is a Name, and .value.id is the 
    module/class reference. Then .attr is the attribute reference that 
    we need to resolve.

    **Only a single deference level is supported** To go deeper would require
    a recursive solution.

    :code:`.real` and :code:`.imag` need special handling because they are actually 
    mapped to function calls.
    '''
    if node.attr == 'imag' or node.attr == 'real':
        return _real_imag(self, node)

    className = node.value.id
    attrName = ''.join( [className, '.', node.attr] )
    
    if attrName in self.registers:
        register = self.registers[attrName]
        regToken = register.token
    else:
        regToken =  next(self._regCount)
        # Get address
        arr = None
        
        if className in self.local_dict:
            classRef = self.local_dict[className]
            if node.attr in classRef.__dict__:
                arr = self.local_dict[className].__dict__[node.attr]
        # Globals is, as usual, slower than the locals, so we prefer not to 
        # search it.  
        elif className in self._global_dict:
            classRef = self._global_dict[className]
            if node.attr in classRef.__dict__:
               arr = self._global_dict[className].__dict__[node.attr]
        
        if np.isscalar(arr):
            # Build tuple and add to the namesReg
            dchar = np.asarray(arr).dtype.char
            self.registers[attrName] = register = NumReg(regToken, attrName, arr, dchar, _KIND_ARRAY)
        else:
            self.registers[attrName] = register = NumReg(regToken, attrName, weakref.ref(arr), arr.dtype.char, _KIND_ARRAY )
    return register
    
def _real_imag(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    This AST function handler builds :code:`_call()` program steps for the use 
    of attribute-like access.

    There is currently no difference between :code:`real(a)` and :code:`a.real`.
    Having a seperate path for slicing existing arrays was considered but it is 
    overly difficult to manage the weak reference.
    '''
    viewName = node.attr


    register = _ASTAssembler[type(node.value)](self, node.value)

    if isinstance(node.value, ast.Name):
        opSig = (viewName, self.lib, register.dchar)
    else:
        opSig = (viewName, self.lib, self.assignDChar)
    opCode, self.assignDChar = OPTABLE[opSig]

    # Make/reuse a temporary for output
    outputRegister = self._newTemp(self.assignDChar, None)

    self._codeStream.write(b''.join((opCode, outputRegister.token, 
                        register.token, _NULL_REG, _NULL_REG)))

    self._releaseTemp(register, outputRegister)
    return outputRegister

def _binop(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    AST function factory for binomial operations, such as :code:`+,-,*,` etc.

    ast.Binop fields are :code:`(left,op,right)`
    '''
    # info('$ast.Binop: %s'%node.op)
    # (left,op,right)
    leftRegister = _ASTAssembler[type(node.left)](self, node.left)
    rightRegister = _ASTAssembler[type(node.right)](self, node.right)
    
    # Check to see if a cast is required
    leftRegister, rightRegister = self._cast2(leftRegister, rightRegister)
        
    # Format: (opCode, lib, left_register, right_register)
    try:
        opWord, self.assignDChar = OPTABLE[(type(node.op), self.lib, leftRegister.dchar, rightRegister.dchar)]
    except KeyError as e:
        if leftRegister.dchar == None or rightRegister.dchar == None:
            raise ValueError( 
                    'Binop did not find arrays: left: {}, right: {}.  Possibly a stack depth issue'.format(
                            leftRegister.name, rightRegister.name))
        else:
            raise e
    
    # Make/reuse a temporary for output
    outputRegister = self._transmit2(leftRegister, rightRegister)
        
    #_messages.append( 'BinOp: %s %s %s' %( node.left, type(node.op), node.right ) )
    self._codeStream.write(b''.join((opWord, outputRegister.token, leftRegister.token, rightRegister.token, _NULL_REG )))
    
    # Release the leftRegister and rightRegister if they are temporaries and weren't reused.
    self._releaseTemp(leftRegister, outputRegister)
    self._releaseTemp(rightRegister, outputRegister)
    return outputRegister
           
def _call(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    AST function factory for a callable, for one to three arguments.
    
    One-function arguments are typical, they may take many forms such as cast
    operations (:code:`float32(x)`) or transcendentals (:code:`sin(x)`) for 
    example.

    Two-function arguments are rarer, an example being :code:`atan2(x, y)`. 
    NE3 supports a number of two-function arguments found in :code:`scipy.misc`.

    The only supported three-function argument is :code:`where(test, a, b)`

    ast.Call fields are: (in Python >= 3.4)
    :code:`('func', 'args', 'keywords', 'starargs', 'kwargs')`

    Only `args` is examined.
    '''
    # info( '$ast.Call: %s'%node.func.id )

    argRegisters = [_ASTAssembler[type(arg)](self, arg) for arg in node.args]
    
    if len(argRegisters) == 1:
        argReg0 = argRegisters[0]
        # _cast1: We may have to do a cast here, for example 'cos(<int>A)'
        opSig = (node.func.id, self.lib, argReg0.dchar)

        argReg0, opSig = self._func_cast(argReg0, opSig)
        opCode, self.assignDChar = OPTABLE[opSig]
        outputRegister = self._transmit1(argReg0)

        self._codeStream.write(b''.join((opCode, outputRegister.token, 
                            argReg0.token, _NULL_REG, _NULL_REG)))
        
    elif len(argRegisters) == 2:
        argReg0, argReg1 = argRegisters
        argRegisters = self._cast2(*argRegisters)
        opCode, self.assignDChar = OPTABLE[(node.func.id, self.lib,
                            argReg0.dchar, argReg1.dchar)]
        outputRegister = self._transmit2(argReg0, argReg1)
                
        self._codeStream.write( b''.join((opCode, outputRegister.token, 
                            argReg0.token, argReg1.token, _NULL_REG)))
        
    elif len(argRegisters) == 3: 
        # The where() ternary operator function is currently the _only_
        # 3 argument function
        argReg0, argReg1, argReg2 = argRegisters
        argReg1, argReg2 = self._cast2(argReg1, argReg2)

        opCode, self.assignDChar = OPTABLE[ (node.func.id, self.lib,
                            argReg0.dchar, argReg1.dchar, argReg2.dchar)]
        # Because we know the first register is the bool, it's the least useful temporary to re-use
        # as it almost certainly must be promoted.
        outputRegister = self._transmit3(argReg1, argReg2, argReg0)
                
        self._codeStream.write( b''.join((opCode, outputRegister.token, 
                            argReg0.token, argReg1.token, argReg2.token)))
        
    else:
        raise ValueError('call(): function calls are 1-3 arguments')
    
    for argReg in argRegisters:
        self._releaseTemp(argReg, outputRegister)
        
    return outputRegister
    
def _compare(self: NumExpr, node: ast.AST) -> None:
    '''
    The equivalent AST factory for :code:`ast.Compare` nodes to :code:`_binop`.

    :code:`ast.Compare` node fields are :code:`(left,ops,comparators)`. Only one 
    right-hand comparison is allowed, use brackets to break up multiple 
    comparisons.

    NumExpr3 does not handle comparisons :code:`[Is, IsNot, In, NotIn]`.
    '''
    # info( 'ast.Compare: left: {}, ops:{}, comparators:{}'.format(node.left, node.ops, node.comparators) )
    # 'Awkward... this ast.Compare node is,' said Yoga disparagingly.  

    if len(node.ops) > 1:
        raise NotImplementedError( 
                'NumExpr3 only supports binary comparisons (between two elements); try inserting brackets' )
    # Force the node into something the _binop machinery can handle
    node.right = node.comparators[0]
    node.op = node.ops[0]
    return _binop(self, node)
   
def _boolop(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    The equivalent AST factory for :code:`ast.Boolop` nodes to :code:`_binop`.
    :code:`ast.Boolop` nodes typically represent bitshift, bitand, and similar
    operations on integers. 

    :code:`ast.Boolop` node fields are :code:`(left,op,right)`. Only one 
    right-hand comparison is allowed, use brackets to break up multiple 
    comparisons.
    '''
    if len(node.values) != 2:
        raise ValueError('NumExpr3 supports binary logical operations only, please separate operations with ().' )
    node.left = node.values[0]
    node.right = node.values[1]
    return _binop(self, node)
    
def _unaryop(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    Currently only :code:`ast.USub`, i.e. the negation operation :code:`'-a', 
    is supported, and the :code:`node.operand` is the value acted upon.
    '''
    operandRegister = _ASTAssembler[type(node.operand)](self, node.operand)
    try:
        opWord, self.assignDChar = OPTABLE[(type(node.op), self.lib, operandRegister.dchar)]
    except KeyError as e:
        if operandRegister.dchar == None :
            raise ValueError( 
                    'Unary did not find operand array {}. Possibly a stack depth issue'.format(
                            operandRegister.name))
        else:
            raise e
    outputRegister = self._transmit1(operandRegister)
    self._codeStream.write(b''.join((opWord, outputRegister.token, operandRegister.token, _NULL_REG, _NULL_REG)))  
        
    # Release the operandRegister if it was a temporary
    self._releaseTemp(operandRegister, outputRegister)
    return outputRegister

def _list(self: NumExpr, node: ast.AST) -> NumReg:
    '''
    Parse a list literal into a numpy.array e.g. 
    :code:`ne3.NumExpr( 'a < [1,2,3]' )`

    Only numbers are supported at present. Mixing floats and integers results 
    in a float array.
    '''
    regToken = next(self._regCount)

    arrayRepr = np.array([element.n for element in node.elts])
    self.registers[regToken] = register = NumReg(regToken, regToken, arrayRepr, arrayRepr.dtype.char, _KIND_ARRAY)
    return register

                
def _unsupported(self: NumExpr, node: ast.AST) -> None:
    raise KeyError('unimplemented ASTNode: ' + type(node))

# _ASTAssembler is a function dictionary that is used for fast flow-control.
# Think of it being equivalent to a switch-case flow control in C
_ASTAssembler = defaultdict(_unsupported, 
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
                    ast.List:_list})
######################### END OF AST HANDLERS ##################################


class _WisdomBankSingleton(dict):
    '''
    The wisdomBank connects strings to their NumExpr objects, so if the same 
    expression pattern is called, it will be retrieved from the bank.
    Also this permits serialization via pickle.
    '''

    def __init__(self, wisdomFile: str='', maxEntries: int=256):
        # Call super
        super(_WisdomBankSingleton, self).__init__(self)
        # attribute dictionary breaks a lot of things in the intepreter
        # dict.__init__(self)
        self.__wisdomFile = wisdomFile
        self.maxEntries = maxEntries
        pass
    
    @property 
    def wisdomFile(self) -> str:
        if not bool(self.__wisdomFile):
            if not os.access('ne3_wisdom.pkl', os.W_OK):
                raise OSError('insufficient permissions to write to {}'.format('ne3_wisdom.pkl'))
            self.__wisdomFile = 'ne3_wisdom.pkl'
        return self.__wisdomFile
    
    @wisdomFile.setter
    def wisdomFile(self, newName: str) -> None:
        '''Check to see if the user has write permisions on the file.'''
        dirName = os.path.dirname(newName)
        if not os.access(dirName, os.W_OK):
            raise OSError('do not have write perimission for directory {}'.format(dirName))
        self.__wisdomFile = newName
    
    def __setitem__(self, key, value):
        # Protection against growing the cache too much
        if len(self) > self.maxEntries:
            # Remove a 10% of random elements from the cache
            entries_to_remove = self.maxEntries // 10

            keysView = list(self.keys())
            for I, cull in enumerate(keysView):
                self.pop(cull)
                if I >= entries_to_remove: 
                    break
                
        super(_WisdomBankSingleton, self).__setitem__(key, value)
         

    def load(self, wisdomFile: Optional[str]=None) -> None:
        '''
        Load the wisdom from a file on disk (or otherwise file-like object).

        wisdomFile should support the :code:`io.IOBase` or similar interface.
        '''
        if wisdomFile == None:
            wisdomFile = self.wisdomFile
            
        with open(wisdomFile, 'rb') as fh:
            self = pickle.load(fh)

    def dump(self, wisdomFile: Optional[str]=None) -> None:
        '''
        Dump the wisdom into a file on disk (or otherwise file-like object). 

        wisdomFile should support the :code:`io.IOBase` or similar interface.
        '''
        if wisdomFile == None:
            wisdomFile = self.wisdomFile
  
        with open(wisdomFile, 'wb') as fh:
            pickle.dump(self, fh)

wisdom = _WisdomBankSingleton()