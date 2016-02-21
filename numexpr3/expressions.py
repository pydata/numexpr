###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

__all__ = ['E']

import operator
import sys
import threading

import numpy

# Declare a double type that does not exist in Python space
double = numpy.double

# The default kind for undeclared variables
defaultType = numpy.float64
if sys.version_info[0] < 3:
    int_ = int
    long_ = long
else:
    int_ = int
    long_ = long

# RAM: Why do we have this twice, once here and once in necompiler?

# Type to kind is a lookup-dict that has the type (null,bool,unsigned,int,float,complex) and the element size in tuples
# It's used for determining casting
type_to_kindorder = { bool:(1,1), numpy.bool:(1,1), 
                numpy.uint32:(2,4), numpy.uint64:(2,8),
                numpy.int32:(3,4), numpy.int64:(3,8), int_:(3,8), long_:(3,8),
                numpy.float32:(4,4), numpy.float64:(4,8), float:(4,8),
                numpy.complex64:(5,4), numpy.complex128:(5,8), complex:(5,8),
                bytes:(6,1), str:(6,1), numpy.string_:(6,1) }
# Compute a reverse dictionary                
#kindorder_to_type = dict(zip( type_to_kindorder.values(), type_to_kindorder.keys()))
kindorder_to_type = { (1,1):bool, 
                (2,4):numpy.uint32, (2,8):numpy.uint64,
                (3,4):numpy.int32, (3,8):numpy.int64, 
                (4,4):numpy.float32, (4,8):numpy.float64,
                (5,4):numpy.complex64, (5,8):numpy.complex128, 
                (6,1):str }
#print( kindorder_to_type )

#type_to_kind = {bool: 'bool', int_: 'int32', long_: 'int64', float: 'float32',
#                numpy.float32: 'float32', numpy.float64: 'float64',
#                double: 'float64', complex: 'complex128', numpy.complex64: 'complex64',
#                bytes: 'bytes'}
#
#kindorder_to_type = {'bool': bool, 'int': int_, 'long': long_, 'float': float, 'float64': numpy.float64,
#                'double': double, 'complex': complex, 'complex64' : complex64,
#                'bytes': bytes}
                
# RAM: IN SJP's branch, this was letting complex64 cast to double, which is not good behaviour either.
# kind_rank = ['bool', 'int32', 'int64', 'float32', 'float64', 'complex64', 'complex128', 'none']

# kind_rank = { 'b':1, 'u':2, 'i':3, 'f':4, 'c':5 }

scalar_constant_types = tuple( type_to_kindorder.keys() )


from numexpr3 import interpreter


class Expression(object):
    def __init__(self):
        object.__init__(self)

    def __getattr__(self, name):
        if name.startswith('_'):
            return self.__dict__[name]
        else:
            return VariableNode(name, defaultType)


E = Expression()


class Context(threading.local):
    initialized = False

    def __init__(self, dict_):
        if self.initialized:
            raise SystemError('__init__ called too many times')
        self.initialized = True
        self.__dict__.update(dict_)

    def get(self, value, default):
        return self.__dict__.get(value, default)

    def get_current_context(self):
        return self.__dict__

    def set_new_context(self, dict_):
        self.__dict__.update(dict_)

# This will be called each time the local object is used in a separate thread
_context = Context({})


def get_optimization():
    return _context.get('optimization', 'none')


# helper functions for creating __magic__ methods
def ophelper(f):
    def func(*args):
        args = list(args)
        for i, x in enumerate(args):
            if isConstant(x):
                args[i] = x = ConstantNode(x)
            if not isinstance(x, ExpressionNode):
                raise TypeError("unsupported object type: %s" % type(x))
        return f(*args)

    func.__name__ = f.__name__
    func.__doc__ = f.__doc__
    func.__dict__.update(f.__dict__)
    return func


def allConstantNodes(args):
    "returns True if args are all ConstantNodes."
    for x in args:
        if not isinstance(x, ConstantNode):
            return False
    return True


def isConstant(ex):
    "Returns True if ex is a constant scalar of an allowed type."
    return isinstance(ex, scalar_constant_types)

    
# RAM: new casting tool uses types rather than kinds, and sorts by kind/order
def commonCastType(nodes):
    
    for node in nodes:
        print( "Cast node = " + str(node) )
        
    node_types = [node.astType for node in nodes]
    
    # print( "cast node_types = " + str(node_types) )
    str_count = node_types.count(bytes) + node_types.count(str) + node_types.count( numpy.string_ )
    if 0 < str_count < len(node_types):  # some args are strings, but not all
        raise TypeError("strings can only be operated with strings")
    if str_count > 0:  # if there are some, all of them must be
        return str
    
    n = (0,0) # tuple of form (kind,order)
    for x in node_types:
        print( "Cast check x = " + str(x) )
        tup = type_to_kindorder[x]
        n = (max(n[0], tup[0]), max(n[1],tup[1]))
    return kindorder_to_type[n]


max_int32 = 2147483647
min_int32 = -max_int32 - 1

# RAM: do we really want this?  

def getBestConstantType(x):
    print( "RAM: TODO, refactor getBestConstantType" )
    #print( "best constant search for :" + str(x.__class__) )
    # ``numpy.string_`` is a subclass of ``bytes``
    if isinstance(x, (bytes, str)):
        return bytes
    # Numeric conversion to boolean values is not tried because
    # ``bool(1) == True`` (same for 0 and False), so 0 and 1 would be
    # interpreted as booleans when ``False`` and ``True`` are already
    # supported.
    if isinstance(x, (bool, numpy.bool)):
        return bool
    # ``long`` objects are kept as is to allow the user to force
    # promotion of results by using long constants, e.g. by operating
    # a 32-bit array with a long (64-bit) constant.
    if isinstance(x, numpy.int64):
        #print( "best constant = int64" )
        return numpy.int64
    if isinstance(x, numpy.int32):
        #print( "best constant = int32" )
        return numpy.int32
    # ``double`` objects are kept as is to allow the user to force
    # promotion of results by using double constants, e.g. by operating
    # a float (32-bit) array with a double (64-bit) constant.
    if isinstance(x, (double, numpy.float64)):
        return numpy.float64
    if isinstance(x, numpy.float32):
        return numpy.float32
    if isinstance(x, (long, int, numpy.integer)):
        # Constants needing more than 32 bits are always
        # considered ``long``, *regardless of the platform*, so we
        # can clearly tell 32- and 64-bit constants apart.
        if not (min_int32 <= x <= max_int32):
            #     print( "best constant = int64 2" )
            print( "TODO: integer behaviour in Python is funky, long casting to int" )
            return numpy.int64
        return numpy.int32
        # print( "best constant = int32 2" )
        # return numpy.int32
    if isinstance(x, numpy.complex64):
        return numpy.complex64
    if isinstance(x, numpy.complex128):
        return numpy.complex128
    # The duality of float and double in Python avoids that we have to list
    # ``double`` too.
    for converter in float, complex:
        try:
            y = converter(x)
        except StandardError:
            continue
        if y == x:
            return converter


#def getKind(x):
#    converter = bestConstantType(x)
#    return type_to_kind[converter]



def binop(opname, reversed=False, castType=None):
    # Getting the named method from self (after reversal) does not
    # always work (e.g. int constants do not have a __lt__ method).
    opfunc = getattr(operator, "__%s__" % opname)

    @ophelper
    def operation(self, other):
        if reversed:
            self, other = other, self
        if allConstantNodes([self, other]):
            return ConstantNode(opfunc(self.value, other.value))
        else:
            return OpNode(opname, (self, other), castType=castType)

    return operation


def func(func, mintype=None, maxtype=None):
    if bool(mintype): 
        minKindOrder = type_to_kindorder[ mintype ]
    else:
        minKindOrder = (0,0)
    if bool(maxtype): 
        maxKindOrder = type_to_kindorder[ maxtype ]
    else:
        maxKindOrder = (255,255)
    
    @ophelper
    def function(*args):
        if allConstantNodes(args):
            return ConstantNode(func(*[x.value for x in args]))
        
        #print( "-------------- args =" + str(args) )
        castType = commonCastType(args)
        #print( "############ Initial cast type = " + str(castType) )
        kindCheck = type_to_kindorder[ castType ]
        # Upcast to float64 for bool, unsigned int, and int
        # Exception for following NumPy casting rules
        #FIXME: this is not always desirable. The following
        # functions which return ints (for int inputs) on numpy
        # but not on numexpr: copy, abs, fmod, ones_like
        if kindCheck[0] <= 3:
            # Cast to float with same byte number
            print( "castType = " + str(castType) )
            print( "kindCheck = " + str(kindCheck) )
            castType = kindorder_to_type[ (4,kindCheck[1]) ] 
        else:
            # Apply regular casting rules
            if bool(minKindOrder) and (minKindOrder[0] > kindCheck[0] or  minKindOrder[1] > kindCheck[1]):
                print( "Changing " + str(castType) + " to " + str(mintype) )
                castType = mintype
            if bool(maxKindOrder) and (maxKindOrder[0] < kindCheck[0] or  maxKindOrder[1] < kindCheck[1]):
                print( "Changing " + str(castType) + " to " + str(maxtype) )
                castType = maxtype
                
        return FuncNode(func.__name__, args, castType)

    return function


@ophelper
def where_func(a, b, c):
    if isinstance(a, ConstantNode):
        #FIXME: This prevents where(True, a, b)
        raise ValueError("too many dimensions")
    if allConstantNodes([a, b, c]):
        return ConstantNode(numpy.where(a, b, c))
    return FuncNode('where', [a, b, c])


def encode_axis(axis):
    if isinstance(axis, ConstantNode):
        axis = axis.value
    if axis is None:
        axis = interpreter.allaxes
    else:
        if axis < 0:
            raise ValueError("negative axis are not supported")
        if axis > 254:
            raise ValueError("cannot encode axis")
    return RawNode(axis)


def sum_func(a, axis=None):
    axis = encode_axis(axis)
    if isinstance(a, ConstantNode):
        return a
    if isinstance(a, (bool, int_, long_, float, double, complex)):
        a = ConstantNode(a)
    return FuncNode('sum', [a, axis], castType=a.astType)


def prod_func(a, axis=None):
    axis = encode_axis(axis)
    if isinstance(a, (bool, int_, long_, float, double, complex)):
        a = ConstantNode(a)
    if isinstance(a, ConstantNode):
        return a
    return FuncNode('prod', [a, axis], castType=a.astType)


@ophelper
def contains_func(a, b):
    return FuncNode('contains', [a, b], castType=bool)


@ophelper
def div_op(a, b):
    print( get_optimization() )
    
#    if get_optimization() in ('moderate', 'aggressive'):
#        print( a.astType )
#        print( b.astType )
#        print( type_to_kindorder[a.astType]  )
#        # RAM: Mmm... really if we divide by an int we should have future_division...
#        # But b.value can be a string.... this clearly was not working as intended before the refactor.
#        if (isinstance(b, ConstantNode) and
#                (type_to_kindorder[a.astType] == type_to_kindorder[b.astType]) or
#                    type_to_kindorder[a.astType][0] >= 4 ):
#            if isinstance(b.value, str):               
#                return OpNode('mul', [a, ConstantNode(1.0 / eval(b.value))])
#            else:
#                return OpNode('mul', [a, ConstantNode(1.0 / b.value)])
#                
    print( "UNoptimized divide"  )
    return OpNode('div', [a, b])


@ophelper
def truediv_op(a, b):
    if get_optimization() in ('moderate', 'aggressive'):
        if (isinstance(b, ConstantNode) and
                (type_to_kindorder[a.astType] == type_to_kindorder[b.astType]) and 
                    type_to_kindorder[a.astType][0] >= 4 ):
            return OpNode('mul', [a, ConstantNode(1.0 / b.value)])
    castType = commonCastType([a, b])
    if type_to_kindorder[castType][0] in ( 1,2,3 ):
        castType = kindorder_to_type[ (4,castType[1]) ] 
        
    return OpNode('div', [a, b], castType=castType )


@ophelper
def rtruediv_op(a, b):
    return truediv_op(b, a)


@ophelper
def pow_op(a, b):
    if allConstantNodes([a, b]):
        return ConstantNode(a ** b)
    if isinstance(b, ConstantNode):
        x = b.value
        if get_optimization() == 'aggressive':
            RANGE = 50  # Approximate break even point with pow(x,y)
            # Optimize all integral and half integral powers in [-RANGE, RANGE]
            # Note: for complex numbers RANGE could be larger.
            if (int(2 * x) == 2 * x) and (-RANGE <= abs(x) <= RANGE):
                n = int_(abs(x))
                ishalfpower = int_(abs(2 * x)) % 2

                def multiply(x, y):
                    if x is None: return y
                    return OpNode('mul', [x, y])

                r = None
                p = a
                mask = 1
                while True:
                    if (n & mask):
                        r = multiply(r, p)
                    mask <<= 1
                    if mask > n:
                        break
                    p = OpNode('mul', [p, p])
                if ishalfpower:
                    castType = commonCastType([a])
                    if type_to_kindorder[castType][0] <= 3 :
                        castType = kindorder_to_type[ (4,castType[1]) ] 
                        
                    # RAM: typo here
                    r = multiply(r, OpNode('sqrt', [a], castType))
                if r is None:
                    r = OpNode('ones_like', [a])
                if x < 0:
                    r = OpNode('div', [ConstantNode(1), r])
                return r
        if get_optimization() in ('moderate', 'aggressive'):
            if x == -1:
                return OpNode('div', [ConstantNode(1), a])
            if x == 0:
                return OpNode('ones_like', [a])
            if x == 0.5:
                castType = a.astType
                if type_to_kindorder[castType][0] <= 3 : castType = double
                return FuncNode('sqrt', [a], castType=castType)
            if x == 1:
                return a
            if x == 2:
                return OpNode('mul', [a, a])
    return OpNode('pow', [a, b])

# The functions and the minimum and maximum types accepted
functions = {
    'copy': func(numpy.copy),
    'ones_like': func(numpy.ones_like),
    'sqrt': func(numpy.sqrt, numpy.float32 ),

    'sin': func(numpy.sin, numpy.float32 ),
    'cos': func(numpy.cos, numpy.float32 ),
    'tan': func(numpy.tan, numpy.float32 ),
    'arcsin': func(numpy.arcsin, numpy.float32 ),
    'arccos': func(numpy.arccos, numpy.float32 ),
    'arctan': func(numpy.arctan, numpy.float32 ),

    'sinh': func(numpy.sinh, numpy.float32 ),
    'cosh': func(numpy.cosh, numpy.float32 ),
    'tanh': func(numpy.tanh, numpy.float32 ),
    'arcsinh': func(numpy.arcsinh, numpy.float32 ),
    'arccosh': func(numpy.arccosh, numpy.float32 ),
    'arctanh': func(numpy.arctanh, numpy.float32 ),

    'fmod': func(numpy.fmod, numpy.float32 ),
    'arctan2': func(numpy.arctan2, numpy.float32 ),

    'log': func(numpy.log, numpy.float32 ),
    'log1p': func(numpy.log1p, numpy.float32 ),
    'log10': func(numpy.log10, numpy.float32 ),
    'exp': func(numpy.exp, numpy.float32 ),
    'expm1': func(numpy.expm1, numpy.float32 ),

    'abs': func(numpy.absolute, numpy.float32 ),

    'where': where_func,
    
    # RAM: Casting here should be fixed now
    'real': func(numpy.real, numpy.float32, numpy.float32),
    'imag': func(numpy.imag, numpy.float32, numpy.float32),
    'complex': func(numpy.complex64, numpy.complex64),
    'conj': func(numpy.conj, numpy.complex64),

    'sum': sum_func,
    'prod': prod_func,
    'contains': contains_func,
}


class ExpressionNode(object):
    """An object that represents a generic number object.

    This implements the number special methods so that we can keep
    track of how this object has been used.
    """
    astNode = 'generic'

    def __init__(self, value=None, castType=None, children=None):
        object.__init__(self)
        self.value = value
        if castType is None:
            castType = 'none'
        self.astType = castType
        if children is None:
            self.children = ()
        else:
            self.children = tuple(children)

    def get_real(self):
        if self.astNode == 'constant':
            return ConstantNode(complex(self.value).real)
        return OpNode('real', (self,), double )

    real = property(get_real)

    def get_imag(self):
        if self.astNode == 'constant':
            return ConstantNode(complex(self.value).imag)
        return OpNode('imag', (self,), double )

    imag = property(get_imag)

    def __str__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__, self.value,
                                   self.astType, self.children)

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return OpNode('neg', (self,))

    def __invert__(self):
        return OpNode('invert', (self,))

    def __pos__(self):
        return self

    # The next check is commented out. See #24 for more info.

    def __nonzero__(self):
        raise TypeError("You can't use Python's standard boolean operators in "
                        "NumExpr expressions. You should use their bitwise "
                        "counterparts instead: '&' instead of 'and', "
                        "'|' instead of 'or', and '~' instead of 'not'.")

    __add__ = __radd__ = binop('add')
    __sub__ = binop('sub')
    __rsub__ = binop('sub', reversed=True)
    __mul__ = __rmul__ = binop('mul')
    if sys.version_info[0] < 3:
        __div__ = div_op
        __rdiv__ = binop('div', reversed=True)
    __truediv__ = truediv_op
    __rtruediv__ = rtruediv_op
    __pow__ = pow_op
    __rpow__ = binop('pow', reversed=True)
    __mod__ = binop('mod')
    __rmod__ = binop('mod', reversed=True)

    __lshift__ = binop('lshift')
    __rlshift__ = binop('lshift', reversed=True)
    __rshift__ = binop('rshift')
    __rrshift__ = binop('rshift', reversed=True)

    # boolean operations

    __and__ = binop('and', castType=bool )
    __or__ = binop('or', castType=bool)

    __gt__ = binop('gt', castType=bool)
    __ge__ = binop('ge', castType=bool)
    __eq__ = binop('eq', castType=bool)
    __ne__ = binop('ne', castType=bool)
    __lt__ = binop('gt', reversed=True, castType=bool)
    __le__ = binop('ge', reversed=True, castType=bool)


class LeafNode(ExpressionNode):
    leafNode = True


class VariableNode(LeafNode):
    astNode = 'variable'

    def __init__(self, value=None, castType=None, children=None):
        LeafNode.__init__(self, value=value, castType=castType)


class RawNode(object):
    """Used to pass raw integers to interpreter.
    For instance, for selecting what function to use in func1.
    Purposely don't inherit from ExpressionNode, since we don't wan't
    this to be used for anything but being walked.
    """
    astNode = 'raw'
    astType = None

    def __init__(self, value):
        self.value = value
        self.children = ()

    def __str__(self):
        return 'RawNode(%s)' % (self.value,)

    __repr__ = __str__


class ConstantNode(LeafNode):
    astNode = 'constant'

    def __init__(self, value=None, children=None):
        # TODO: This isn't working as intended
        #print( "Original value: " + str( type(value)) )
        bestType = getBestConstantType( value )
        #print( "Best type: " + str( bestType ) )
        LeafNode.__init__(self, value=value, castType = bestType )

    def __neg__(self):
        return ConstantNode(-self.value)

    def __invert__(self):
        return ConstantNode(~self.value)


class OpNode(ExpressionNode):
    astNode = 'op'

    def __init__(self, opcode=None, args=None, castType=None):
        if (castType is None) and (args is not None):
            castType = commonCastType(args)
        ExpressionNode.__init__(self, value=opcode, castType=castType, children=args)


class FuncNode(OpNode):
    def __init__(self, opcode=None, args=None, castType=None):
        if (castType is None) and (args is not None):
            castType = commonCastType(args)
        OpNode.__init__(self, opcode, args, castType)
