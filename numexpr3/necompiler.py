###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import __future__
import sys
import numpy

from numexpr3 import interpreter, expressions, use_vml
from numexpr3.utils import CacheDict

# Declare a double type that does not exist in Python space
#double = numpy.double
#complex64 = numpy.complex64
if sys.version_info[0] < 3:
    int_ = int
    long_ = long
else:
    int_ = numpy.int32
    long_ = numpy.int64


# RAM: refactor
"""
typecode_to_kind = {'b': 'bool', 'i': 'int', 'l': 'long', 'f': 'float',
                    'd': 'double', 'c': 'complex', 'x' : 'complex64',
                    's': 'bytes', 'n': 'none'}
kind_to_typecode = {'bool': 'b', 'int': 'i', 'long': 'l', 'float': 'f',
                    'double': 'd', 'complex': 'c', 'complex64' : 'x',
                    'bytes': 's', 'none': 'n'}
type_to_typecode = {bool: 'b', int_: 'i', long_: 'l', float: 'f',
                    double: 'd', complex: 'c', complex64: 'x', bytes: 's'}
"""
                    
#typecode_to_kind = {'b1': 'bool', 'i1': 'int8', 'i2': 'int16', 'i4': 'int32', 'i8': 'int64',
#                    'u1': 'uint8', 'u2': 'uint16', 'u4':'uint32', 'u8': 'uint64',
#                    'f4': 'float32', 'f8': 'float64', 
#                    'c8': 'complex64', 'c16': 'complex128',
#                    's1': 'str', 'n0': 'none'}
#                    
## Kind to typecode and type to typecode can have more rules that the reverse dict               
#kind_to_typecode = {'bool': 'b1',  'int8': 'i1', 'int16':'i2', 'int32':'i4', 'int64':'i8', 
#                    'int': 'i8',
#                    'uint8': 'u1', 'uint16':'u2', 'uint32':'u4', 'uint64':'u8', 
#                    'float32': 'f4', 'float64':'f8',
#                    'float': 'f8', 'double':'f8',
#                    'complex64':'c8', 'complex128':'c16',
#                    'none':'n0', 'str': 's1' }
#                    
# RAM: convert Python native types like long, double, explicitely to Numpy kind/order format    
#type_to_typecode = {bool: 'b1', int_: 'i', long_: 'l', float: 'f',
#                    double: 'd', complex: 'c', complex64: 'x', bytes: 's'}    
                    
type_to_typecode = { numpy.bool: 'b1', bool: 'b1', 
                    numpy.int8: 'i1', numpy.int16: 'i2', numpy.int32: 'i4', numpy.int64: 'i8', 
                    int: 'i8', long: 'i8',
                    numpy.uint8: 'u1', numpy.uint16: 'u2', numpy.uint32: 'u4', numpy.uint64: 'u8',
                    numpy.float32: 'f4', numpy.float64: 'f8', 
                    float: 'f8',
                    numpy.complex64: 'c8', numpy.complex128: 'c16', 
                    bytes: 's1', str:'s1', numpy.string_: 's1', None: 'n0' }                    


typecode_to_type = { 'b1': numpy.bool,
                    'i1': numpy.int8, 'i2':numpy.int16, 'i4':numpy.int32, 'i8':numpy.int64, 
                    'u1':numpy.uint8, 'u2':numpy.uint16, 'u4':numpy.uint32, 'u8':numpy.uint64,
                    'f4':numpy.float32, 'f8': numpy.float64, 
                    'c8':numpy.complex64, 'c16':numpy.complex128, 
                    's1':str, 'n0':None }       
                    
type_to_kindorder = expressions.type_to_kindorder
kindorder_to_type = expressions.kindorder_to_type
defaultType = expressions.defaultType

scalar_constant_kinds = expressions.scalar_constant_types


class ASTNode(object):
    """Abstract Syntax Tree node.

    Members:

    astNode      -- type of node (op, constant, variable, raw, or alias)
    astType      -- the type of the result (bool, float, etc.)
    value        -- value associated with this node.
                    An opcode, numerical value, a variable name, etc.
    children     -- the children below this node
    reg          -- the register assigned to the result for this node.
    """
    cmpnames = ['astNode', 'astType', 'value', 'children']

    def __init__(self, astNode='generic', astType='unknown',
                 value=None, children=()):
        object.__init__(self)
        self.astNode = astNode
        self.astType = astType
        self.value = value
        self.children = tuple(children)
        self.reg = None

    def __eq__(self, other):
        if self.astNode == 'alias':
            self = self.value
        if other.astNode == 'alias':
            other = other.value
        if not isinstance(other, ASTNode):
            return False
        for name in self.cmpnames:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        if self.astNode == 'alias':
            self = self.value
        return hash((self.astNode, self.astType, self.value, self.children))

    def __str__(self):
        return 'AST(%s, %s, %s, %s, %s)' % (self.astNode, self.astType, 
                                            self.value, self.children, self.reg)

    def __repr__(self):
        return '<AST object at %s>' % id(self)

    def key(self):
        return (self.astNode, self.astType, self.value, self.children)

    def typecode(self):
        return type_to_typecode[self.astType]

    def postorderWalk(self):
        for c in self.children:
            for w in c.postorderWalk():
                yield w
        yield self

    def allOf(self, *astNodes):
        astNodes = set(astNodes)
        for w in self.postorderWalk():
            if w.astNode in astNodes:
                yield w


def expressionToAST(ex):
    """Take an expression tree made out of expressions.ExpressionNode,
    and convert to an AST tree.

    This is necessary as ExpressionNode overrides many methods to act
    like a number.
    """
    return ASTNode(ex.astNode, ex.astType, ex.value,
                   [expressionToAST(c) for c in ex.children])


def buildTypecodeList( in_string ):
    # Cheap split operation on sigs without importing re library is to make the string a list and cat numbers
    in_list = list( in_string )
    for J in range(len(in_list)-1, -1, -1): # Go backward so we can pop safely
        if in_list[J] in '248016379':
            if in_list[J-1] in '248016379': in_list[J-1] = in_list[J-1] + in_list.pop(J)
                
    for J in range(len(in_list)-1,-1,-2): in_list[J-1] = in_list[J-1] + in_list.pop(J)
    return  in_list
    
def findBestSigMatch( code_base, op_string ):
    """
    RAM: This is to replace sigPerms with something a little more deterministic given the new, more complicated 
    casting rules, to search efficiently for a valid function signature, when we need to cast some value.
    
    Although this looks more complicated, it's not recursive, and it's cutting down the problem complexity 
    often.
    
    Returns [sig, funcReturn]
    """  
    op_list = numpy.array( buildTypecodeList(op_string) )
    
    # Reduce the problem dimension by decimating opcodes + funccodes
    # This could be made faster by precomputing an ordered list of the keys at load
    lenOpcodes = len( interpreter.opcodes.keys() )
    code_ids = interpreter.opcodes.keys() + interpreter.funccodes.keys()
    code_indices = [] # So we can decide if each match is an opcode or function.
    for J in range( len(code_ids)-1, -1, -1):
        if not code_ids[J].startswith( code_base ):
            code_ids.remove( code_ids[J] )
        else:
            code_ids[J] = code_ids[J].split('_')[1]
            code_indices.append(J)
            
    for J, identifier in enumerate(code_ids):
        code_ids[J] = numpy.array( buildTypecodeList( identifier ) )
     
    matches = numpy.zeros( len(code_ids), dtype='int' )   
    for J, code in enumerate(code_ids):
        matches[J] = numpy.sum( code == op_list )
    
    try:
        bestMatchIndex = numpy.argmax( matches )
        return code_ids[bestMatchIndex], bool( code_indices[bestMatchIndex] >= lenOpcodes )
    except:
        raise NotImplementedError( "couldn't find matching opcode for '%s'"%(code_base + '_' + op_string))


#def sigPerms(s):
#    """Generate all possible signatures derived by upcasting the given
#    signature.
#    
#    RAM: Now signatures can be C16, or F8 or F16, etc. so we need to do a bit of 
#    string parsing to seperate the different ones.
#    """
#    # TODO: this needs a complete re-write
#    kinds = 'buifc'
#    # This is strange recursive thing...
#    print( "Upcast signature : " + str(s) )
#    print( s.split)
#    
#    if not s:
#        yield ''
#    elif s[0] in kinds:
#        start = kinds.index(s[0])
#        for x in kinds[start:]:
#            for y in sigPerms(s[1:]):
#                yield x + y
#    elif s[0] == 's':  # numbers shall not be cast to strings
#        for y in sigPerms(s[1:]):
#            yield 's' + y
#    else:
#        yield s



def typeCompileAst(ast):
    """Assign appropiate types to each node in the AST.

    Will convert opcodes and functions to appropiate upcast version,
    and add "cast" ops if needed.
    """

    children = list(ast.children)
    if ast.astNode == 'op':
        retsig = ast.typecode()
        basesig = ''.join(x.typecode() for x in list(ast.children))
        
        # print( "DEBUG: exec %s"%( ast.value + '_' + retsig + basesig) )
        # Somewhere we missed a casting operation...
        
        # Find some operation that will work on an acceptable casting of args.
#        for sig in sigPerms(basesig):
#            value = (ast.value + '_' + retsig + sig).encode('ascii')
#            if value in interpreter.opcodes:
#                break
#        else:
#        What on earth does this code block do? It search opcode, then funccodes...
#        but the function needs some new childern nodes apparently
#            for sig in sigPerms(basesig):
#                funcname = (ast.value + '_' + retsig + sig).encode('ascii')
#                if funcname in interpreter.funccodes:
#                    value = ('func_%sn' % (retsig + sig)).encode('ascii')
#                    children += [ASTNode('raw', 'none',
#                                         interpreter.funccodes[funcname])]
#                    break
#            else:
#                raise NotImplementedError( "couldn't find matching opcode for '%s'"%(ast.value + '_' + retsig + basesig))
        
        sig, returnFunc = findBestSigMatch( ast.value, retsig + basesig )
        joined_sig = "".join(sig)
        if returnFunc:
            # print( "Found function in ast.value" )
            funcname = (ast.value + '_' + joined_sig ).encode('ascii')
            value = ('func_%sn0'%joined_sig).encode('ascii') # RAM: no-op is now n0
            children += [ASTNode('raw', 'none', interpreter.funccodes[funcname])]
        else: # It's an opcode, not a function
            value = (ast.value + '_' + joined_sig ).encode('ascii')

        # print( sig )
        # print( basesig )
        # RAM: Compare the two lists of typecodes element-by-element to determine casting op requirements
        for i, (have, want) in enumerate(zip( buildTypecodeList( basesig ) , sig[1:] )):
            if have != want:
                # print( "i: %d"%i )
                castType = typecode_to_type[want]
                # print( "CASTING: have " + have + " to want " + want + " with type " + str(castType) )
                if children[i].astNode == 'constant':
                    children[i] = ASTNode('constant', castType, children[i].value)
                else:
                    children[i] = ASTNode('op', castType, "cast", [children[i]])
    else:
        value = ast.value
        children = ast.children
    # print( "I have %d"%len(children) + " childern" )
    return ASTNode(ast.astNode, ast.astType, value,
                   [typeCompileAst(c) for c in children])


class Register(object):
    """Abstraction for a register in the VM.

    Members:
    node          -- the AST node this corresponds to
    temporary     -- True if this isn't an input or output
    immediate     -- not a register, but an immediate value
    n             -- the physical register number.
                     None if no number assigned yet.
    """

    def __init__(self, astnode, temporary=False):
        self.node = astnode
        self.temporary = temporary
        self.immediate = False
        self.n = None

    def __str__(self):
        if self.temporary:
            name = 'Temporary'
        else:
            name = 'Register'
        return '%s(%s, %s, %s)' % (name, self.node.astNode,
                                   self.node.astType, self.n,)

    def __repr__(self):
        return self.__str__()


class Immediate(Register):
    """Representation of an immediate (integer) operand, instead of
    a register.
    """

    def __init__(self, astnode):
        Register.__init__(self, astnode)
        self.immediate = True

    def __str__(self):
        return 'Immediate(%d)' % (self.node.value,)


def stringToExpression(s, types, context):
    """Given a string, convert it to a tree of ExpressionNode's.
    """
    old_ctx = expressions._context.get_current_context()
    try:
        expressions._context.set_new_context(context)
        # first compile to a code object to determine the names
        if context.get('truediv', False):
            flags = __future__.division.compiler_flag
        else:
            flags = 0
        c = compile(s, '<expr>', 'eval', flags)
        # make VariableNode's for the names
        names = {}
        for name in c.co_names:
            if name == "None":
                names[name] = None
            elif name == "True":
                names[name] = True
            elif name == "False":
                names[name] = False
            else:
                t = types.get(name, defaultType)
                names[name] = expressions.VariableNode(name, t)
        names.update(expressions.functions)
        # now build the expression
        ex = eval(c, names)
        if expressions.isConstant(ex):
            ex = expressions.ConstantNode(ex, expressions.getKind(ex))
        elif not isinstance(ex, expressions.ExpressionNode):
            raise TypeError("unsupported expression type: %s" % type(ex))
    finally:
        expressions._context.set_new_context(old_ctx)
    return ex


def isReduction(ast):
    return ast.value.startswith(b'sum_') or ast.value.startswith(b'prod_')


def getInputOrder(ast, input_order=None):
    """Derive the input order of the variables in an expression.
    """
    variables = {}
    for a in ast.allOf('variable'):
        variables[a.value] = a
    variable_names = set(variables.keys())

    if input_order:
        if variable_names != set(input_order):
            raise ValueError(
                "input names (%s) don't match those found in expression (%s)"
                % (input_order, variable_names))

        ordered_names = input_order
    else:
        ordered_names = list(variable_names)
        ordered_names.sort()
    ordered_variables = [variables[v] for v in ordered_names]
    return ordered_variables


#def castConstant(x, castType):
#    # Exception for 'float' types that will return the NumPy float32 type
#    #if kind == 'float':
#    #    return numpy.float32(x)
#    return castType(x)


def getConstants(ast):
    const_map = {}
    for a in ast.allOf('constant'):
        const_map[(a.astType, a.value)] = a
    ordered_constants = const_map.keys()
    ordered_constants.sort()
    constants_order = [const_map[v] for v in ordered_constants]
    constants = [a.astType(a.value)
                 for a in constants_order]
    return constants_order, constants


def sortNodesByOrder(nodes, order):
    order_map = {}
    for i, (_, v, _) in enumerate(order):
        order_map[v] = i
    dec_nodes = [(order_map[n.value], n) for n in nodes]
    dec_nodes.sort()
    return [a[1] for a in dec_nodes]


def assignLeafRegisters(inodes, registerMaker):
    """Assign new registers to each of the leaf nodes.
    """
    leafRegisters = {}
    for node in inodes:
        key = node.key()
        if key in leafRegisters:
            node.reg = leafRegisters[key]
        else:
            node.reg = leafRegisters[key] = registerMaker(node)


def assignBranchRegisters(inodes, registerMaker):
    """Assign temporary registers to each of the branch nodes.
    """
    for node in inodes:
        node.reg = registerMaker(node, temporary=True)


def collapseDuplicateSubtrees(ast):
    """Common subexpression elimination.
    """
    seen = {}
    aliases = []
    for a in ast.allOf('op'):
        if a in seen:
            target = seen[a]
            a.astNode = 'alias'
            a.value = target
            a.children = ()
            aliases.append(a)
        else:
            seen[a] = a
    # Set values and registers so optimizeTemporariesAllocation
    # doesn't get confused
    for a in aliases:
        while a.value.astNode == 'alias':
            a.value = a.value.value
    return aliases


def optimizeTemporariesAllocation(ast):
    """Attempt to minimize the number of temporaries needed, by
    reusing old ones.
    """
    nodes = [n for n in ast.postorderWalk() if n.reg.temporary]
    users_of = dict((n.reg, set()) for n in nodes)

#    node_regs = dict((n, set(c.reg for c in n.children if c.reg.temporary))
#                     for n in nodes)
    if nodes and nodes[-1] is not ast:
        nodes_to_check = nodes + [ast]
    else:
        nodes_to_check = nodes
    for n in nodes_to_check:
        for c in n.children:
            if c.reg.temporary:
                users_of[c.reg].add(n)

    unused = dict([(tc, set()) for tc in scalar_constant_kinds])
    for n in nodes:
        for c in n.children:
            reg = c.reg
            if reg.temporary:
                users = users_of[reg]
                users.discard(n)
                if not users:
                    unused[reg.node.astType].add(reg)
        if unused[n.astType]:
            reg = unused[n.astType].pop()
            users_of[reg] = users_of[n.reg]
            n.reg = reg


def setOrderedRegisterNumbers(order, start):
    """Given an order of nodes, assign register numbers.
    """
    for i, node in enumerate(order):
        node.reg.n = start + i
    return start + len(order)


def setRegisterNumbersForTemporaries(ast, start):
    """Assign register numbers for temporary registers, keeping track of
    aliases and handling immediate operands.
    """
    seen = 0
    signature = ''
    aliases = []
    for node in ast.postorderWalk():
        if node.astNode == 'alias':
            aliases.append(node)
            node = node.value
        if node.reg.immediate:
            node.reg.n = node.value
            continue
        reg = node.reg
        if reg.n is None:
            reg.n = start + seen
            seen += 1
            signature += reg.node.typecode()
    for node in aliases:
        node.reg = node.value.reg
    return start + seen, signature


def convertASTtoThreeAddrForm(ast):
    """Convert an AST to a three address form.

    Three address form is (op, reg1, reg2, reg3), where reg1 is the
    destination of the result of the instruction.

    I suppose this should be called three register form, but three
    address form is found in compiler theory.
    """
    return [(node.value, node.reg) + tuple([c.reg for c in node.children])
            for node in ast.allOf('op')]


def compileThreeAddrForm(program):
    """Given a three address form of the program, compile it a string that
    the VM understands.
    """

#    def nToChr(reg):
#        if reg is None:
#            return b'\xff'
#        elif reg.n < 0:
#            raise ValueError("negative value for register number %s" % reg.n)
#        else:
#            if sys.version_info[0] < 3:
#                return chr(reg.n)
#            else:
#                # int.to_bytes is not available in Python < 3.2
#                #return reg.n.to_bytes(1, sys.byteorder)
#                return bytes([reg.n])
                
        
    def nTo4Byte(reg):
        if reg is None:
            return bytearray( [255,255,255,255] )
        elif reg.n < 0:
            raise ValueError("negative value for register number %s" % reg.n)
        else:
            # THIS SHOULD BE 4 BYTES
            return bytearray( [(reg.n & (0xff << pos*8)) >> pos*8 for pos in xrange(4)] )

    def quadrupleToBytearray(opcode, store, a1=None, a2=None):
        # RAM: must make this threeAddrForm 16 bytes [4 x 4] long to expand opcode space
        cop = bytearray( [(interpreter.opcodes[opcode] & (0xff << pos*8)) >> pos*8 for pos in xrange(4)] )
        cs = nTo4Byte(store)
        ca1 = nTo4Byte(a1)
        ca2 = nTo4Byte(a2)

        print( "DEBUG: Quad op "+str(opcode)+ ": " + str(cop).encode('hex') + ", " + str(cs).encode('hex') + ", " 
            + str(ca1).encode('hex') + ", " + str(ca2).encode('hex') )
        return cop + cs + ca1 + ca2

    def toBytearray(args):
        while len(args) < 4:
            args += (None,)
        opcode, store, a1, a2 = args[:4]
        operations = quadrupleToBytearray(opcode, store, a1, a2)

        args = args[4:]
        while args:
            # TODO: RAM this should be all zeros or /xFF/xFF/xFF/xFF
            nextop = quadrupleToBytearray(b'noop', *args[:3])
            operations.extend(nextop)
            args = args[3:]
        return operations

    # RAM: now using bytearray instead of string to permit null characters with fixed-word length of 4 bytes
    prog_bytes = bytearray([])
    for t in program:
        prog_bytes.extend( toBytearray(t) )
    # print( "DEBUG prog_bytes = " + prog_bytes.__str__().encode('hex') )
    return prog_bytes


context_info = [
    ('optimization', ('none', 'moderate', 'aggressive'), 'aggressive'),
    ('truediv', (False, True, 'auto'), 'auto')
]


def getContext(kwargs, frame_depth=1):
    d = kwargs.copy()
    context = {}
    for name, allowed, default in context_info:
        value = d.pop(name, default)
        if value in allowed:
            context[name] = value
        else:
            raise ValueError("'%s' must be one of %s" % (name, allowed))

    if d:
        raise ValueError("Unknown keyword argument '%s'" % d.popitem()[0])
    if context['truediv'] == 'auto':
        caller_globals = sys._getframe(frame_depth + 1).f_globals
        context['truediv'] = \
            caller_globals.get('division', None) == __future__.division

    return context


def precompile(ex, signature=(), context={}):
    """Compile the expression to an intermediate form.
    """
    types = dict(signature)
    input_order = [name for (name, type_) in signature]

    if isinstance(ex, (str, unicode)):
        ex = stringToExpression(ex, types, context)

    # the AST is like the expression, but the node objects don't have
    # any odd interpretations

    ast = expressionToAST(ex)
    
    print( "Debug: ast = " + str(ast) )

    if ex.astNode != 'op':
        ast = ASTNode('op', value='copy', astType=ex.astType, children=(ast,))

    ast = typeCompileAst(ast)

    aliases = collapseDuplicateSubtrees(ast)

    assignLeafRegisters(ast.allOf('raw'), Immediate)
    assignLeafRegisters(ast.allOf('variable', 'constant'), Register)
    assignBranchRegisters(ast.allOf('op'), Register)

    # assign registers for aliases
    for a in aliases:
        a.reg = a.value.reg

    input_order = getInputOrder(ast, input_order)
    constants_order, constants = getConstants(ast)

    if isReduction(ast):
        ast.reg.temporary = False

    optimizeTemporariesAllocation(ast)

    ast.reg.temporary = False
    r_output = 0
    ast.reg.n = 0

    r_inputs = r_output + 1
    r_constants = setOrderedRegisterNumbers(input_order, r_inputs)
    r_temps = setOrderedRegisterNumbers(constants_order, r_constants)
    r_end, tempsig = setRegisterNumbersForTemporaries(ast, r_temps)

    threeAddrProgram = convertASTtoThreeAddrForm(ast)
    input_names = tuple([a.value for a in input_order])
    signature = ''.join(type_to_typecode[types.get(x, defaultType)]
                        for x in input_names)
    return threeAddrProgram, signature, tempsig, constants, input_names


def NumExpr(ex, signature=(), **kwargs):
    """
    Compile an expression built using E.<variable> variables to a function.

    ex can also be specified as a string "2*a+3*b".

    The order of the input variables and their types can be specified using the
    signature parameter, which is a list of (name, type) pairs.

    Returns a `NumExpr` object containing the compiled function.
    """
    # NumExpr can be called either directly by the end-user, in which case
    # kwargs need to be sanitized by getContext, or by evaluate,
    # in which case kwargs are in already sanitized.
    # In that case frame_depth is wrong (it should be 2) but it doesn't matter
    # since it will not be used (because truediv='auto' has already been
    # translated to either True or False).

    context = getContext(kwargs, frame_depth=1)
    threeAddrProgram, inputsig, tempsig, constants, input_names = \
        precompile(ex, signature, context)
    # print( "threeAddrProgram = " + str(threeAddrProgram) )
    
    program = compileThreeAddrForm(threeAddrProgram)
    
#    hex_input = ":"; hex_temp = ":";
#    for x in inputsig: hex_input += x.encode('hex') + ":"
#    for x in tempsig: hex_temp += x.encode('hex') + ":"
    # for x in program:  hex_program += x.encode('hex') + ":"
    
    print( "inputsig = %s"%inputsig ) # This is all the op-codes
    print( "tempsig = %s"%tempsig )
    # print( "program = " + program.__str__().encode('hex') )
    print( "constants = " + str(constants) )
    print( "input_names = " + str(input_names) )
    
    # What if we make program a ByteArray instead? https://docs.python.org/2/c-api/bytearray.html
    # make inputsig and tempsig lists of types instead of basic strings
    # Do we want byte-arrays too?  Or can we have lists of strings?
    # I suppose it matters with the zeros...
    
    
#    inputsig = buildTypecodeList(inputsig)
#    for J, x in enumerate(inputsig):
#        print( "inputsig%d"%J + " : %s"%str(x) )
#    print( "STOP" )
    return interpreter.NumExpr( buildTypecodeList(inputsig),
                               buildTypecodeList(tempsig),
                               program, constants, input_names)


#def encodeBytecode( bytecode ):
#    """
#    RAM: this is a helper function designed to take a 1-4 character string and convert it to a bytearray 
#    suitable for passing into interpreter.cpp.  This is to replace the .encode('ascii') commands.
#    
#    Can also take an int64/long, which is converted to hex.
#    """
#    # Check if we have an int (i.e. an op-code) or a string
#    if bytecode == None:
#        return bytearray( "\0\0\0\0" )
#        
#    elif isinstance( bytecode, Register ):
#        # Still need to parse deeper!  Actually I need to call encodeBytecode much later
#        # Or maybe we should make the thing a list instead of a big string?
#        return bytecode 
#        
#    elif isinstance( bytecode, long ):
#        return bytearray( hex( bytecode ) )
#        
#    elif isinstance( bytecode, str ):
#        lbc = len(bytecode)
#        if lbc == 0:
#            return bytearray( "\0\0\0\0" )
#        elif lbc == 1:
#            return bytearray( bytecode + "\0\0\0" )
#        elif lbc == 2:
#            return bytearray( bytecode + "\0\0" )
#        elif lbc == 3:
#            return bytearray( bytecode + "\0" )
#        elif lbc == 4:
#            return bytearray( bytecode  )
#        else:
#            print( "DEBUG: invalid bytecode length: " +str(bytecode) +" of len: " + str(lbc) )
#    else:
#        print( "DEBUG: unrecognized type for bytecode encoder : " + str(bytecode) )
    

def disassemble(nex):
    """
    Given a NumExpr object, return a list which is the program disassembled.
    """
    print( "TODO: necompiler.disassemble, needs to be debugged" )
    rev_opcodes = {}
    for op in interpreter.opcodes:
        rev_opcodes[interpreter.opcodes[op]] = op
    r_constants = 1 + len(nex.signature)
    r_temps = r_constants + len(nex.constants)

    def getArg(pc, offset):
        if sys.version_info[0] < 3:
            arg = ord(nex.program[pc + offset])
            op = rev_opcodes.get(ord(nex.program[pc]))
        else:
            arg = nex.program[pc + offset]
            op = rev_opcodes.get(nex.program[pc])
        try:
            code = op.split(b'_')[1][offset - 1]
        except IndexError:
            return None
        if sys.version_info[0] > 2:
            # int.to_bytes is not available in Python < 3.2
            #code = code.to_bytes(1, sys.byteorder)
            code = bytes([code])
        if arg == 4294967295: # RAM: Max word
            return None
        if code != b'n':
            if arg == 0:
                return b'r0'
            elif arg < r_constants:
                return ('r%d[%s]' % (arg, nex.input_names[arg - 1])).encode('ascii')
            elif arg < r_temps:
                return ('c%d[%s]' % (arg, nex.constants[arg - r_constants])).encode('ascii')
            else:
                return ('t%d' % (arg,)).encode('ascii')
        else:
            return arg

    source = []
    for pc in range(0, len(nex.program), 4):
        if sys.version_info[0] < 3:
            op = rev_opcodes.get(ord(nex.program[pc]))
        else:
            op = rev_opcodes.get(nex.program[pc])
        dest = getArg(pc, 1)
        arg1 = getArg(pc, 2)
        arg2 = getArg(pc, 3)
        source.append((op, dest, arg1, arg2))
    return source


#def getType(a):
#    kind = a.dtype.kind
#    if kind == 'b':
#        return bool
#    if kind in 'iu':
#        if a.dtype.itemsize > 4:
#            return long_  # ``long`` is for integers of more than 32 bits
#        if kind == 'u' and a.dtype.itemsize == 4:
#            return long_  # use ``long`` here as an ``int`` is not enough
#        return int_
#    if kind == 'f':
#        if a.dtype.itemsize > 4:
#            return numpy.float64  # ``double`` is for floats of more than 32 bits
#        return numpy.float32
#    if kind == 'c':
#        # RAM need to distinguish between complex64 and complex128 here
#        if a.dtype.itemsize > 8:
#            return numpy.complex128
#        return numpy.complex64
#    if kind == 'S':
#        return bytes
#    raise ValueError("unknown type %s" % a.dtype.name)


def getExprNames(text, context):
    ex = stringToExpression(text, {}, context)
    ast = expressionToAST(ex)
    input_order = getInputOrder(ast, None)
    #try to figure out if vml operations are used by expression
    if not use_vml:
        ex_uses_vml = False
    else:
        for node in ast.postorderWalk():
            if node.astNode == 'op' \
                    and node.value in ['sin', 'cos', 'exp', 'log',
                                       'expm1', 'log1p',
                                       'pow', 'div',
                                       'sqrt', 'inv',
                                       'sinh', 'cosh', 'tanh',
                                       'arcsin', 'arccos', 'arctan',
                                       'arccosh', 'arcsinh', 'arctanh',
                                       'arctan2', 'abs']:
                ex_uses_vml = True
                break
        else:
            ex_uses_vml = False

    return [a.value for a in input_order], ex_uses_vml


# Dictionaries for caching variable names and compiled expressions
_names_cache = CacheDict(256)
_numexpr_cache = CacheDict(256)


def evaluate(ex, local_dict=None, global_dict=None,
             out=None, order='K', casting='safe', **kwargs):
    """Evaluate a simple array expression element-wise, using the new iterator.

    ex is a string forming an expression, like "2*a+3*b". The values for "a"
    and "b" will by default be taken from the calling function's frame
    (through use of sys._getframe()). Alternatively, they can be specifed
    using the 'local_dict' or 'global_dict' arguments.

    Parameters
    ----------

    local_dict : dictionary, optional
        A dictionary that replaces the local operands in current frame.

    global_dict : dictionary, optional
        A dictionary that replaces the global operands in current frame.

    out : NumPy array, optional
        An existing array where the outcome is going to be stored.  Care is
        required so that this array has the same shape and type than the
        actual outcome of the computation.  Useful for avoiding unnecessary
        new array allocations.

    order : {'C', 'F', 'A', or 'K'}, optional
        Controls the iteration order for operands. 'C' means C order, 'F'
        means Fortran order, 'A' means 'F' order if all the arrays are
        Fortran contiguous, 'C' order otherwise, and 'K' means as close to
        the order the array elements appear in memory as possible.  For
        efficient computations, typically 'K'eep order (the default) is
        desired.

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when making a copy or
        buffering.  Setting this to 'unsafe' is not recommended, as it can
        adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
    """
    if not isinstance(ex, (str, unicode)):
        raise ValueError("must specify expression as a string")
    # Get the names for this expression
    context = getContext(kwargs, frame_depth=1)
    expr_key = (ex, tuple(sorted(context.items())))
    if expr_key not in _names_cache:
        _names_cache[expr_key] = getExprNames(ex, context)
    names, ex_uses_vml = _names_cache[expr_key]
    # Get the arguments based on the names.
    call_frame = sys._getframe(1)
    if local_dict is None:
        local_dict = call_frame.f_locals
    if global_dict is None:
        global_dict = call_frame.f_globals

    arguments = []
    for name in names:
        try:
            a = local_dict[name]
        except KeyError:
            a = global_dict[name]
        arguments.append(numpy.asarray(a))

    # Create a signature
    signature = [(name, arg.dtype.type) for (name, arg) in zip(names, arguments)]

    # Look up numexpr if possible.
    numexpr_key = expr_key + (tuple(signature),)
    try:
        compiled_ex = _numexpr_cache[numexpr_key]
    except KeyError:
        compiled_ex = _numexpr_cache[numexpr_key] = \
            NumExpr(ex, signature, **context)
    kwargs = {'out': out, 'order': order, 'casting': casting,
              'ex_uses_vml': ex_uses_vml}
  
    result = compiled_ex(*arguments, **kwargs)
    return result
