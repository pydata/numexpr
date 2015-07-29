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

from numexpr import interpreter, expressions, use_vml
from numexpr.utils import CacheDict

# Declare a double type that does not exist in Python space
double = numpy.double
complex64 = numpy.complex64
if sys.version_info[0] < 3:
    int_ = int
    long_ = long
else:
    int_ = numpy.int32
    long_ = numpy.int64

typecode_to_kind = {'b': 'bool', 'i': 'int', 'l': 'long', 'f': 'float',
                    'd': 'double', 'c': 'complex', 'x' : 'complex64',
                    's': 'bytes', 'n': 'none'}
kind_to_typecode = {'bool': 'b', 'int': 'i', 'long': 'l', 'float': 'f',
                    'double': 'd', 'complex': 'c', 'complex64' : 'x',
                    'bytes': 's', 'none': 'n'}
type_to_typecode = {bool: 'b', int_: 'i', long_: 'l', float: 'f',
                    double: 'd', complex: 'c', complex64: 'x', bytes: 's'}
type_to_kind = expressions.type_to_kind
kind_to_type = expressions.kind_to_type
default_type = kind_to_type[expressions.default_kind]

# Final addtions for Python 3 (mainly for PyTables needs)
if sys.version_info[0] > 2:
    typecode_to_kind['s'] = 'str'
    kind_to_typecode['str'] = 's'
    type_to_typecode[str] = 's'

scalar_constant_kinds = kind_to_typecode.keys()


class ASTNode(object):
    """Abstract Syntax Tree node.

    Members:

    astType      -- type of node (op, constant, variable, raw, or alias)
    astKind      -- the type of the result (bool, float, etc.)
    value        -- value associated with this node.
                    An opcode, numerical value, a variable name, etc.
    children     -- the children below this node
    reg          -- the register assigned to the result for this node.
    """
    cmpnames = ['astType', 'astKind', 'value', 'children']

    def __init__(self, astType='generic', astKind='unknown',
                 value=None, children=()):
        object.__init__(self)
        self.astType = astType
        self.astKind = astKind
        self.value = value
        self.children = tuple(children)
        self.reg = None

    def __eq__(self, other):
        if self.astType == 'alias':
            self = self.value
        if other.astType == 'alias':
            other = other.value
        if not isinstance(other, ASTNode):
            return False
        for name in self.cmpnames:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        if self.astType == 'alias':
            self = self.value
        return hash((self.astType, self.astKind, self.value, self.children))

    def __str__(self):
        return 'AST(%s, %s, %s, %s, %s)' % (self.astType, self.astKind,
                                            self.value, self.children, self.reg)

    def __repr__(self):
        return '<AST object at %s>' % id(self)

    def key(self):
        return (self.astType, self.astKind, self.value, self.children)

    def typecode(self):
        return kind_to_typecode[self.astKind]

    def postorderWalk(self):
        for c in self.children:
            for w in c.postorderWalk():
                yield w
        yield self

    def allOf(self, *astTypes):
        astTypes = set(astTypes)
        for w in self.postorderWalk():
            if w.astType in astTypes:
                yield w


def expressionToAST(ex):
    """Take an expression tree made out of expressions.ExpressionNode,
    and convert to an AST tree.

    This is necessary as ExpressionNode overrides many methods to act
    like a number.
    """
    return ASTNode(ex.astType, ex.astKind, ex.value,
                   [expressionToAST(c) for c in ex.children])


def sigPerms(s):
    """Generate all possible signatures derived by upcasting the given
    signature.
    """
    codes = 'bilfdcx'
    if not s:
        yield ''
    elif s[0] in codes:
        start = codes.index(s[0])
        for x in codes[start:]:
            for y in sigPerms(s[1:]):
                yield x + y
    elif s[0] == 's':  # numbers shall not be cast to strings
        for y in sigPerms(s[1:]):
            yield 's' + y
    else:
        yield s


def typeCompileAst(ast):
    """Assign appropiate types to each node in the AST.

    Will convert opcodes and functions to appropiate upcast version,
    and add "cast" ops if needed.
    """

    
    children = list(ast.children)
    if ast.astType == 'op':
        retsig = ast.typecode()
        basesig = ''.join(x.typecode() for x in list(ast.children))
        # Find some operation that will work on an acceptable casting of args.
        for sig in sigPerms(basesig):
            value = (ast.value + '_' + retsig + sig).encode('latin-1')
            if value in interpreter.opcodes:
                break
        else:
            for sig in sigPerms(basesig):
                funcname = (ast.value + '_' + retsig + sig).encode('latin-1')
                if funcname in interpreter.funccodes:
                    value = ('func_%sn' % (retsig + sig)).encode('latin-1')
                    children += [ASTNode('raw', 'none',
                                         interpreter.funccodes[funcname])]
                    break
            else:
                raise NotImplementedError(
                    "couldn't find matching opcode for '%s'"
                    % (ast.value + '_' + retsig + basesig))
        # First just cast constants, then cast variables if necessary:
        for i, (have, want) in enumerate(zip(basesig, sig)):
            if have != want:
                kind = typecode_to_kind[want]
                if children[i].astType == 'constant':
                    children[i] = ASTNode('constant', kind, children[i].value)
                else:
                    opname = "cast"
                    children[i] = ASTNode('op', kind, opname, [children[i]])
    else:
        value = ast.value
        children = ast.children
    return ASTNode(ast.astType, ast.astKind, value,
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
        return '%s(%s, %s, %s)' % (name, self.node.astType,
                                   self.node.astKind, self.n,)

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
                t = types.get(name, default_type)
                names[name] = expressions.VariableNode(name, type_to_kind[t])
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


def convertConstantToKind(x, kind):
    # Exception for 'float' types that will return the NumPy float32 type
    if kind == 'float':
        return numpy.float32(x)
    return kind_to_type[kind](x)


def getConstants(ast):
    const_map = {}
    for a in ast.allOf('constant'):
        const_map[(a.astKind, a.value)] = a
    ordered_constants = const_map.keys()
    ordered_constants.sort()
    constants_order = [const_map[v] for v in ordered_constants]
    constants = [convertConstantToKind(a.value, a.astKind)
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
            a.astType = 'alias'
            a.value = target
            a.children = ()
            aliases.append(a)
        else:
            seen[a] = a
    # Set values and registers so optimizeTemporariesAllocation
    # doesn't get confused
    for a in aliases:
        while a.value.astType == 'alias':
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
                    unused[reg.node.astKind].add(reg)
        if unused[n.astKind]:
            reg = unused[n.astKind].pop()
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
        if node.astType == 'alias':
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

    def nToChr(reg):
        if reg is None:
            return b'\xff'
        elif reg.n < 0:
            raise ValueError("negative value for register number %s" % reg.n)
        else:
            if sys.version_info[0] < 3:
                return chr(reg.n)
            else:
                # int.to_bytes is not available in Python < 3.2
                #return reg.n.to_bytes(1, sys.byteorder)
                return bytes([reg.n])

    def quadrupleToString(opcode, store, a1=None, a2=None):
        cop = unichr(interpreter.opcodes[opcode]).encode('latin-1')
        cs = nToChr(store)
        ca1 = nToChr(a1)
        ca2 = nToChr(a2)
        return cop + cs + ca1 + ca2

    def toString(args):
        while len(args) < 4:
            args += (None,)
        opcode, store, a1, a2 = args[:4]
        s = quadrupleToString(opcode, store, a1, a2)
        l = [s]
        args = args[4:]
        while args:
            s = quadrupleToString(b'noop', *args[:3])
            l.append(s)
            args = args[3:]
        return b''.join(l)

    prog_str = b''.join([toString(t) for t in program])
    return prog_str


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

    if ex.astType != 'op':
        ast = ASTNode('op', value='copy', astKind=ex.astKind, children=(ast,))

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
    signature = ''.join(type_to_typecode[types.get(x, default_type)]
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
    program = compileThreeAddrForm(threeAddrProgram)
    return interpreter.NumExpr(inputsig.encode('latin-1'),
                               tempsig.encode('latin-1'),
                               program, constants, input_names)


def disassemble(nex):
    """
    Given a NumExpr object, return a list which is the program disassembled.
    """
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
        if arg == 255:
            return None
        if code != b'n':
            if arg == 0:
                return b'r0'
            elif arg < r_constants:
                return ('r%d[%s]' % (arg, nex.input_names[arg - 1])).encode('latin-1')
            elif arg < r_temps:
                return ('c%d[%s]' % (arg, nex.constants[arg - r_constants])).encode('latin-1')
            else:
                return ('t%d' % (arg,)).encode('latin-1')
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


def getType(a):
    kind = a.dtype.kind
    if kind == 'b':
        return bool
    if kind in 'iu':
        if a.dtype.itemsize > 4:
            return long_  # ``long`` is for integers of more than 32 bits
        if kind == 'u' and a.dtype.itemsize == 4:
            return long_  # use ``long`` here as an ``int`` is not enough
        return int_
    if kind == 'f':
        if a.dtype.itemsize > 4:
            return double  # ``double`` is for floats of more than 32 bits
        return float
    if kind == 'c':
        # RAM need to distinguish between complex64 and complex128 here
        if a.dtype.itemsize > 8:
            return complex
        return complex64
    if kind == 'S':
        return bytes
    raise ValueError("unknown type %s" % a.dtype.name)


def getExprNames(text, context):
    ex = stringToExpression(text, {}, context)
    ast = expressionToAST(ex)
    input_order = getInputOrder(ast, None)
    #try to figure out if vml operations are used by expression
    if not use_vml:
        ex_uses_vml = False
    else:
        for node in ast.postorderWalk():
            if node.astType == 'op' \
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
    signature = [(name, getType(arg)) for (name, arg) in zip(names, arguments)]

    # Look up numexpr if possible.
    numexpr_key = expr_key + (tuple(signature),)
    try:
        compiled_ex = _numexpr_cache[numexpr_key]
    except KeyError:
        compiled_ex = _numexpr_cache[numexpr_key] = \
            NumExpr(ex, signature, **context)
    kwargs = {'out': out, 'order': order, 'casting': casting,
              'ex_uses_vml': ex_uses_vml}
    return compiled_ex(*arguments, **kwargs)
