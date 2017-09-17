/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: BSD
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "module.hpp"
#include <structmember.h>

#include "numexpr_config.hpp"
#include "interp_header_GENERATED.hpp"
#include "numexpr_object.hpp"
#include "benchmark.hpp"

#if defined(_WIN32) && defined(BENCHMARKING)
    extern LARGE_INTEGER TIMES[512];
    extern LARGE_INTEGER T_NOW;
    extern double FREQ;
#elif defined(BENCHMARKING) // Linux
    extern timespec TIMES[512];
    extern timespec T_NOW;
#endif

static void
NumExpr_dealloc(NumExprObject *self)
{
    // Temporaries are held at the module level in an memory block.
    free( self->program );
    free( self->registers );
    free( self->scalar_mem );
    Py_TYPE(self)->tp_free( (PyObject*)self );
}

static PyObject *
NumExpr_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    NumExprObject *self = (NumExprObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
#define INIT_WITH(name, object) \
        self->name = object; \
        if (!self->name) { \
            Py_DECREF(self); \
            return NULL; \
        }
        Py_INCREF(Py_None);
        self->program = NULL;
        self->registers = NULL;
        self->scalar_mem = NULL;
        self->n_reg = 0;
        self->n_scalar = 0;
        self->n_temp = 0;
        self->program_len = 0;
        self->scalar_mem_size = 0;
#undef INIT_WITH
    }
    return (PyObject *)self;
}

static int
ssize_from_dchar(char c)
{
    switch (c) {
        case '?': return sizeof(npy_bool);
        case 'b': return sizeof(npy_int8);
        case 'B': return sizeof(npy_uint8);
        case 'h': return sizeof(npy_int16);
        case 'H': return sizeof(npy_uint16);
#ifdef _WIN32
        case 'l': return sizeof(npy_int32);
        case 'L': return sizeof(npy_uint32);
        case 'q': return sizeof(npy_int64);
        case 'Q': return sizeof(npy_uint64);
#else
        case 'i': return sizeof(npy_int32);
        case 'I': return sizeof(npy_uint32);
        case 'l': return sizeof(npy_int64);
        case 'L': return sizeof(npy_uint64);
#endif
        case 'f': return sizeof(npy_float32);
        case 'F': return sizeof(npy_complex64);
        case 'd': return sizeof(npy_float64);
        case 'D': return sizeof(npy_complex128);
        case 'S': return 0;  // strings are ok but size must be computed
        case 'U': return 0;  // strings are ok but size must be computed
        default:
            PyErr_Format(PyExc_TypeError,
                                "signature char '%c' not in '?bBhHiIlLdDfFSU'", c);
            return -1;
    }
}

static int
NumExpr_init(NumExprObject *self, PyObject *args)
{
    // NE2to3: Now uses structs instead of seperate arrays for each field
    PyObject *bytes_prog = NULL, *reg_tuple = NULL;
    NumExprOperation *program = NULL;
    NumExprReg *registers = NULL;
    PyObject *arrayObj;
    PyObject *iter_reg = NULL;
    int I, program_len = 0;
    NE_REGISTER n_reg = 0, n_scalar = 0, n_temp = 0, n_array = 0, returnReg = -1, returnOperand = -1;

    // Build const blocks variables
    npy_intp total_scalar_itemsize = 0, mem_offset = 0, total_temp_itemsize = 0;
    char *scalar_mem = NULL, *mem_loc;

    BENCH_TIME(50);
    
    if( ! PyArg_ParseTuple(args, "SO", &bytes_prog, &reg_tuple ) ) { 
        PyErr_Format(PyExc_RuntimeError,
                    "numexpr_object.cpp: Could not parse input arguments." );
        return -1; 
    }
    // NE2to3: Move the check_program() logic into _Init before allocating 
    // new memory. It makes zeros sense to have it in interpreter.cpp
    if( ! PyBytes_Check(bytes_prog) ) { // Check if program_bytes is a byte string
        PyErr_Format(PyExc_RuntimeError,
                    "numexpr_object.cpp: argument 'program' is not a bytes string.");
        return -1;
    }
    if (PyBytes_GET_SIZE(bytes_prog) % NE_PROG_LEN != 0) {
        PyErr_Format(PyExc_RuntimeError, 
            "numexpr_object.cpp: invalid program: prog_len %d mod %d = %d", 
            PyBytes_GET_SIZE(bytes_prog), NE_PROG_LEN, PyBytes_GET_SIZE(bytes_prog) % NE_PROG_LEN );
        return -1;
    }
    if( ! PyTuple_Check(reg_tuple) ) { // Check if reg_tuple is a tuple
        PyErr_Format(PyExc_RuntimeError,
                    "numexpr_object.cpp: argument 'registers' is not a tuple.");
        return -1;
    }
    n_reg = (int)PyTuple_GET_SIZE( reg_tuple );
    if( n_reg > NE_MAX_BUFFERS ) { // Numpy is limited to 32 args at present
        PyErr_Format(PyExc_RuntimeError,
                    "numexpr_object.cpp: No. buffers (%d) exceeds %d.", n_reg, NE_MAX_BUFFERS);
        return -1;
    }

    program_len = (int)( PyBytes_GET_SIZE(bytes_prog) / NE_PROG_LEN );
    // Cast from Python_Bytes->char array->NumExprOperation struct array
    // which works because we use struct.pack() in Python.

    // We are avoiding using PyMalloc and similar due to the changes in 3.6 
    // resulting in some inconsistant behavoir.
    program = (NumExprOperation*)malloc( program_len * sizeof(NumExprOperation) );
    memcpy( program, PyBytes_AsString( bytes_prog ), program_len*sizeof(NumExprOperation) );

    // Build registers
    // arrays = PyMem_New(PyObject*, n_reg);
    registers = (NumExprReg *)malloc(n_reg * sizeof(NumExprReg) );
    // registers = PyMem_New(NumExprReg, n_reg);
    //arrays = (PyObject**)malloc(n_reg * sizeof(PyObject *) );
    for( I = 0; I < n_reg; I++ ) {
        // Get reference and check format of register
        iter_reg = PyTuple_GET_ITEM( reg_tuple, I );

        // Python does error checking here for us.
        // PyUnicode_AsUTF8 doesn't exist in 2.7
        // We can use PyString_AsString
#if PY_MAJOR_VERSION >= 3
        // if (PyTuple_GetItem( iter_reg, 2 ) == Py_None) {
        //     // Should only happen if this is the return array
        //     // In which case we can get the return type from the last program
        //     PyErr_Format(PyExc_RuntimeError,
        //         "numexpr_object.cpp: register[%d] has unknown dtype type.\n", I );
        //     return -1;
        // }
        registers[I].dchar = (char)PyUnicode_AsUTF8( PyTuple_GetItem( iter_reg, 2 ) )[0];
#else        
        registers[I].dchar = (char)PyString_AsString( PyTuple_GetItem( iter_reg, 2 ) )[0];  
#endif
        registers[I].kind = (npy_uint8)PyLong_AsUnsignedLong( PyTuple_GetItem( iter_reg, 3 ) );

        // PyArray_ITEMSIZE gives a wrong result for scalars (and strings), so 
        // a lookup table is needed.
        if (registers[I].kind == KIND_TEMP) {
            // For temps the dchar is not defined, and we allocate based on max_itemsize alone
            registers[I].itemsize = (npy_intp)PyLong_AsUnsignedLong( PyTuple_GetItem( iter_reg, 4 ) );
        }
        else {
            registers[I].itemsize = (npy_intp)ssize_from_dchar(registers[I].dchar);
        }
        
        // printf( "Register #%d: dchar: %c, array: %p, kind: %d, itemsize: %d\n", I, registers[I].dchar, arrays[I], registers[I].kind, registers[I].itemsize );

        // Arrays are now only checked for scalars.
        registers[I].mem = NULL;
        switch( registers[I].kind ) {
            case KIND_ARRAY:
                //printf( "Found array at register %d with %lu elements, dtype %c with itemsize %lu\n", 
                //    I, registers[I].elements, registers[I].dchar, registers[I].itemsize );
                n_array++;
                break;
            case KIND_SCALAR:
                //printf( "Found scalar at register %d, dtype %c with itemsize %lu\n", 
                //       I, registers[I].dchar, registers[I].itemsize );
                n_scalar++;
                total_scalar_itemsize += registers[I].itemsize;
                break;
            case KIND_TEMP:
                //printf( "Found temporary at register %d, dtype %c with itemsize %lu\n", 
                //        I, registers[I].dchar, registers[I].itemsize );
                n_temp++;
                total_temp_itemsize += registers[I].itemsize;
                // Per-register stride is set in GENERATED functions now, as 
                // the stride of a temporary can change throughout the program.
                registers[I].stride = registers[I].itemsize;
                break;
            case KIND_RETURN:
                //printf( "Found return at register %d\n", I );
                returnReg = I;
                returnOperand = n_array;
                n_array++;
                break;
            default:
                PyErr_Format(PyExc_RuntimeError,
                        "numexpr_object.cpp: register[%d] has unknown register type %d.\n", I, registers[I].kind );
                return -1;
                break;
        }        
    }
    
    // Pre-build the scalar blocks.
    // In NE2 the scalars were BLOCK_SIZE1 arrays, but if we set the stride to 
    // zero they can be single values, which makes pickling much more practical.
    if( total_scalar_itemsize > 0) {
        scalar_mem = (char *)malloc( total_scalar_itemsize );
        for( I = 0; I < n_reg; I++ ) {
            if( registers[I].kind != KIND_SCALAR ) 
                continue;

            // Fill in the scalar registers
            arrayObj = (PyObject *)PyTuple_GET_ITEM( PyTuple_GET_ITEM( reg_tuple, I ), 1 );
            if ( ! PyArray_Check( arrayObj ) ) {
                PyErr_Format(PyExc_RuntimeError,
                    "numexpr_object.cpp: scalar register[%d] is not of type ndarray.\n", I );
            }
            registers[I].mem = mem_loc = scalar_mem + mem_offset;
            // Stride for scalars is always zero
            registers[I].stride = 0;
            // Copy scalar value
            memcpy( mem_loc, PyArray_DATA( (PyArrayObject *)arrayObj), registers[I].itemsize );
            // Advance memory pointer
            mem_offset += registers[I].itemsize;
        }
    }

    //REPLACE_MEM(program);
    //REPLACE_MEM(registers);
    //free(arrays);
    //REPLACE_MEM(scalar_mem);
    self->program = program;
    self->registers = registers;
    self->scalar_mem = scalar_mem;
    self->program_len = program_len;
    self->n_reg = n_reg;
    self->n_array = n_array;
    self->n_scalar = n_scalar;
    self->n_temp = n_temp;
    self->scalar_mem_size = total_scalar_itemsize;
    self->total_temp_itemsize = total_temp_itemsize;
    self->returnReg = returnReg;
    self->returnOperand = returnOperand;


    // DEBUG: print the program to screen
    // printf( "NumExpr_init(): program_len = %d, n_array = %d, n_scalar = %d, n_temp = %d\n", 
    //    program_len, n_array, n_scalar, n_temp );
    // for( I = 0; I < self->program_len; I++ ) {
    //     printf( "NE_init: Program[%d]: %d %d %d %d %d \n", I,
    //         (int)self->program[I].op, (int)self->program[I].ret, (int)self->program[I].arg1, 
    //         (int)self->program[I].arg2, (int)self->program[I].arg3 );
    // }
    DIFF_TIME(50);
    return 0;
}


static PyObject*
NumExpr_getstate(NumExprObject *self) {
    // Pickle support
    // The NumExprObject struct and its childern is packed into a C-string
    // which is then returned as a PyBytes object which is pickleable.

    // Compute total size of self, program, registers, and scalarmem
    npy_intp totalSize = sizeof(NumExprObject) + 
                         self->program_len*sizeof(NumExprOperation) +
                         self->n_reg*sizeof(NumExprReg) + 
                         self->scalar_mem_size;


    char* state = (char *)malloc(totalSize);
    npy_intp statePoint = 0;

    // printf( "TODO: likely PyObject_HEAD should not be overwritten.\n" );
    // Copy top-level NumExprObject
    memcpy( state, self, sizeof(NumExprObject) );
    statePoint += sizeof(NumExprObject);
    // Copy program
    memcpy( state+statePoint, self->program, self->program_len*sizeof(NumExprOperation) );
    statePoint += self->program_len*sizeof(NumExprOperation);
    // Copy registers
    memcpy( state+statePoint, self->registers, self->n_reg*sizeof(NumExprReg) );
    statePoint += self->n_reg*sizeof(NumExprReg);
    // Copy scalar_mem
    memcpy( state+statePoint, self->scalar_mem, self->scalar_mem_size );


    return PyBytes_FromStringAndSize( state, totalSize );
}

static PyObject*
NumExpr_setstate(NumExprObject *self, PyObject *args) {
    // printf( "In __setstate__, ob_base: %p\n", &self->ob_base );
    npy_intp statePoint = 0, mem_offset = 0;

    PyObject *state_obj = NULL;
    // Get 'state' from args
    if( ! PyArg_ParseTuple(args, "S",  &state_obj ) ) { 
        PyErr_Format(PyExc_RuntimeError,
                    "numexpr_object.cpp: Pickled state expected as bytes." );
        return NULL; 
    }
    char* state = PyBytes_AsString( state_obj );

    //memcpy( self+SIZEOF_PYOBJECT_HEAD, state+SIZEOF_PYOBJECT_HEAD, sizeof(NumExprObject)-SIZEOF_PYOBJECT_HEAD );
    memcpy( self, state, sizeof(NumExprObject) );
    statePoint += sizeof(NumExprObject);

    // Do some simple limit checks on program_len and n_reg
    // printf( "TODO: checks, program_len: %d, n_reg: %d\n", self->program_len, self->n_reg );

    // Load program
    self->program = (NumExprOperation*)malloc( self->program_len*sizeof(NumExprOperation) );
    memcpy( self->program, state+statePoint, self->program_len*sizeof(NumExprOperation) );
    statePoint += self->program_len*sizeof(NumExprOperation);

    // Load registers
    self->registers = (NumExprReg*)malloc( self->n_reg*sizeof(NumExprReg) );
    memcpy( self->registers, state+statePoint, self->n_reg*sizeof(NumExprReg) );
    statePoint += self->n_reg*sizeof(NumExprReg);

    // Load scalar_mem
    self->scalar_mem = (char *)malloc( self->scalar_mem_size );
    memcpy( self->scalar_mem, state+statePoint, self->scalar_mem_size );
    // Restore scalar memory pointers in registers
    
    for ( NE_REGISTER R = 0; R < self->n_reg; R++) {
        if( self->registers[R].kind != KIND_SCALAR ) continue;
        // Set pointer in the register to the block in scalar_mem
        self->registers[R].mem = self->scalar_mem + mem_offset;
        // Advance memory pointer
        mem_offset += self->registers[R].itemsize;
    }

    return Py_BuildValue("");
}
 
static PyObject*
NumExpr_print_state(NumExprObject *self, PyObject *args) {
    printf( "Sizeof(NumExprObject): %zd\n", sizeof(NumExprObject) );
    printf( "Sizeof(self->ob_base): %zd\n", sizeof(self->ob_base) );
    printf( "Object base pointer: %p\n", &self->ob_base );
    printf( "\nProgram pointer: %p\n", self->program );
    printf( "program_len: %d\n", self->program_len );
    for( int I = 0; I < self->program_len; I++ ) {
        printf( "    #%d:: OP: %u, RET: %u, ARG1: %u, ARG2: %u, ARG3: %u\n", I,
            self->program[I].op, self->program[I].ret, self->program[I].arg1, 
            self->program[I].arg2, self->program[I].arg3  );
    }
    printf( "\nRegisters pointer: %p\n", self->registers );
    printf( "n_reg: %d\n", self->n_reg );
    printf( "n_array: %d\n", self->n_array );
    printf( "n_scalar: %d\n", self->n_scalar );
    printf( "n_temp: %d\n", self->n_temp );
    for( int J = 0; J < self->n_reg; J++ ) {
        printf( "    #%d:: mem: %p, dchar: %c, kind: %d, itemsize: %Id, stride: %Id\n", 
            J, self->registers[J].mem, self->registers[J].dchar, 
            self->registers[J].kind, (int)self->registers[J].itemsize, 
            (int)self->registers[J].stride );
    }

    printf( "\nScalar memory pointer: %p\n", self->scalar_mem );
    printf( "scalar_mem_size: %ld\n", self->scalar_mem_size );
    

    return Py_BuildValue(""); 
}

// C-api structures:
// https://docs.python.org/3/c-api/structures.html
static PyMethodDef NumExpr_methods[] = {
    {"run", (PyCFunction) NumExpr_run, METH_VARARGS|METH_KEYWORDS, NULL},
    {"print_state", (PyCFunction) NumExpr_print_state, METH_NOARGS, NULL},
    {"__getstate__", (PyCFunction) NumExpr_getstate, METH_NOARGS, NULL},
    {"__setstate__", (PyCFunction) NumExpr_setstate, METH_VARARGS, NULL},
    {NULL, NULL}
};

static PyMemberDef NumExpr_members[] = {
    // registers and program should be private because they aren't PyObjects.
    //{ CHARP("registers"), T_OBJECT_EX, offsetof(NumExprObject, registers), READONLY, NULL},
    //{ CHARP("program"), T_OBJECT_EX, offsetof(NumExprObject, program), READONLY, NULL},
    {NULL},
};

PyTypeObject NumExprType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "numexpr3.interpreter.CompiledExec",   /*tp_name*/
    sizeof(NumExprObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)NumExpr_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    (ternaryfunc)NumExpr_run,  /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "NumExpr compiled executable",         /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    NumExpr_methods,           /* tp_methods */
    NumExpr_members,           /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)NumExpr_init,    /* tp_init */
    0,                         /* tp_alloc */
    NumExpr_new,               /* tp_new */
};

