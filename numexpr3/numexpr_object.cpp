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



static void
NumExpr_dealloc(NumExprObject *self)
{
    // Free temporaries is done by free_temps_space()

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
        case 'i': return sizeof(npy_int32);
        case 'I': return sizeof(npy_uint32);
        case 'l': return sizeof(npy_int64);
        case 'L': return sizeof(npy_uint64);
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
NumExpr_init(NumExprObject *self, PyObject *args, PyObject *kwargs)
{
    // NE2to3: Now uses structs instead of seperate arrays for each field
    PyObject *bytes_prog = NULL, *reg_tuple = NULL;
    NumExprOperation *program = NULL;
    NumExprReg *registers = NULL;
    PyObject **arrays; // C-array of PyArrayObjects
    PyObject *iter_reg = NULL;
    int n_reg = 0, n_scalar = 0, n_temp = 0, n_ndarray = 0, program_len = 0;
    int I; 

    // Build const blocks variables
    npy_intp total_scalar_itemsize = 0, mem_offset = 0;
    char *scalar_mem, *mem_loc;
    
    
    static char *kwlist[] = { CHARP("program"), CHARP("registers"), NULL };
    
    if( ! PyArg_ParseTupleAndKeywords(args, kwargs, "SO", kwlist, 
                                      &bytes_prog, &reg_tuple ) ) { 
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
    
    // We have to copy bytes_prog or Python garbage collects it
    //program = (NumExprOperation*)PyMem_Malloc( program_len*sizeof(NumExprOperation) );
    //program = (NumExprOperation*)PyMem_New( NumExprOperation, program_len );
    program = (NumExprOperation*)malloc( program_len * sizeof(NumExprOperation) );
    memcpy( program, PyBytes_AsString( bytes_prog ), program_len*sizeof(NumExprOperation) );

    // Build registers
    // registers = PyMem_New(NumExprReg, n_reg);
    // arrays = PyMem_New(PyObject*, n_reg);
    registers = (NumExprReg *)malloc(n_reg * sizeof(NumExprReg) );
    arrays = (PyObject**)malloc(n_reg * sizeof(PyObject *) );
    for( I = 0; I < n_reg; I++ ) {
        // Get reference and check format of register
        iter_reg = PyTuple_GET_ITEM( reg_tuple, I );

        // Fill in the registers
        // Note: in NE2 the array handles are only set in NumExpr_Run() because
        // NumPy iterators are used.
        arrays[I] = (PyObject *)PyTuple_GET_ITEM( iter_reg, 1 );

        // Python does error checking here for us.
        // PyUnicode_AsUTF8 doesn't exist in 2.7
        // We can use PyString_AsString
#if PY_MAJOR_VERSION >= 3        
        registers[I].dchar = (char)PyUnicode_AsUTF8( PyTuple_GetItem( iter_reg, 2 ) )[0];
#else        
        registers[I].dchar = (char)PyString_AsString( PyTuple_GetItem( iter_reg, 2 ) )[0];  
#endif
        registers[I].kind = (npy_uint8)PyLong_AsUnsignedLong( PyTuple_GetItem( iter_reg, 3 ) );

        // PyArray_ITEMSIZE gives a wrong result for scalars (and strings), so 
        // a lookup table is needed.
        registers[I].itemsize = (npy_intp)ssize_from_dchar(registers[I].dchar);
        // The stride is always one element for scalars and temps but it will 
        // potentially be different for numpy.ndarrays.
        registers[I].stride = registers[I].itemsize;

        if( arrays[I] == Py_None ) {
            registers[I].elements = 0;
            registers[I].mem = NULL;
        }
        else if ( PyArray_Check( arrays[I] ) ) {
            // https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayObject
            registers[I].elements = PyArray_Size(arrays[I]);
            registers[I].mem = NULL;
        }
        
        switch( registers[I].kind ) {
            case KIND_ARRAY:
                //printf( "Found array at register %d with %lu elements, dtype %c with itemsize %lu\n", 
                //    I, registers[I].elements, registers[I].dchar, registers[I].itemsize );
                n_ndarray++;
                break;
            case KIND_SCALAR:
                //printf( "Found scalar at register %d, dtype %c with itemsize %lu\n", 
                //       I, registers[I].dchar, registers[I].itemsize );
                n_scalar++;
                // TODO: build a BLOCK_SIZE1 ones array
                total_scalar_itemsize += registers[I].itemsize;
                break;
            case KIND_TEMP:
                //printf( "Found temporary at register %d, dtype %c with itemsize %lu\n", 
                //        I, registers[I].dchar, registers[I].itemsize );
                n_temp++;
                break;
            case KIND_RETURN:
                //printf( "Found return at register %d\n", I );
                // This is handled by NumExpr_run() as the array pointer can change.
                n_ndarray++;
                break;
            default:
                PyErr_Format(PyExc_RuntimeError,
                        "numexpr_object.cpp: register[%d] has unknown register type %d.\n", I, registers[I].kind );
                return -1;
                break;
        }    

         
    }
    // DEBUG
    //printf( "NumExpr_init(): program_len = %d, n_ndarray = %d, n_scalar = %d, n_temp = %d\n", 
    //    program_len, n_ndarray, n_scalar, n_temp );
    
    
    // Pre-build the scalar blocks.
    // Here we address the memory directly to avoid having a nested casting
    // tree as in NE2. This should make the code more general and hence more 
    // maintainable.
    // The other option would be PyArray_Ones()
    //scalar_mem_size = total_scalar_itemsize * BLOCK_SIZE1;

    //scalar_mem = PyMem_New( char, scalar_mem_size );
    // TODO: we shouldn't need BLOCKSIZE1 arrays for scalars, nditer should be able to deal with 
    // single elements and return a step of 0.
    if( total_scalar_itemsize > 0) {
        scalar_mem = (char *)malloc( total_scalar_itemsize );
        for( I = 0; I < n_reg; I++ ) {
            if( registers[I].kind != KIND_SCALAR ) continue;
            
            registers[I].mem = mem_loc = scalar_mem + mem_offset;
            // Stride for scalars is always zero
            registers[I].stride = 0;
            // Copy scalar value
            memcpy( mem_loc, PyArray_DATA( (PyArrayObject *)arrays[I]), registers[I].itemsize );
            // Advance memory pointer
            mem_offset += registers[I].itemsize;

            // RAM: Passes valgrind check
        }
    }

    #define REPLACE_OBJ(argument) \
    {PyObject *temp = self->argument; \
     self->argument = argument; \
     Py_XDECREF(temp);}
    #define INCREF_REPLACE_OBJ(argument) {Py_INCREF(argument); REPLACE_OBJ(argument);}
    #define REPLACE_MEM(argument) {PyMem_Del(self->argument); self->argument=argument;}


    //REPLACE_MEM(program);
    //REPLACE_MEM(registers);
    //PyMem_Del(arrays);
    free(arrays);
    //REPLACE_MEM(scalar_mem);
    self->program = program;
    self->registers = registers;
    self->scalar_mem = scalar_mem;
    self->program_len = program_len;
    self->n_reg = n_reg;
    self->n_ndarray = n_ndarray;
    self->n_scalar = n_scalar;
    self->n_temp = n_temp;
    self->scalar_mem_size = total_scalar_itemsize;
    // DEBUG: print the program to screen
//    for( I = 0; I < self->program_len; I++ ) {
//        printf( "NE_init: Program[%d]: %d %d %d %d %d \n", I,
//            (int)self->program[I].op, (int)self->program[I].ret, (int)self->program[I].arg1, 
//            (int)self->program[I].arg2, (int)self->program[I].arg3 );
//    }
    #undef REPLACE_OBJ
    #undef INCREF_REPLACE_OBJ
    #undef REPLACE_MEM
    
    // Release debugging memory (if used)
    //PyMem_Del(testObj);
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

    printf( "TODO: likely PyObject_HEAD should not be overwritten.\n" );
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
    printf( "TODO: checks, program_len: %d, n_reg: %d\n", self->program_len, self->n_reg );

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
    printf( "n_ndarray: %d\n", self->n_ndarray );
    printf( "n_scalar: %d\n", self->n_scalar );
    printf( "n_temp: %d\n", self->n_temp );
    for( int J = 0; J < self->n_reg; J++ ) {
        printf( "    #%d:: mem: %p, dchar: %c, kind: %d, itemsize: %Id, stride: %Id, elements: %Id\n", 
            J, self->registers[J].mem, self->registers[J].dchar, 
            self->registers[J].kind, self->registers[J].itemsize, 
            self->registers[J].stride, self->registers[J].elements );
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

