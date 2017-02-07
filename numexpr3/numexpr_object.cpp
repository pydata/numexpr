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
    // Where are threads.params dealloc'd?
    
    // Free temporaries
    for ( NE_REGISTER R = 0; R < self->n_reg; R++) {
        if( self->registers[R].kind == KIND_TEMP ) {
            PyMem_Free( self->registers[R].mem  );
        }
    }
    
    PyMem_Del( self->program );
    PyMem_Del( self->registers );
    PyMem_Del( self->scalar_mem );
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

        self->program = NULL;
        self->registers = NULL;
        self->scalar_mem = NULL;
        self->n_reg = 0;
        self->n_scalar = 0;
        self->n_temp = 0;
        self->program_len = 0;
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
    
//static int
//NPYENUM_from_dchar(char c)
//{
//    switch (c) {
//        case '?': return NPY_BOOL;
//        case 'b': return NPY_INT8;
//        case 'B': return NPY_UINT8;
//        case 'h': return NPY_INT16;
//        case 'H': return NPY_UINT16;
//        case 'i': return NPY_INT32;
//        case 'I': return NPY_UINT32;
//        case 'l': return NPY_INT64;
//        case 'L': return NPY_UINT64;
//        case 'f': return NPY_FLOAT32;
//        case 'F': return NPY_COMPLEX64;
//        case 'd': return NPY_FLOAT64;
//        case 'D': return NPY_COMPLEX128;
//        case 'S': return NPY_STRING;
//        case 'U': return NPY_UNICODE;
//        default:
//            PyErr_SetString(PyExc_TypeError, "signature char not in '?bBhHiIlLdDfFSU'");
//            return -1;
//    }
//}

//static int
//ssize_from_NPYENUM(int e)
//{
//    switch (e) {
//        case NPY_BOOL:       return sizeof(npy_bool);
//        case NPY_INT8:       return sizeof(npy_int8);
//        case NPY_UINT8:      return sizeof(npy_uint8);
//        case NPY_INT16:      return sizeof(npy_int16);
//        case NPY_UINT16:     return sizeof(npy_uint16);
//        case NPY_INT32:      return sizeof(npy_int32);
//        case NPY_UINT32:     return sizeof(npy_uint32);
//        case NPY_INT64:      return sizeof(npy_int64);
//        case NPY_UINT64:     return sizeof(npy_uint64);
//        case NPY_FLOAT32:    return sizeof(npy_float32);
//        case NPY_FLOAT64:    return sizeof(npy_float64);
//        case NPY_COMPLEX64:  return sizeof(npy_complex64);
//        case NPY_COMPLEX128: return sizeof(npy_complex128);  
//        case NPY_UNICODE:    return 0;  // strings are ok but size must be computed
//        case NPY_STRING:     return 0;  // strings are ok but size must be computed
//        default:
//            PyErr_SetString(PyExc_TypeError, "signature value not in NPY_ENUMs");
//            return -1;
//    }
//}

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
    int I, J, K; 

    // Build const blocks variables
    npy_intp total_scalar_itemsize = 0, scalar_size = 0;
    char *scalar_pointer, *scalar_mem, *mem_loc;
    int scalar_mem_size = 0, mem_offset = 0;
    
    
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
    program = (NumExprOperation*)PyMem_New( NumExprOperation, program_len );
    memcpy( program, PyBytes_AsString( bytes_prog ), program_len*sizeof(NumExprOperation) );

    
    // Build registers
    registers = PyMem_New(NumExprReg, n_reg);
    arrays = PyMem_New(PyObject*, n_reg);
    for( I = 0; I < n_reg; I++ ) {
        // Get reference and check format of register
        iter_reg = PyTuple_GET_ITEM( reg_tuple, I );
        
        // Fill in the registers
        // Note: in NE2 the array handles are only set in NumExpr_Run() because
        // NumPy iterators are used.
        arrays[I] = (PyObject *)PyTuple_GET_ITEM( iter_reg, 1 );
        
        // Python does error checking here for us.
        registers[I].dchar = (char)PyUnicode_AsUTF8( PyTuple_GetItem( iter_reg, 2 ) )[0];
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
            // Apparently the documented way to access the data pointer is depricated in 1.7...
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
                //        I, registers[I].dchar, registers[I].itemsize );
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
    scalar_mem_size = total_scalar_itemsize * BLOCK_SIZE1;
    // TODO: Hrm... BLOCK_SIZE1 scales with the data type... it should be a fixed 
    // size in bytes to fit into L1 cache nicely.
    scalar_mem = PyMem_New( char, scalar_mem_size );

    for( I = 0; I < n_reg; I++ ) {
        if( registers[I].kind != KIND_SCALAR ) continue;
        
        registers[I].mem = mem_loc = scalar_mem + mem_offset;

        
        mem_offset += BLOCK_SIZE1 * registers[I].itemsize;

        // Fill in the BLOCKs for scalar arrays
        scalar_size = registers[I].itemsize;
        scalar_pointer = PyArray_BYTES( (PyArrayObject *)arrays[I] );

        for( J = 0; J < BLOCK_SIZE1; J++ ) {
            // Essentially a memcpy but likely optimized to be faster by the compiler
            for( K = 0; K < scalar_size; K++ ) {
                *mem_loc = scalar_pointer[K];
                mem_loc++;
            }
            
        } 
        // RAM: Passes valgrind check
    }

    #define REPLACE_OBJ(argument) \
    {PyObject *temp = self->argument; \
     self->argument = argument; \
     Py_XDECREF(temp);}
    #define INCREF_REPLACE_OBJ(argument) {Py_INCREF(argument); REPLACE_OBJ(argument);}
    #define REPLACE_MEM(argument) {PyMem_Del(self->argument); self->argument=argument;}
    
    //PyMem_Del(iter_reg);
    //PyMem_Del(iter_arr);
    
    // Do some more reading on how CPython handles garbage collection.
    // https://docs.python/org/3/c-api/memory.html
    REPLACE_MEM(program);
    REPLACE_MEM(registers);
    REPLACE_MEM(scalar_mem);
    self->program_len = program_len;
    self->n_reg = n_reg;
    self->n_ndarray = n_ndarray;
    self->n_scalar = n_scalar;
    self->n_temp = n_temp;
    
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

static PyMethodDef NumExpr_methods[] = {
    {"run", (PyCFunction) NumExpr_run, METH_VARARGS|METH_KEYWORDS, NULL},
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
    "numexpr.NumExpr",         /*tp_name*/
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
    "NumExpr objects",         /* tp_doc */
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

