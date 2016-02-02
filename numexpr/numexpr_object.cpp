/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "module.hpp"
#include <structmember.h>

#include "numexpr_config.hpp"
#include "interpreter.hpp"
#include "numexpr_object.hpp"


static int
size_from_NPYENUM(int e)
{
    switch (e) {
        case NPY_BOOL: return sizeof(npy_bool);
        case NPY_INT32: return sizeof(npy_int32);
        case NPY_INT64: return sizeof(npy_int64);
        case NPY_FLOAT32: return sizeof(npy_float32);
        case NPY_FLOAT64: return sizeof(npy_float64);
        case NPY_COMPLEX64: return sizeof(npy_complex64);
        case NPY_COMPLEX128: return sizeof(npy_complex128);    
        case NPY_STRING: return 0;  /* strings are ok but size must be computed */
        default:
            PyErr_SetString(PyExc_TypeError, "signature value not in NPY_ENUMs");
            return -1;
    }
}

static int
typecode_from_bytecode(char* bytecode)
{
    switch (bytecode[0]) {
        // RAM: re-factor to have kind and order seperately
        // http://docs.scipy.org/doc/numpy-1.10.0/reference/c-api.dtype.html
        case 'b': return NPY_BOOL;
        case 'i':
            switch( bytecode[1] ) {
                case '1': return NPY_INT8;
                case '2': return NPY_INT16;
                case '4': return NPY_INT32;
                case '8': return NPY_INT64;
            }
        case 'u':
            switch( bytecode[1] ) {
                case '1': return NPY_UINT8;
                case '2': return NPY_UINT16;
                case '4': return NPY_UINT32;
                case '8': return NPY_UINT64;
            }
        case 'f':
            switch( bytecode[1] ) {
                case '4': return NPY_FLOAT32;
                case '8': return NPY_FLOAT64;
            }
        case 'c': 
            switch( bytecode[1] ) {
                case '8': return NPY_COMPLEX64;
                case '1': return NPY_COMPLEX128; // This one is three chars, but it's still unique ATM
            }
        case 's': return NPY_STRING;
        default:
            PyErr_SetString(PyExc_TypeError, "signature value not in '[buifcs]<byte-depth>'"); //RAM: TODO fix error message
            return -1;
    }
}

static void
NumExpr_dealloc(NumExprObject *self)
{
    Py_XDECREF(self->signature);
    Py_XDECREF(self->tempsig);
    Py_XDECREF(self->constsig);
    //Py_XDECREF(self->fullsig);
    Py_XDECREF(self->program_bytes);
    PyMem_Del( self->program_words );
    Py_XDECREF(self->constants);
    Py_XDECREF(self->input_names);
    PyMem_Del(self->mem);
    PyMem_Del(self->rawmem);
    PyMem_Del(self->memsteps);
    PyMem_Del(self->memsizes);
    Py_TYPE(self)->tp_free((PyObject*)self);
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

        // RAM: replace all string inputs with bytearrays
        //INIT_WITH(signature, PyBytes_FromString(""));
        INIT_WITH(signature, PyByteArray_FromStringAndSize( "", 0) )
        //INIT_WITH(tempsig, PyBytes_FromString(""));
        INIT_WITH(tempsig, PyByteArray_FromStringAndSize( "", 0) )
        //INIT_WITH(constsig, PyBytes_FromString(""));
        INIT_WITH(constsig, PyByteArray_FromStringAndSize( "", 0) )
        //INIT_WITH(fullsig, PyBytes_FromString(""));
        //INIT_WITH(fullsig, PyByteArray_FromStringAndSize( "", 0) )
        //INIT_WITH(program, PyBytes_FromString(""));
        INIT_WITH(program_bytes, PyByteArray_FromStringAndSize( "", 0) )
        INIT_WITH(constants, PyTuple_New(0));
        Py_INCREF(Py_None);
        self->input_names = Py_None;
        self->program_words = NULL;
        self->word_count = 0;
        self->mem = NULL;
        self->rawmem = NULL;
        self->memsteps = NULL;
        self->memsizes = NULL;
        self->rawmemsize = 0;
        self->n_inputs = 0;
        self->n_constants = 0;
        self->n_temps = 0;
#undef INIT_WITH
    }
    return (PyObject *)self;
}



static int
NumExpr_init(NumExprObject *self, PyObject *args, PyObject *kwds)
{
    int i, j, mem_offset;
    int n_inputs, n_constants, n_temps;
    int len_sig, len_constsig, len_tempsig;
    PyObject *signature = NULL, *tempsig = NULL, *constsig = NULL;
    PyObject *program_bytes = NULL, *constants = NULL;
    unsigned short *sig_words, *tempsig_words, *constsig_words;

    PyObject *input_names = NULL, *o_constants = NULL;
    int *itemsizes = NULL;
    char **mem = NULL, *rawmem = NULL;
    npy_intp *memsteps;
    npy_intp *memsizes;
    int rawmemsize;
    static char *kwlist[] = {"signature", "tempsig",
                             "program_bytes",  "constants",
                             "input_names", NULL};

    // RAM: change signature of strings to that of bytearrays
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOO", kwlist,
                                     &signature,
                                     &tempsig,
                                     &program_bytes, &o_constants,
                                     &input_names)) {
        return -1;
    }

    // OK, problem, the number of bytes in signature and temp signature doesn't reflect the number of 
    // inputs anymore.  We need to convert/parse the signatures...
    // So I need a new C-function for that...
    // Maybe I should fix the Python instead and pass in tuples for signature? Yes that's much easier...
    char buffer[1024];

    if( signature && ( n_inputs = (int)PySequence_Length(signature) ) > 0 ){
        if (!PySequence_Check(signature) ) {
                PyErr_SetString(PyExc_TypeError, "signature must be a sequence");
                return -1;
        }
        sprintf( buffer, "print( 'ne_object::finit n_inputs = %d' )", n_inputs );
        PyRun_SimpleString( buffer );


        //PyObject* pho3 = PyString_AsEncodedObject( PyString_FromString(PyByteArray_AsString(sig_bytearray)), "hex", "strict");
        //sprintf( buffer, "print( 'ne_object::finit signature byte array = %s' )", PyString_AsString(pho3) );
        //PyRun_SimpleString( buffer );

        // TODO: this works at present but it might break if the input strings aren't 2 characters long each.
        sig_words = (unsigned short *)malloc( n_inputs );
        for (i = 0; i < n_inputs; i++) {
            sig_words[i] = typecode_from_bytecode( PyString_AS_STRING(PySequence_GetItem(signature, i)) );
            sprintf( buffer, "print( 'ne_object::finit signature[%d] = %d' )", i, sig_words[i] );
            PyRun_SimpleString( buffer );
        }
    }
    
    if( tempsig && (n_temps = (int)PySequence_Length(tempsig)) > 0 ){
        if (!PySequence_Check(tempsig) ) {
                PyErr_SetString(PyExc_TypeError, "tempsig must be a sequence");
                return -1;
        }
        sprintf( buffer, "print( 'ne_object::finit n_temps = %d' )", n_temps );
        PyRun_SimpleString( buffer );

        //PyObject* pho3 = PyString_AsEncodedObject( PyString_FromString(PyByteArray_AsString(sig_bytearray)), "hex", "strict");
        //sprintf( buffer, "print( 'ne_object::finit signature byte array = %s' )", PyString_AsString(pho3) );
        //PyRun_SimpleString( buffer );

        // TODO: this works at present but it might break if the input strings aren't 2 characters long each.
        tempsig_words = (unsigned short *)malloc( n_temps );
        for (i = 0; i < n_temps; i++) {
            tempsig_words[i] = typecode_from_bytecode( PyString_AS_STRING(PySequence_GetItem(tempsig, i)) );
            sprintf( buffer, "print( 'ne_object::finit tempsig[%d] = %d' )", i, tempsig_words[i] );
            PyRun_SimpleString( buffer );
        }

    }

    PyRun_SimpleString( "print('DEBUG: Numexpr finit')" );

    if (o_constants) {
        if (!PySequence_Check(o_constants) ) {
                PyErr_SetString(PyExc_TypeError, "constants must be a sequence");
                return -1;
        }
        n_constants = (int)PySequence_Length(o_constants);
		
        constsig_words = (unsigned short *)malloc( n_constants );
        if (!(constants = PyTuple_New(n_constants)))
            return -1;
        //if (!(constsig = PyByteArray_FromStringAndSize(NULL, n_constants))) {
        //    Py_DECREF(constants);
        //    return -1;
        //}

        
        if (!(itemsizes = PyMem_New(int, n_constants))) {
            Py_DECREF(constants);
            return -1;
        }
        for (i = 0; i < n_constants; i++) {
            PyObject *o;
            if (!(o = PySequence_GetItem(o_constants, i))) { /* new reference */
                Py_DECREF(constants);
                //Py_DECREF(constsig);
                PyMem_Del(itemsizes);
                return -1;
            }
            PyTuple_SET_ITEM(constants, i, o); /* steals reference */
            
            // TODO: add some checks?
            char *dtype = PyString_AsString( PyObject_GetAttrString( PyObject_GetAttrString(o, "__class__"), "__name__" ) );
            sprintf( buffer, "print( '@@@@@@@ class name @@@@@@@ = %s')", dtype );
            PyRun_SimpleString( buffer );
            
            if( strcmp(dtype, "ndarray") == 0 ) {
                PyErr_SetString(PyExc_TypeError, "TODO: account for array scalars" );
                Py_DECREF(constants);
                PyMem_Del(itemsizes);
                return -1;
            }
            else if( strcmp(dtype, "bool") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found boolean constant')" );
                constsig_words[i] = NPY_BOOL;
                itemsizes[i] = size_from_NPYENUM( NPY_BOOL );
            }
            else if( strcmp(dtype, "int32") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found int32 constant')" );
                constsig_words[i] = NPY_INT32;
                itemsizes[i] = size_from_NPYENUM( NPY_INT32 );
            }
            else if( strcmp(dtype, "int64") == 0 || strcmp(dtype, "int") == 0 || strcmp(dtype, "long") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found int64 constant')" );
                constsig_words[i] = NPY_INT64;
                itemsizes[i] = size_from_NPYENUM( NPY_INT64 );
            }
            else if( strcmp(dtype, "float32") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found float32 constant')" );
                constsig_words[i] = NPY_FLOAT32;
                itemsizes[i] = size_from_NPYENUM( NPY_FLOAT32 );
            }
            else if( strcmp(dtype, "float64") == 0 || strcmp(dtype, "float") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found float64 constant')" );
                constsig_words[i] = NPY_FLOAT64;
                itemsizes[i] = size_from_NPYENUM( NPY_FLOAT64 );
            }
            else if( strcmp(dtype, "complex64") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found complex64 constant')" );
                constsig_words[i] = NPY_COMPLEX64;
                itemsizes[i] = size_from_NPYENUM( NPY_COMPLEX64 );
            }
            else if( strcmp(dtype, "complex128") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found complex128 constant')" );
                constsig_words[i] = NPY_COMPLEX128;
                itemsizes[i] = size_from_NPYENUM( NPY_COMPLEX128 );
            }
            else if( strcmp(dtype, "str") == 0 ) {
                PyRun_SimpleString( "print('DEBUG: Found string constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_STRING;
                constsig_words[i] = NPY_STRING;
                itemsizes[i] = (int)PyBytes_GET_SIZE(o);
            }
            //if( PyArray_IsPythonScalar(o)) {
                //if PyInt_Check(o)
                //if PyObject_IsInstance(o, something) {
                //    
                //}
                
            //}
            //else if (PyArray_Check(o)) {
                //PyArray_Descr* DTYPE = PyArray_DTYPE( o );
                //sprintf( buffer, "print( 'ne_object::finit DTYPE.kind = %c, DTYPE.type = %s', elsize = %d )", DTYPE->kind, DTYPE->type, DTYPE->elsize );
                //PyRun_SimpleString( buffer );
    
                //sprintf( buffer, "print( 'ne_object::finit object[%d] type = %d' )", i,  PyArray_TYPE(o) );
                //PyRun_SimpleString( buffer );
            //}

            /*
            if (PyBool_Check(o)) {
                PyRun_SimpleString( "print('DEBUG: Found boolean constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_BOOL;
                constsig_words[i] = NPY_BOOL;
                itemsizes[i] = size_from_NPYENUM( NPY_BOOL );
                continue;
            }
#if PY_MAJOR_VERSION < 3
            // RAM: Ergh!  What is in the numpy API that might be better?
            if (PyInt_Check(o)) {
#else
            if (PyArray_IsScalar(o, Int32)) {
#endif
            //if( PyArray_IsScalar(o, npy_int32)) {
                PyRun_SimpleString( "print('DEBUG: Found int32 constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_INT32;
                constsig_words[i] = NPY_INT32;
                itemsizes[i] = size_from_NPYENUM( NPY_INT32 );
                continue;
            }
#if PY_MAJOR_VERSION < 3
            if (PyLong_Check(o)) {
#else
            if (PyArray_IsScalar(o, Int64)) {
#endif
            //if( PyArray_IsScalar(o, npy_int64)) {
                PyRun_SimpleString( "print('DEBUG: Found int64 constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_INT64;
                constsig_words[i] = NPY_INT64;
                itemsizes[i] = size_from_NPYENUM( NPY_INT64 );
                continue;
            }
            // The Float32 scalars are the only ones that should reach here 
            if (PyArray_IsScalar(o, Float32)) {
                PyRun_SimpleString( "print('DEBUG: Found float32 constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_FLOAT32;
                constsig_words[i] = NPY_FLOAT32;
                itemsizes[i] = size_from_NPYENUM( NPY_FLOAT32 );
                continue;
            }
            if (PyFloat_Check(o)) {
                PyRun_SimpleString( "print('DEBUG: Found float64 constant')" );
                // Python float constants are double precision by default 
                //PyByteArray_AS_STRING(constsig)[i] = NPY_FLOAT64;
                constsig_words[i] = NPY_FLOAT64;
                itemsizes[i] = size_from_NPYENUM( NPY_FLOAT64 );
                continue;
            }
            // NumPy single precision complex number
            if (PyArray_IsScalar(o,CFloat)) {
                PyRun_SimpleString( "print('DEBUG: Found complex64 constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_COMPLEX64;
                constsig_words[i] = NPY_COMPLEX64;
                itemsizes[i] = size_from_NPYENUM( NPY_COMPLEX64 );
                continue;                
            }
            // Python double precision complex number 
            if (PyComplex_Check(o)) {
                PyRun_SimpleString( "print('DEBUG: Found complex128 constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_COMPLEX128;
                constsig_words[i] = NPY_COMPLEX128;
                itemsizes[i] = size_from_NPYENUM( NPY_COMPLEX128 );
                continue;
            }
            if (PyBytes_Check(o)) {
                PyRun_SimpleString( "print('DEBUG: Found string constant')" );
                //PyByteArray_AS_STRING(constsig)[i] = NPY_STRING;
                constsig_words[i] = NPY_STRING;
                itemsizes[i] = (int)PyBytes_GET_SIZE(o);
                continue;
            }
            */
            else {
                PyErr_SetString(PyExc_TypeError, "constants must be of type bool/int16/int32/float32/float64/complex64/complex128/string");
                //Py_DECREF(constsig);
                Py_DECREF(constants);
                PyMem_Del(itemsizes);
                return -1;
            }
        }
    } else {
        n_constants = 0;
        if (!(constants = PyTuple_New(0)))
            return -1;
        //if (!(constsig = PyByteArray_FromStringAndSize("",0))) {
        //    Py_DECREF(constants);
        //    return -1;
        //}
    }



    
    //fullsig = PyBytes_FromFormat("%d%s%s%s", get_return_sig(program_bytes),
     //   PyBytes_AS_STRING(signature), PyBytes_AS_STRING(constsig),
    //    PyBytes_AS_STRING(tempsig));
    // RAM: What is fullsig?  Just a big byte array?  Is it actually used anymore???
    // Instead of the original fullsig, let's convert each word in program_bytes to an unsigned short 
    // in the array program_words
     
    //char buffer[1024];
    //PyObject* pho1 = PyString_AsEncodedObject( PyString_FromString(PyByteArray_AsString(constsig)), "hex", "strict");
    //sprintf( buffer, "print( 'ne_object::finit constsig = %s' )", PyString_AsString(pho1) );
    //PyRun_SimpleString( buffer );
    

    //if (!fullsig) {
    //    PyRun_SimpleString( "print('DEBUG: not fullsig block')" );	
    //    Py_DECREF(constants);
    //    Py_DECREF(constsig);
    //    PyMem_Del(itemsizes);
    //    return -1;
    //}


    if (!input_names) {
        input_names = Py_None;
    }

    // Compute the size of registers. We leave temps out (will be
    //   malloc'ed later on).
    rawmemsize = 0;
    for (i = 0; i < n_constants; i++)
        rawmemsize += itemsizes[i];
    rawmemsize *= BLOCK_SIZE1;

    mem = PyMem_New(char *, 1 + n_inputs + n_constants + n_temps); // RAM: pointer array to address space
    rawmem = PyMem_New(char, rawmemsize); // RAM: The actual address space
    memsteps = PyMem_New(npy_intp, 1 + n_inputs + n_constants + n_temps);
    memsizes = PyMem_New(npy_intp, 1 + n_inputs + n_constants + n_temps);
    if (!mem || !rawmem || !memsteps || !memsizes) {
        Py_DECREF(constants);
        //Py_DECREF(constsig);
        //Py_DECREF(fullsig);
        PyMem_Del(itemsizes);
        PyMem_Del(mem);
        PyMem_Del(rawmem);
        PyMem_Del(memsteps);
        PyMem_Del(memsizes);
        return -1;
    }


    // RAM: Where the heck is output and input allocated???  In PyMem_New(char, rawmemsize)...
    //  0                                                  -> output
    //   [1, n_inputs+1)                                    -> inputs
    //   [n_inputs+1, n_inputs+n_consts+1)                  -> constants
    //   [n_inputs+n_consts+1, n_inputs+n_consts+n_temps+1) -> temps
    
    /* Fill in 'mem' and 'rawmem' for constants */
    
    mem_offset = 0;
    for (i = 0; i < n_constants; i++) {
        // RAM: This is an NPY_ENUMTYPE now
        //unsigned short dtype = PyByteArray_AS_STRING(constsig)[i]; 
        unsigned short dtype = constsig_words[i];
        int size = itemsizes[i];
        mem[i+n_inputs+1] = rawmem + mem_offset;
        mem_offset += BLOCK_SIZE1 * size;
        
        memsteps[i+n_inputs+1] = memsizes[i+n_inputs+1] = size;
        sprintf( buffer, "print( 'ne_object::finit sizes = %d' )", size );
        PyRun_SimpleString( buffer );


        /* fill in the constants */
        // RAM: why is this not a switch-case?
        if (dtype == NPY_BOOL) {
            npy_bool *bmem = (npy_bool*)mem[i+n_inputs+1];
            npy_bool value = (npy_bool)PyLong_AsLong(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                bmem[j] = value;
            }
        } else if (dtype == NPY_INT32) {
            npy_int32 *imem = (npy_int32*)mem[i+n_inputs+1];
            npy_int32 value = (npy_int32)PyLong_AsLong(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                imem[j] = value;
            }
        } else if (dtype == NPY_INT64) {
            npy_int64 *lmem = (npy_int64*)mem[i+n_inputs+1];
            npy_int64 value = (npy_int64)PyLong_AsLongLong(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                lmem[j] = value;
            }
        } else if (dtype == NPY_FLOAT32) {
            /* In this particular case the constant is in a NumPy scalar
             and in a regular Python object */
            npy_float32 *fmem = (npy_float32*)mem[i+n_inputs+1];
            npy_float32 value = (npy_float32)PyArrayScalar_VAL(PyTuple_GET_ITEM(constants, i),
                                            Float);
            for (j = 0; j < BLOCK_SIZE1; j++) {
                fmem[j] = value;
            }
        } else if (dtype == NPY_FLOAT64) {
            npy_float64 *dmem = (npy_float64*)mem[i+n_inputs+1];
            npy_float64 value = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                dmem[j] = value;
            }
        } else if (dtype == NPY_COMPLEX64) {
            /* In this particular case the constant is in a NumPy scalar
             and in a regular Python object */
            npy_float32 *fmem = (npy_float32*)mem[i+n_inputs+1];
            npy_complex64 value = PyArrayScalar_VAL(PyTuple_GET_ITEM(constants, i),
                                            CFloat);
            // TODO: check if complex64 constants work correctly,
            for (j = 0; j < 2*BLOCK_SIZE1; j+=2) {
                fmem[j] = value.real;
                fmem[j+1] = value.imag;
            }
        } else if (dtype == NPY_COMPLEX128) {
            npy_float64 *cmem = (npy_float64*)mem[i+n_inputs+1];
            Py_complex value = PyComplex_AsCComplex(PyTuple_GET_ITEM(constants, i));
            // TODO: check if complex128 constants work correctly,
            for (j = 0; j < 2*BLOCK_SIZE1; j+=2) {
                cmem[j] = value.real;
                cmem[j+1] = value.imag;
            }
        } else if (dtype == NPY_STRING) {
            char *smem = (char*)mem[i+n_inputs+1];
            char *value = PyBytes_AS_STRING(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < size*BLOCK_SIZE1; j+=size) {
                memcpy(smem + j, value, size);
            }
        } else {
                char buffer[256];
                sprintf( buffer, "print( 'ne_object finit, Unidentified constant  n_constant = %d, of numpy_type constsig = x%d' )", i, dtype );
                PyRun_SimpleString( buffer );
        }

    }
    /* This is no longer needed since no unusual item sizes appear
       in temporaries (there are no string temporaries). */
    PyMem_Del(itemsizes);

    /* Fill in 'memsteps' and 'memsizes' for temps */
    //PyObject* pho2 = PyString_AsEncodedObject( PyString_FromString(PyByteArray_AsString(tempsig)), "hex", "strict");
    //sprintf( buffer, "print( 'ne_object::finit tempsig = %s' )", PyString_AsString(pho2) );
    //PyRun_SimpleString( buffer );

    for (i = 0; i < n_temps; i++) {
        //PyRun_SimpleString( "print('TODO: fix tempsig code block')" );	

        //char c = PyBytes_AS_STRING(tempsig)[i];
        //unsigned short dtype = tempsig_words[i];

        int dsize = size_from_NPYENUM(tempsig_words[i]);

        memsteps[i+n_inputs+n_constants+1] = dsize;
        memsizes[i+n_inputs+n_constants+1] = dsize;
    }
    /* See if any errors occured (e.g., in size_from_char) or if mem_offset is wrong */
    if (PyErr_Occurred() || mem_offset != rawmemsize) {
        if (mem_offset != rawmemsize) {
            PyErr_Format(PyExc_RuntimeError, "mem_offset does not match rawmemsize");
        }
        Py_DECREF(constants);
        //Py_DECREF(constsig);
        //Py_DECREF(fullsig);
        PyMem_Del(mem);
        PyMem_Del(rawmem);
        PyMem_Del(memsteps);
        PyMem_Del(memsizes);
        return -1;
    }

    // RAM: Let's figure out how to diagnose just what memsteps and memsizes are
    for( int j = 0; j < 1 + n_inputs + n_constants + n_temps; j++ )
    {
        sprintf( buffer, "print( 'ne_object memory, memsteps[%d]=%d, memsizes[%d]=%d' )", j, memsteps[j], j, memsizes[j] );
        PyRun_SimpleString( buffer );
    }


    #define REPLACE_OBJ(arg) \
    {PyObject *tmp = self->arg; \
     self->arg = arg; \
     Py_XDECREF(tmp);}
    #define INCREF_REPLACE_OBJ(arg) {Py_INCREF(arg); REPLACE_OBJ(arg);}
    #define REPLACE_MEM(arg) {PyMem_Del(self->arg); self->arg=arg;}

    //INCREF_REPLACE_OBJ(signature);
    //INCREF_REPLACE_OBJ(tempsig);
    //REPLACE_OBJ(constsig);
    //REPLACE_OBJ(fullsig);
    INCREF_REPLACE_OBJ(program_bytes);
    REPLACE_OBJ(constants);
    INCREF_REPLACE_OBJ(input_names);
    REPLACE_MEM(mem);
    REPLACE_MEM(rawmem);
    REPLACE_MEM(memsteps);
    REPLACE_MEM(memsizes);
    self->rawmemsize = rawmemsize;
    self->n_inputs = n_inputs;
    self->n_constants = n_constants;
    self->n_temps = n_temps;
    self->sig_words = sig_words;
    self->tempsig_words = tempsig_words;
    self->constsig_words = constsig_words;

    #undef REPLACE_OBJ
    #undef INCREF_REPLACE_OBJ
    #undef REPLACE_MEM

    return check_program(self);
}

static PyMethodDef NumExpr_methods[] = {
    {"run", (PyCFunction) NumExpr_run, METH_VARARGS|METH_KEYWORDS, NULL},
    {NULL, NULL}
};

static PyMemberDef NumExpr_members[] = {
    {"signature", T_OBJECT_EX, offsetof(NumExprObject, signature), READONLY, NULL},
    {"constsig", T_OBJECT_EX, offsetof(NumExprObject, constsig), READONLY, NULL},
    {"tempsig", T_OBJECT_EX, offsetof(NumExprObject, tempsig), READONLY, NULL},
    //{"fullsig", T_OBJECT_EX, offsetof(NumExprObject, fullsig), READONLY, NULL},

    {"program_bytes", T_OBJECT_EX, offsetof(NumExprObject, program_bytes), READONLY, NULL},
    {"constants", T_OBJECT_EX, offsetof(NumExprObject, constants),
     READONLY, NULL},
    {"input_names", T_OBJECT, offsetof(NumExprObject, input_names), 0, NULL},
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

