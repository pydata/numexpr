/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: BSD
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "module.hpp"
#include <numpy/npy_cpu.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <vector>

#include "numexpr_config.hpp"
#include "complex_functions.hpp"
#include "string_functions.hpp"
#include "interp_header_GENERATED.hpp"
#include "numexpr_object.hpp"
#include "functions_GENERATED.cpp"

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif


using namespace std;

// Global state
thread_data th_params;


// returns the sig of the nth op, '\0' if no more ops -1 on failure 
//static int
//op_signature(int op, NE_WORD n) {
//    if (n >= NUMEXPR_MAX_ARGS) {
//        return 0;
//    }
//    if (op < 0 || op > OP_END) {
//        return -1;
//    }
//    // RAM: this should be the integer direct now.
//    return op;
//}

int
NPYENUM_from_dchar(char c)
{
    switch (c) {
        case '?': return NPY_BOOL;
        case 'b': return NPY_INT8;
        case 'B': return NPY_UINT8;
        case 'h': return NPY_INT16;
        case 'H': return NPY_UINT16;
        case 'i': return NPY_INT32;
        case 'I': return NPY_UINT32;
        case 'l': return NPY_INT64;
        case 'L': return NPY_UINT64;
        case 'f': return NPY_FLOAT32;
        case 'F': return NPY_COMPLEX64;
        case 'd': return NPY_FLOAT64;
        case 'D': return NPY_COMPLEX128;
        case 'S': return NPY_STRING;
        case 'U': return NPY_UNICODE;
        default:
            PyErr_Format(PyExc_TypeError,
                                "signature char '%c' not in '?bBhHiIlLdDfFSU'", c);
            return -1;
    }
}

char
get_return_sig(NumExprObject *self)
{
    int last_reg = self->program[self->n_reg - 1].ret;
    return self->registers[last_reg].dchar;
}



// TODO: refactor get_reduction_axis
static int
get_reduction_axis(NumExprObject *self) {
    
//    Py_ssize_t end = PyBytes_Size(program);
//    int axis = ((unsigned char *)PyBytes_AS_STRING(program))[end-1];
//    if (axis != 255 && axis >= NPY_MAXDIMS)
//        axis = NPY_MAXDIMS - axis;
//    return axis;
    return -1;
}


// Get space for VM temporary registers
int 
get_temps_space(NumExprObject *self, size_t block_size)
{
    int R;
    for ( R = 0; R < self->n_reg; R++) {
        if( self->registers[R].kind == KIND_TEMP ) {
            // GIL _not_ released here.
            // RAM: Why not move this inside the threads and use PyMem_RawMalloc?
            self->registers[R].mem = (char *)PyMem_Malloc( block_size * self->registers[R].itemsize );
            if ( self->registers[R].mem == NULL) {
                return -1;
            }
        }
    }
    return 0;
}

// Free space for VM temporary registers
void 
free_temps_space(const NumExprObject *self)
{
    int R;
    for( R = 0; R < self->n_reg; R++ ) {
        if( self->registers[R].kind == KIND_TEMP ) {
            free( self->registers[R].mem );
        }
    }
}

// module.cpp also needs access to copy_threadsafe
NumExprObject* 
NumExprObject_copy_threadsafe( const NumExprObject *self )
{
    int R;
    // There's a bit of wasted effort in copying all the registers compared 
    // to the NE2 mode of simply copying **mem and *memsteps but this is much
    // better encapsulated.  Micro-optimizations can wait.
    // TODO: keep an array of pre-built NumExprObjects in an array in the 
    // global_state???  Would be more efficient than excessive construction/
    // destruction.
    
    //NumExprObject *copy = (NumExprObj*)malloc( sizeof(NumExprObject) );
    // NumExprObject is a Python object, althought perhaps it shouldn't be.  
    // If we removed the array references in registers?
    NumExprObject *copy = PyMem_New( NumExprObject, 1 );
    // PyMem_New pops a nasty warning.  Probably I can't use memcpy with PyObjects?
    // program is immutable, but registers is not (especially the mem pointers)                
    memcpy( copy, self, sizeof(NumExprObject) );
    //*copy = *self;
    
    // RAM: is 
    copy->registers = (NumExprReg *)PyMem_Malloc( self->n_reg*sizeof(NumExprReg) );
    
    //copy->registers = PyMem_New( NumExprReg, self->n_reg );
    for( R = 0; R < self->n_reg; R++ ) {
        copy->registers[R] = self->registers[R];
        //printf( "self.mem: %p, copy.mem: %p\n", self->registers[R].mem, copy->registers[R].mem );
    }
    
//    for( int I = 0; I < self->program_len; I++ ) {
//        printf( "self->program: %p, copy->program: %p\n", self->program, copy->program );
//        printf( "self::[%d]: %d %d %d %d %d \n", I,
//            (int)self->program[I].op, (int)self->program[I].ret, (int)self->program[I].arg1, 
//            (int)self->program[I].arg2, (int)self->program[I].arg3 );
//        printf( "copy::[%d]: %d %d %d %d %d \n", I,
//            (int)copy->program[I].op, (int)copy->program[I].ret, (int)copy->program[I].arg1, 
//            (int)copy->program[I].arg2, (int)copy->program[I].arg3 );
//    }

    return copy;
}

// Serial/parallel task iterator version of the VM engine
int vm_engine_iter_task(NpyIter *iter, 
                    const NumExprObject *params,
                    int *pc_error, char **errorMessage)
{
    NpyIter_IterNextFunc *iterNext;
    npy_intp block_size, *sizePtr;
    char **iterDataPtr;
    npy_intp *iterStrides;

    iterNext = NpyIter_GetIterNext(iter, errorMessage);
    if (iterNext == NULL) {
        return -1;
    }

    // Ok, here we have a problem.  We rely on iterDataPtr but it's not ordered 
    // correctly for temporaries (and probably consts).  I guess the easy 
    // solution is to iterate through params->registers[arrayCount].mem = iterDataPtr[I]???
    // and strides->registers[arrayCount].mem = iterDataPtr[I]???
    // This is turning into more overhead than arranging everything beforehand 
    // in Python so registers are ordered 
    sizePtr = NpyIter_GetInnerLoopSizePtr(iter);
    iterDataPtr = NpyIter_GetDataPtrArray(iter);
    iterStrides = NpyIter_GetInnerStrideArray(iter);
            
//    //DEBUG
//    for( int I = 0; I < params->program_len; I++ ) {                                         
//        printf( "params[%d]:: r:%d a1:%d a1:%d a2:%d a3:%d \n", I,
//            (int)params->program[I].op, (int)params->program[I].ret, (int)params->program[I].arg1, 
//            (int)params->program[I].arg2, (int)params->program[I].arg3 );
//    }
//    for( int I = 0; I < params->n_reg; I++ ) {
//        printf( "regs[%d]:: kind:%d, mem:%p, \n", I, params->registers[I].kind, params->registers[I].mem  );
//        
//    }
//    // Seems like this iterDataPtr array isn't fully allocated most of the time?
//    for( int I = 0; I < params->n_reg; I++ ) {
//        if( params->registers[I].kind == KIND_ARRAY ) {
//            printf( "iterDataPtr[%d]:: %p, iterStrides[%d]:: %p \n", I, iterDataPtr[I], I, (void *)iterStrides[I] );
//        }
//        
//    }

    // First do all the blocks with a compile-time fixed size. This makes a 
    // big difference (30-50% on some tests).
    // TODO: RAM, this can be replaced in the generator with a fixed size in 
    // _bytes_ instead of _elements_

    block_size = *sizePtr;
    // RAM: let's try having a variable block size?
    // Success, with auto-vectorization it doesn't need to be a fixed size, 
    // compared to unrolling loops. Looks like we can cut-down the number of 
    // includes which will shrink the machine code.
    while( block_size > 0 ) {
#define REDUCTION_INNER_LOOP            
#include "interp_body_GENERATED.cpp"
#undef REDUCTION_INNER_LOOP
        iterNext(iter);
        block_size = *sizePtr;   
    }
    
//    while (block_size == BLOCK_SIZE1) {
//        //printf( "vm_iter_engine run block.\n" );
//#define REDUCTION_INNER_LOOP
//#define BLOCK_SIZE BLOCK_SIZE1
//#include "interp_body_GENERATED.cpp"
//#undef BLOCK_SIZE
//#undef REDUCTION_INNER_LOOP
//        iterNext(iter);
//        block_size = *sizePtr;
//    }
//
//    /* Then finish off the rest */
//    if (block_size > 0) do {
//#define REDUCTION_INNER_LOOP
//#define BLOCK_SIZE block_size
//#include "interp_body_GENERATED.cpp"
//#undef BLOCK_SIZE
//#undef REDUCTION_INNER_LOOP
//    } while (iterNext(iter));

    return 0;
}

static int
vm_engine_iter_outer_reduce_task(NpyIter *iter, 
                const NumExprObject *params, int *pc_error, char **errorMessage)
{
    NpyIter_IterNextFunc *iterNext;
    npy_intp block_size, *sizePtr;
    char **iterDataPtr;
    npy_intp *iterStrides;

    iterNext = NpyIter_GetIterNext(iter, errorMessage);
    if (iterNext == NULL) {
        return -1;
    }

    sizePtr = NpyIter_GetInnerLoopSizePtr(iter);
    iterDataPtr = NpyIter_GetDataPtrArray(iter);
    iterStrides = NpyIter_GetInnerStrideArray(iter);


     // First do all the blocks with a compile-time fixed size.
     // This makes a big difference (30-50% on some tests).
    block_size = *sizePtr;
    while( block_size > 0 ) {
#define NO_OUTPUT_BUFFERING          
#include "interp_body_GENERATED.cpp"
#undef NO_OUTPUT_BUFFERING
        iterNext(iter);
        block_size = *sizePtr;   
    }
    
    
//    while (block_size == BLOCK_SIZE1) {
//#define BLOCK_SIZE BLOCK_SIZE1
//#define NO_OUTPUT_BUFFERING // Because it's a reduction
//#include "interp_body_GENERATED.cpp"
//#undef NO_OUTPUT_BUFFERING
//#undef BLOCK_SIZE
//        iterNext(iter);
//        block_size = *sizePtr;
//    }
//
//    // Then finish off the rest 
//    if (block_size > 0) do {
//#define BLOCK_SIZE block_size
//#define NO_OUTPUT_BUFFERING // Because it's a reduction
//#include "interp_body_GENERATED.cpp"
//#undef NO_OUTPUT_BUFFERING
//#undef BLOCK_SIZE
//    } while (iterNext(iter));
    
    return 0;
}

// Parallel iterator version of VM engine 
static int
vm_engine_iter_parallel(NpyIter *iter, const NumExprObject *params,
                        bool need_output_buffering, int *pc_error,
                        char **errorMessage)
{
     
    int i;
    npy_intp numblocks, taskfactor;

    if (errorMessage == NULL) {
        return -1;
    }

    // Populate parameters for worker threads 
    NpyIter_GetIterIndexRange(iter, &th_params.start, &th_params.vlen);
    
    // Try to make it so each thread gets 16 tasks.  This is a compromise
    // between 1 task per thread and one block per task.

    taskfactor = TASKS_PER_THREAD*BLOCK_SIZE1*gs.n_thread;
    numblocks = (th_params.vlen - th_params.start + taskfactor - 1) /
                            taskfactor;
    th_params.block_size = numblocks * BLOCK_SIZE1;

    th_params.params = NumExprObject_copy_threadsafe(params);
    th_params.need_output_buffering = need_output_buffering;
    th_params.ret_code = 0;
    th_params.pc_error = pc_error;
    th_params.errorMessage = errorMessage;
    th_params.iter[0] = iter;
    // Make one copy for each additional thread
    for (i = 1; i < gs.n_thread; ++i) {
        th_params.iter[i] = NpyIter_Copy(iter);
        if (th_params.iter[i] == NULL) {
            --i;
            for (; i > 0; --i) {
                NpyIter_Deallocate(th_params.iter[i]);
            }
            return -1;
        }
    }
    
    Py_BEGIN_ALLOW_THREADS;

    // Synchronization point for all threads (wait for initialization)
    pthread_mutex_lock(&gs.count_threads_mutex);
    if (gs.count_threads < gs.n_thread) {
        gs.count_threads++;
        pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
    }
    else {
        pthread_cond_broadcast(&gs.count_threads_cv);
    }
    pthread_mutex_unlock(&gs.count_threads_mutex);

    // Synchronization point for all threads (wait for finalization)
    pthread_mutex_lock(&gs.count_threads_mutex);
    if (gs.count_threads > 0) {
        gs.count_threads--;
        pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
    }
    else {
        pthread_cond_broadcast(&gs.count_threads_cv);
    }
    pthread_mutex_unlock(&gs.count_threads_mutex);

    Py_END_ALLOW_THREADS;

    // Deallocate all the iterator and memsteps copies
    for (i = 1; i < gs.n_thread; ++i) {
        NpyIter_Deallocate(th_params.iter[i]);
        //PyMem_Del(th_params.memsteps[i]);
    }
    
    return th_params.ret_code;
    
    return 0;
}

static int
run_interpreter(NumExprObject *self, NpyIter *iter, NpyIter *reduce_iter,
                     bool reduction_outer_loop, bool need_output_buffering,
                     int *pc_error)
{
        
    
    int returnValue;
    //Py_ssize_t plen;
    // RAM: Why do we need a seperate NumExprObject now???
    // For now we'll call this 'safeParams' but I'm not clear that it's required.
    NumExprObject *safeParams = NULL;
    char *errorMessage = NULL;
    
//    for( int I = 0; I < self->program_len; I++ ) {
//        printf( "run_interpreter::Program[%d]: %d %d %d %d %d \n", I,
//            (int)self->program[I].op, (int)self->program[I].ret, (int)self->program[I].arg1, 
//            (int)self->program[I].arg2, (int)self->program[I].arg3 );
//    }

                      
    *pc_error = -1;
    
    // RAM: we have self->program_len now
//    if (PyBytes_AsStringAndSize(self->program_bytes, (char **)&(params.program),
//                                &plen) < 0) {
//        return -1;
//    }

    //params.prog_len = self->program_len;
    //params.output = NULL;
    //params.inputs = NULL;
    //params.n_ndarray = self->n_ndarray;
    //params.n_scalar = self->n_scalar;
    //params.n_temp = self->n_temp;
    // Here we have a bit of pain, because params doesn't work directly from
    // program and registers.
    //params.mem = self->mem;
    //params.memsteps = self->memsteps;
    //params.memsizes = self->memsizes;
    //params.r_end = (int)PyBytes_Size(self->fullsig);
    //params.outBuffer = NULL;
    
                      
    if ((gs.n_thread == 1) || gs.force_serial) {
        // Can do it as one "task"
        if (reduce_iter == NULL) {
            // Allocate memory for output buffering if needed
//            vector<char> out_buffer(need_output_buffering ?
//                                (self->memsizes[0] * BLOCK_SIZE1) : 0);
//            params.out_buffer = need_output_buffering ? &out_buffer[0] : NULL;
                                      

            //printf( "run_interpreter() #2\n" ); 

            safeParams = NumExprObject_copy_threadsafe( self );
                                                      
            //printf( "run_interpreter() #3\n" ); 
                      
//            if( need_output_buffering ) {
            vector<char> out_buffer(need_output_buffering ?
                                (GET_RETURN_REG(self).itemsize * BLOCK_SIZE1) : 0);
            safeParams->outBuffer = (char *)(need_output_buffering ? &out_buffer[0] : NULL );
//            }
            // Reset the iterator to allocate its buffers
            if(NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
                return -1;
            }
            
            get_temps_space( safeParams, BLOCK_SIZE1);
            Py_BEGIN_ALLOW_THREADS;     
            returnValue = vm_engine_iter_task(iter, safeParams, pc_error, &errorMessage);
            Py_END_ALLOW_THREADS;
            free_temps_space(safeParams);
        }
        else {
            if (reduction_outer_loop) {
                char **dataPtr;
                NpyIter_IterNextFunc *iterNext;

                dataPtr = NpyIter_GetDataPtrArray(reduce_iter);
                iterNext = NpyIter_GetIterNext(reduce_iter, NULL);
                if (iterNext == NULL) {
                    return -1;
                }
                
                safeParams = NumExprObject_copy_threadsafe( self );                                    
                get_temps_space( safeParams, BLOCK_SIZE1);
                Py_BEGIN_ALLOW_THREADS;
                do {
                    returnValue = NpyIter_ResetBasePointers(iter, dataPtr, &errorMessage);
                    if (returnValue >= 0) {
                        returnValue = vm_engine_iter_outer_reduce_task(iter,
                                                safeParams,
                                                pc_error, &errorMessage);
                    }
                    if (returnValue < 0) {
                        break;
                    }
                } while (iterNext(reduce_iter));
                Py_END_ALLOW_THREADS;
                free_temps_space(safeParams);
            }
            else {
                char **dataPtr;
                NpyIter_IterNextFunc *iterNext;

                dataPtr = NpyIter_GetDataPtrArray(iter);
                iterNext = NpyIter_GetIterNext(iter, NULL);
                if (iterNext == NULL) {
                    return -1;
                }

                safeParams = NumExprObject_copy_threadsafe( self );                                    
                get_temps_space(safeParams, BLOCK_SIZE1);
                Py_BEGIN_ALLOW_THREADS;
                do {
                    returnValue = NpyIter_ResetBasePointers(reduce_iter, dataPtr,
                                                                    &errorMessage);
                    if (returnValue >= 0) {
                        returnValue = vm_engine_iter_task(reduce_iter, 
                                                safeParams, pc_error, &errorMessage);
                    }
                    if (returnValue < 0) {
                        break;
                    }
                } while (iterNext(iter));
                Py_END_ALLOW_THREADS;
                free_temps_space(safeParams);
            }
        }
    }
    else {
        
        if (reduce_iter == NULL) {
            // Here the parallel engine will make save copies of the NumExprObject
            returnValue = vm_engine_iter_parallel(iter, self, need_output_buffering,
                        pc_error, &errorMessage);
        }
        else {
            errorMessage = CHARP("Parallel engine doesn't support reduction.\n");
            returnValue = -1;
        }
    }

    if (returnValue < 0 && errorMessage != NULL) {
        PyErr_SetString(PyExc_RuntimeError, errorMessage);
    }
    
    return 0;
}

static int
run_interpreter_const(NumExprObject *self, char *output, int *pc_error)
{
     
    NumExprObject *params;
    //Py_ssize_t plen;
    //char **mem;
    //npy_intp *memsteps;

    //*pc_error = -1;
    // RAM: program isn't bytes anymore.
//    if (PyBytes_AsStringAndSize(self->program, (char **)&(params.program),
//                                &plen) < 0) {
//        return -1;
//    }
    if (self->n_ndarray != 0) {
        return -1;
    }

    params = NumExprObject_copy_threadsafe( self );
    get_temps_space(params, 1);
#define SINGLE_ITEM_CONST_LOOP
#define block_size 1
#define NO_OUTPUT_BUFFERING // Because it's constant
#include "interp_body_GENERATED.cpp"
#undef NO_OUTPUT_BUFFERING
#undef block_size
#undef SINGLE_ITEM_CONST_LOOP
    free_temps_space(params);
    
    return 0;
}

PyObject *
NumExpr_run(NumExprObject *self, PyObject *args, PyObject *kwds)
{
 
    PyArrayObject *operands[NPY_MAXARGS];
    PyArray_Descr *dtypes[NPY_MAXARGS];
    PyObject *tmp, *ret;
    PyObject *objectRef, *arrayRef;
    npy_uint32 op_flags[NPY_MAXARGS];
    NPY_CASTING casting = NPY_SAFE_CASTING;
    NPY_ORDER order = NPY_KEEPORDER;
    npy_uint16 I, n_input;
    int r, pc_error = 0;
    int reduction_axis = -1;
    npy_intp reduction_size = 1;
    int ex_uses_vml = 0, is_reduction = 0;
    int arrayCounter = 0;
    bool reduction_outer_loop = false, need_output_buffering = false;
    
    // NumPy iterators for strided arrays
    NpyIter *iter = NULL, *reduce_iter = NULL;
    
    // To specify axes when doing a reduction
    int op_axes_values[NPY_MAXARGS][NPY_MAXDIMS],
         op_axes_reduction_values[NPY_MAXARGS];
    int *op_axes_ptrs[NPY_MAXDIMS];
    int oa_ndim = 0;
    int **op_axes = NULL;

    

    // Check whether we need to restart threads
    if (!gs.init_threads_done || gs.pid != getpid()) {
        numexpr_set_nthreads(gs.n_thread);
    }
                      
    // Don't force serial mode by default
    gs.force_serial = 0;

    // Check whether there's a reduction as the final step
    // RAM: this is clumsy and we'll remove it later, probably by adding 
    // flags to the front of the program.
   
    is_reduction = LAST_OP(self) >= OP_REDUCTION;

    n_input = (int)PyTuple_Size(args);
    // RAM: Ok, here we have to check how many arguments we need, which is 
    // n_reg - n_temp - n_scalar
    
    //printf( "NumExpr_run() #3\n" );
                      
    if ( self->n_ndarray != n_input) {
        return PyErr_Format(PyExc_ValueError,
                "NumExpr_run(): number of inputs %d doesn't match program ndarrays %d", n_input, self->n_ndarray );
    }
    else if (n_input >= NPY_MAXARGS) {
        // This is related to the NumPy limit of 32 arguments in an nditer
        return PyErr_Format(PyExc_ValueError,
                        "NumExpr_run(): too many inputs");
    }
    
    //printf( "NumExpr_run() #4\n" ); 
    memset(operands, 0, sizeof(operands));
    memset(dtypes, 0, sizeof(dtypes));

    if (kwds) {
            
//        tmp = PyDict_GetItemString(kwds, "out");
//        if (tmp == NULL) {
//            return PyErr_Format(PyExc_ValueError,
//                                "out keyword argument is required");
//        }
//        if (tmp == Py_None ) {
//            printf( "Found None for output array." );
//        }
                          
        tmp = PyDict_GetItemString(kwds, "casting"); // borrowed ref
        if (tmp != NULL && !PyArray_CastingConverter(tmp, &casting)) {
            return NULL;
        }
        tmp = PyDict_GetItemString(kwds, "order"); // borrowed ref
        if (tmp != NULL && !PyArray_OrderConverter(tmp, &order)) {
            return NULL;
        }
        // VML can be mixed-and-matched with STD now.
        
        tmp = PyDict_GetItemString(kwds, "ex_uses_vml"); // borrowed ref
        if (tmp == NULL) {
            return PyErr_Format(PyExc_ValueError,
                                "ex_uses_vml parameter is required");
        } 
        if (tmp == Py_True) {
            ex_uses_vml = 1;
        }
        
        // borrowed reference
        // RAM: any allocation for 'out' is done Python-side now.  It should 
        // always be in register 0.
        // Or do we want to do it here?
        
//        operands[0] = (PyArrayObject *)PyDict_GetItemString(kwds, "out");
//        if (operands[0] != NULL) {
//            if ((PyObject *)operands[0] == Py_None) {
//                operands[0] = NULL;
//            }
//            else if (!PyArray_Check(operands[0])) {
//                return PyErr_Format(PyExc_ValueError,
//                                    "out keyword parameter is not an array");
//            }
//            else {
//                Py_INCREF(operands[0]);
//            }
//       }
    }
    //printf( "NumExpr_run() #5\n" ); 
                      
    // Run through all the registers and match the input arguments to that
    // parsed by NumExpr_init
    for (I = 0; I < self->n_reg; I++) {
        
        if( arrayCounter > n_input ) { goto fail; }
        if( self->registers[I].kind == KIND_SCALAR 
           || self->registers[I].kind == KIND_TEMP ) {
           continue;
        }
        
        objectRef = PyTuple_GET_ITEM(args, arrayCounter); // borrowed ref

        //char c = PyBytes_AS_STRING(self->signature)[i];
        int typecode = NPYENUM_from_dchar( self->registers[I].dchar );
        // Convert it if it's not an array
        if (!PyArray_Check(objectRef)) {
            if (typecode == -1) { goto fail; }
            arrayRef = PyArray_FROM_OTF(objectRef, typecode, NPY_ARRAY_NOTSWAPPED);
        }
        else {
            Py_INCREF(objectRef);
            arrayRef = objectRef;
        }
        //printf( "Found array #%d at register %d\n", arrayCounter, I );
        operands[arrayCounter] = (PyArrayObject *)arrayRef;
        dtypes[arrayCounter] = PyArray_DescrFromType(typecode);

        if (operands[arrayCounter] == NULL || dtypes[arrayCounter] == NULL) { goto fail; }
        
        // RAM: I think this is still ok..., but we should not compare [0] to [0]
        // operands[0] should never be NULL now.
        // Also we don't need output buffering if there's enough operations between
        // the output and 'output as an input'
        if ( arrayCounter > 0 && operands[0] != NULL) {
            // Check for the case where "out" is one of the inputs
            // TODO: Probably should deal with the general overlap case,
            //       but NumPy ufuncs don't do that yet either.
            if (PyArray_DATA(operands[0]) == PyArray_DATA(operands[arrayCounter])) {
                printf("DEBUG: output buffering requested.\n");
                need_output_buffering = true;
            }
        }
        


        op_flags[arrayCounter] = NPY_ITER_READONLY|
#ifdef USE_VML
                        (ex_uses_vml ? (NPY_ITER_CONTIG|NPY_ITER_ALIGNED) : 0)|
#endif
#ifndef USE_UNALIGNED_ACCESS
                        NPY_ITER_ALIGNED|
#endif
                        NPY_ITER_NBO;
        
        arrayCounter++;
    }
            
            
    //printf( "NumExpr_run() #6\n" ); 
    // Ne2: This built the output array.
    // Ne3: We prefer to do this in Python, as multi-line exec is permitted.
    /*
    if (is_reduction) {
        // A reduction can not result in a string,
        // so we don't need to worry about item sizes here.
        char retsig = get_return_sig(self);
        reduction_axis = get_reduction_axis(self);

        // Need to set up op_axes for the non-reduction part
        if (reduction_axis != 255) {
            // Get the number of broadcast dimensions
            for (I = 0; I < n_input; ++I) {
                int ndim = PyArray_NDIM(operands[I]);
                if (ndim > oa_ndim) {
                    oa_ndim = ndim;
                }
            }
            if (reduction_axis < 0 || reduction_axis >= oa_ndim) {
                PyErr_Format(PyExc_ValueError,
                        "reduction axis is out of bounds");
                goto fail;
            }
            // Fill in the op_axes
            op_axes_ptrs[0] = NULL;
            op_axes_reduction_values[0] = -1;
            for (I = 0; I < n_input; ++I) {
                int J = 0, idim, ndim = PyArray_NDIM(operands[I]);
                for (idim = 0; idim < oa_ndim-ndim; ++idim) {
                    if (idim != reduction_axis) {
                        op_axes_values[I][J++] = -1;
                    }
                    else {
                        op_axes_reduction_values[I] = -1;
                    }
                }
                for (idim = oa_ndim-ndim; idim < oa_ndim; ++idim) {
                    if (idim != reduction_axis) {
                        op_axes_values[I][J++] = idim-(oa_ndim-ndim);
                    }
                    else {
                        npy_intp size = PyArray_DIM(operands[I],
                                                    idim-(oa_ndim-ndim));
                        if (size > reduction_size) {
                            reduction_size = size;
                        }
                        op_axes_reduction_values[I] = idim-(oa_ndim-ndim);
                    }
                }
                op_axes_ptrs[I] = op_axes_values[I];
            }
            // op_axes has one less than the broadcast dimensions
            --oa_ndim;
            if (oa_ndim > 0) {
                op_axes = op_axes_ptrs;
            }
            else {
                reduction_size = 1;
            }
        }
        // A full reduction can be done without nested iteration
        if (oa_ndim == 0) {
            if (operands[0] == NULL) {
                npy_intp dim = 1;
                operands[0] = (PyArrayObject *)PyArray_SimpleNew(0, &dim,
                                            NPYENUM_from_dchar(retsig));
                if (!operands[0])
                    goto fail;
            } else if (PyArray_SIZE(operands[0]) != 1) {
                PyErr_Format(PyExc_ValueError,
                        "argument 'out' must have size 1 for a full reduction");
                goto fail;
            }
        }

        dtypes[0] = PyArray_DescrFromType(NPYENUM_from_dchar(retsig));

        op_flags[0] = NPY_ITER_READWRITE|
                      NPY_ITER_ALLOCATE|
                      // Copy, because it can't buffer the reduction
                      NPY_ITER_UPDATEIFCOPY|
                      NPY_ITER_NBO|
#ifndef USE_UNALIGNED_ACCESS
                      NPY_ITER_ALIGNED|
#endif
                      (oa_ndim == 0 ? 0 : NPY_ITER_NO_BROADCAST);
    }
    else { // Not a reduction
        char retsig = get_return_sig(self);
        if (retsig != 'S' and retsig != 'U') {
            dtypes[0] = PyArray_DescrFromType(NPYENUM_from_dchar(retsig));
        } else {
            // Since the *only* supported operation returning a string
            // is a copy, the size of returned strings
            // can be directly gotten from the first (and only)
            // input/constant/temporary.
            if (n_input > 0) {  // input, like in 'a' where a -> 'foo'
                dtypes[0] = PyArray_DESCR(operands[0]);
                Py_INCREF(dtypes[0]);
            } else {  // constant, like in '"foo"'
                dtypes[0] = PyArray_DescrNewFromType(NPY_STRING);
                //dtypes[0]->elsize = (int)self->memsizes[1];
                // RAM: this is a strange way to lookup itemsize
                dtypes[0]->elsize = self->registers[ (int)self->program[0].ret ].itemsize;
            }  // no string temporaries, so no third case
        }
        if (dtypes[0] == NULL) {
            goto fail;
        }
        op_flags[0] = NPY_ITER_WRITEONLY|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_CONTIG|
                      NPY_ITER_NBO|
#ifndef USE_UNALIGNED_ACCESS
                      NPY_ITER_ALIGNED|
#endif
                      NPY_ITER_NO_BROADCAST;
    }
    */
    //printf( "NumExpr_run() #7\n" ); 
                      
    // Check for empty arrays in expression
    if (n_input > 0) {
        char retsig = get_return_sig(self);

        // Check length for all inputs
        int zeroi, zerolen = 0;
        for (I=0; I < n_input; I++) {
            if (PyArray_SIZE(operands[I]) == 0) {
                zerolen = 0;
                zeroi = I;
                break;
            }
        }

        if (zerolen != 0) {
            // Allocate the output
            int ndim = PyArray_NDIM(operands[zeroi]);
            npy_intp *dims = PyArray_DIMS(operands[zeroi]);
            operands[0] = (PyArrayObject *)PyArray_SimpleNew(ndim, dims,
                                              NPYENUM_from_dchar(retsig));
            if (operands[0] == NULL) { goto fail; }

            ret = (PyObject *)operands[0];
            Py_INCREF(ret);
            goto cleanup_and_exit;
        }
    }

    //printf( "NumExpr_run() #8\n" ); 
    // A case with a single constant output
    if (n_input == 0) {
        char retsig = get_return_sig(self);

        // Allocate the output
        if (operands[0] == NULL) {
            npy_intp dim = 1;
            operands[0] = (PyArrayObject *)PyArray_SimpleNew(0, &dim,
                                        NPYENUM_from_dchar(retsig));
            if (operands[0] == NULL) { goto fail; }
        }
        else {
            PyArrayObject *arrayObj;
            if (PyArray_SIZE(operands[0]) != 1) {
                PyErr_SetString(PyExc_ValueError,
                        "output for a constant expression must have size 1");
                goto fail;
            }
            else if (!PyArray_ISWRITEABLE(operands[0])) {
                PyErr_SetString(PyExc_ValueError,
                        "output is not writeable");
                goto fail;
            }
            Py_INCREF(dtypes[0]);
            arrayObj = (PyArrayObject *)PyArray_FromArray(operands[0], dtypes[0],
                                        NPY_ARRAY_ALIGNED|NPY_ARRAY_UPDATEIFCOPY);
            if (arrayObj == NULL) { goto fail; }
            Py_DECREF(operands[0]);
            operands[0] = arrayObj;
        }

        r = run_interpreter_const(self, PyArray_BYTES(operands[0]), &pc_error);

        ret = (PyObject *)operands[0];
        Py_INCREF(ret);
        goto cleanup_and_exit;
    }

    //printf( "NumExpr_run() #9\n" ); 
    // Allocate the iterator or nested iterators
    if (reduction_size == 1) {
        //printf( "NumExpr_run() #9A, arrayCounter = %d\n", arrayCounter ); 
        // When there's no reduction, reduction_size is 1 as well
        
        // RAM: Ah... problem. operands needs to be n_ndarray only...
        iter = NpyIter_AdvancedNew(arrayCounter, operands,
                            NPY_ITER_BUFFERED|
                            NPY_ITER_REDUCE_OK|
                            NPY_ITER_RANGED|
                            NPY_ITER_DELAY_BUFALLOC|
                            NPY_ITER_EXTERNAL_LOOP,
                            order, casting,
                            op_flags, dtypes,
                            -1, NULL, NULL,
                            BLOCK_SIZE1);
        //printf( "NumExpr_run() #9B\n" );                           
        if (iter == NULL) { goto fail; }

    } else {
        npy_uint32 op_flags_outer[NPY_MAXDIMS];
        // The outer loop is unbuffered 
        op_flags_outer[0] = NPY_ITER_READWRITE|
                            NPY_ITER_ALLOCATE|
                            NPY_ITER_NO_BROADCAST;
        for (I = 0; I < n_input; ++I) {
            op_flags_outer[I] = NPY_ITER_READONLY;
        }
        // Arbitrary threshold for which is the inner loop...benchmark?
        if (reduction_size < INNER_LOOP_MAX_SIZE) {
            reduction_outer_loop = true;
            iter = NpyIter_AdvancedNew(arrayCounter, operands,
                                NPY_ITER_BUFFERED|
                                NPY_ITER_RANGED|
                                NPY_ITER_DELAY_BUFALLOC|
                                NPY_ITER_EXTERNAL_LOOP,
                                order, casting,
                                op_flags, dtypes,
                                oa_ndim, op_axes, NULL,
                                BLOCK_SIZE1);
            if (iter == NULL) { goto fail; }

            // If the output was allocated, get it for the second iterator 
            if (operands[0] == NULL) {
                operands[0] = NpyIter_GetOperandArray(iter)[0];
                Py_INCREF(operands[0]);
            }

            op_axes[0] = &op_axes_reduction_values[0];
            for (I = 0; I < n_input; ++I) {
                op_axes[I] = &op_axes_reduction_values[I];
            }
            op_flags_outer[0] &= ~NPY_ITER_NO_BROADCAST;
            reduce_iter = NpyIter_AdvancedNew(arrayCounter, operands,
                                NPY_ITER_REDUCE_OK,
                                order, casting,
                                op_flags_outer, NULL,
                                1, op_axes, NULL,
                                0);
            if (reduce_iter == NULL) {
                goto fail;
            }
        }
        else {
            PyArray_Descr *dtypes_outer[NPY_MAXDIMS];

            // If the output is being allocated, need to specify its dtype
            dtypes_outer[0] = dtypes[0];
            for (I = 0; I < n_input; ++I) {
                dtypes_outer[I] = NULL;
            }
            iter = NpyIter_AdvancedNew(arrayCounter, operands,
                                NPY_ITER_RANGED,
                                order, casting,
                                op_flags_outer, dtypes_outer,
                                oa_ndim, op_axes, NULL,
                                0);
            if (iter == NULL) {
                goto fail;
            }

            // If the output was allocated, get it for the second iterator
            if (operands[0] == NULL) {
                operands[0] = NpyIter_GetOperandArray(iter)[0];
                Py_INCREF(operands[0]);
            }

            op_axes[0] = &op_axes_reduction_values[0];
            for (I = 0; I < n_input; ++I) {
                op_axes[I] = &op_axes_reduction_values[I];
            }
            op_flags[0] &= ~NPY_ITER_NO_BROADCAST;
            reduce_iter = NpyIter_AdvancedNew(arrayCounter, operands,
                                NPY_ITER_BUFFERED|
                                NPY_ITER_REDUCE_OK|
                                NPY_ITER_DELAY_BUFALLOC|
                                NPY_ITER_EXTERNAL_LOOP,
                                order, casting,
                                op_flags, dtypes,
                                1, op_axes, NULL,
                                BLOCK_SIZE1);
            if (reduce_iter == NULL) {
                goto fail;
            }
        }
    }
    //printf( "NumExpr_run() #10\n" ); 
    // Initialize the output to the reduction unit
    //printf( "TODO: Output size should be appropriate for reductions.\n" );
//    if (is_reduction) {
//        PyArrayObject *iterArray = NpyIter_GetOperandArray(iter)[0];
//        if (LAST_OP(self) >= OP_SUM &&
//            LAST_OP(self) < OP_PROD) {
//                PyObject *zero = PyLong_FromLong(0);
//                PyArray_FillWithScalar(iterArray, zero);
//                Py_DECREF(zero);
//        } else {
//                PyObject *one = PyLong_FromLong(1);
//                PyArray_FillWithScalar(iterArray, one);
//                Py_DECREF(one);
//        }
//    }
    
    //printf( "NumExpr_run() #11\n" ); 
    // Get the sizes of all the operands
    // RAM: this shouldn't be here really, if you change the itemsize 
    // the program will fail.....
    // We could verify that self->registers[I].itemsize == dtypes_tmp[I]->elsize;
//    dtypes_tmp = NpyIter_GetDescrArray(iter);
//    for (I = 0; I < self->n_reg; I++) {
//        if( self->registers[I].kind = KIND_ARRAY ) {     
//            self->registers[I].itemsize = dtypes_tmp[I]->elsize;
//        }
//    }

    // For small calculations, just use 1 thread
    // RAM: this should scale with the number of threads perhaps?  Only use 
    // 1 thread per BLOCK_SIZE1 in the iterator?
    // Also this is still on an element rather than bytesize basis.
    if (NpyIter_GetIterSize(iter) < 2*BLOCK_SIZE1) {
        //printf( "NumExpr_run() FORCING SERIAL MODE\n" ); 

        gs.force_serial = 1;
    }

    // Reductions do not support parallel execution yet
    if (is_reduction) {
        gs.force_serial = 1;
    }


//    for( int I = 0; I < self->program_len; I++ ) {
//        printf( "NE_run::self[%d]: %d %d %d %d %d \n", I,
//            (int)self->program[I].op, (int)self->program[I].ret, (int)self->program[I].arg1, 
//            (int)self->program[I].arg2, (int)self->program[I].arg3 );
//    }
    
    r = run_interpreter(self, iter, reduce_iter,
                             reduction_outer_loop, need_output_buffering,
                             &pc_error);
                        
    //printf( "NumExpr_run() #12\n" ); 
    if (r < 0) {
        if (r == -1) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError,
                    "NumExpr_run(): an error occurred while running the program.");
            }
        } else if (r == -2 || r== -3) {
            PyErr_Format(PyExc_RuntimeError,
                "NumExpr_run(): bad argument at pc = %d.", pc_error);
        } else {
            PyErr_SetString(PyExc_RuntimeError,
                "NumExpr_run(): unknown error occurred while running the program.");
        }
        goto fail;
    }

    //printf( "NumExpr_run() #13\n" ); 
    // Get the output from the iterator
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[0];
    Py_INCREF(ret);

    NpyIter_Deallocate(iter);
    if (reduce_iter != NULL) {
        NpyIter_Deallocate(reduce_iter);
    }
cleanup_and_exit:
    for (I = 0; I < n_input; I++) {
        Py_XDECREF(operands[I]);
        Py_XDECREF(dtypes[I]);
    }

    return ret;
fail:
    for (I = 0; I < n_input; I++) {
        Py_XDECREF(operands[I]);
        Py_XDECREF(dtypes[I]);
    }
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }
    if (reduce_iter != NULL) {
        NpyIter_Deallocate(reduce_iter);
    }
    
    return NULL;
}

/*
Local Variables:
   c-basic-offset: 4
End:
*/