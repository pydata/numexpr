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
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif

using namespace std;

extern global_state gs;


int
NPYENUM_from_dchar(char c)
{
    switch (c) {
        case '?': return NPY_BOOL;
        case 'b': return NPY_INT8;
        case 'B': return NPY_UINT8;
        case 'h': return NPY_INT16;
        case 'H': return NPY_UINT16;
#ifdef _WIN32
        case 'l': return NPY_INT32;
        case 'L': return NPY_UINT32;
        case 'q': return NPY_INT64;
        case 'Q': return NPY_UINT64;
#else
        case 'i': return NPY_INT32;
        case 'I': return NPY_UINT32;
        case 'l': return NPY_INT64;
        case 'L': return NPY_UINT64;
#endif
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
get_return_sig(NumExprObject* self)
{
    NE_REGISTER last_reg = self->program[self->program_len - 1].ret;
    return self->registers[last_reg].dchar;
}


// TODO: refactor get_reduction_axis
static int
get_reduction_axis(NumExprObject* self) {
    
//    Py_ssize_t end = PyBytes_Size(program);
//    int axis = ((unsigned char *)PyBytes_AS_STRING(program))[end-1];
//    if (axis != 255 && axis >= NPY_MAXDIMS)
//        axis = NPY_MAXDIMS - axis;
//    return axis;
    return -1;
}

// This function prepares the pointers towards the temporary memory block, and 
// the NumExprObjects used by threads.
// Replacement for get_temps_space() and NumExprObject_copy_threadsafe().
int 
prepareThreads( NumExprObject* self, NpyIter *iter, int *pc_error, char **errorMessage ) {
    // Variables
    int I, R; 
    npy_intp memOffset = 0, taskFactor, numBlocks;
    NumExprReg* saveRegister;

    // Setup error tracking
    gs.ret_code = 0;
    gs.pc_error = pc_error;
    gs.errorMessage = errorMessage;

    // printf( "prepareThreads #1\n");
    // Stuff from vm_engine_iter_parallel:
    NpyIter_GetIterIndexRange(iter, &gs.start, &gs.vlen);
    
    // Try to make it so each thread gets 16 tasks.  This is a compromise
    // between 1 task per thread and one block per task.
    // RAM: would be nice to benchmark this sort of assumption.
    taskFactor = TASKS_PER_THREAD*BLOCK_SIZE1*gs.n_thread;
    numBlocks = (gs.vlen - gs.start + taskFactor - 1) /
                            taskFactor;
    gs.task_size = numBlocks * BLOCK_SIZE1; // Note this is much bigger than BLOCK_SIZE1

    // printf("Original: gs.tempSize = %d, gs.tempStack = %p\n", gs.tempSize, gs.tempStack );
    // Ensure the temporary memory storage is big enough
    // printf( "prepareThreads #1B\n");
    if( BLOCK_SIZE1 * self->total_temp_itemsize >  gs.tempSize ) {
        // printf( "temp size too small, resizing #1\n");
        numexpr_set_tempsize( BLOCK_SIZE1 * self->total_temp_itemsize );
        // printf("Engorged: gs.tempSize = %d, gs.tempStack = %p\n", gs.tempSize, gs.tempStack );
    }

    // `Do` part of the `for` loop before, use `self` for the first thread.
    gs.iter[0] = iter;
    gs.params[0] = *self;
    // printf( "prepareThreads #2\n");
    // Setup temporaries memory pointers for the first NumExprObject
    for( R=0; R < self->n_reg; R++ ) {
        // printf( "reg#%d has kind %d and itemsize %d\n", R, self->registers[R].kind, self->registers[R].itemsize );
        if( self->registers[R].kind != KIND_TEMP ) continue;

        self->registers[R].mem = gs.tempStack + memOffset;
        // printf( "    param #%d, reg #%d, points to %p\n", I, R, gs.params[I].registers[R].mem );
        memOffset += BLOCK_SIZE1 * self->registers[R].itemsize;
    }
    // printf( "prepareThreads #3\n");
    // Make copies of iterators and NumExprObjects for each additional thread
    for( I = 1; I < gs.n_thread; ++I ) {
        // TODO: add reduce_iter when you parallelize reductions
        gs.iter[I] = NpyIter_Copy(iter);
        if (gs.iter[I] == NULL) {
            // Error: deallocate all iterators and return an error code.
            --I;
            for (; I > 0; --I) {
                NpyIter_Deallocate(gs.iter[I]);
            }
            return -1;
        }
        // Save reference to gs.params[I].registers as memcpy overwrites it.
        saveRegister = gs.params[I].registers;

        // Copy the NumExprObjects in gs.params
        memcpy( &gs.params[I], self, sizeof(NumExprObject) );
        gs.params[I].registers = saveRegister;

        // Make copies of self->registers
        memcpy( gs.params[I].registers, self->registers, self->n_reg * sizeof(NumExprReg) );

        // Setup temporaries memory pointers
        for( R=0; R < self->n_reg; R++ ) {

            if( self->registers[R].kind != KIND_TEMP) continue;

            gs.params[I].registers[R].mem = gs.tempStack + memOffset;
            // printf( "    param #%d, reg #%d, points to %p\n", I, R, gs.params[I].registers[R].mem );
            memOffset += BLOCK_SIZE1 * self->registers[R].itemsize;
        }
    }


    return 0;
}

int 
finishThreads() {
    // Deallocate the iterators
    for (int I = 1; I < gs.n_thread; ++I) {
        NpyIter_Deallocate(gs.iter[I]);     
    }
    return 0;
}

// Serial/parallel task iterator version of the VM engine
int vm_engine_iter_task(NpyIter *iter, 
                    const NumExprObject *params,
                    int *pc_error, char **errorMessage)
{
    NpyIter_IterNextFunc *iterNext;
    npy_intp task_size, *sizePtr;
    char **iterDataPtr;
    npy_intp *iterStrides;

    iterNext = NpyIter_GetIterNext(iter, errorMessage);
    if (iterNext == NULL) {
        return -1;
    }
    
    sizePtr = NpyIter_GetInnerLoopSizePtr(iter);
    iterDataPtr = NpyIter_GetDataPtrArray(iter);
    iterStrides = NpyIter_GetInnerStrideArray(iter);

    // DEBUG
    // printf( "DEBUG vm_enginer_iter_task\n" );
    // for( int I = 0; I < params->program_len; I++ ) {                                         
    //     printf( "program[%d]:: r:%d a1:%d a1:%d a2:%d a3:%d \n", I,
    //         (int)params->program[I].op, (int)params->program[I].ret, (int)params->program[I].arg1, 
    //         (int)params->program[I].arg2, (int)params->program[I].arg3 );
    // }
    // for( int I = 0; I < params->n_reg; I++ ) {
    //     printf( "regs[%d]:: kind:%d, mem:%p, \n", I, params->registers[I].kind, params->registers[I].mem  );
    // }
    // printf( "params: %p, n_reg: %d, iter: %p\n", params, params->n_reg, iter );
    // for( int I = 0; I < params->n_reg; I++ ) {
    //     int ac = 0;
    //     if( params->registers[I].kind == KIND_ARRAY || params->registers[I].kind == KIND_RETURN ) {
    //         printf( "iterDataPtr[%d]:: %p, iterStrides[%d]:: %ld, sizePtr:: %ld \n", 
    //             I, iterDataPtr[ac], I, (void *)iterStrides[ac], *sizePtr );
    //         ac++;
    //     }

    // }
    // printf("END DEBUG\n");

    // First do all the blocks with a compile-time fixed size. This makes a 
    // big difference (30-50% on some tests).
    // TODO: this can be replaced in the generator with a fixed size in 
    // _bytes_ instead of _elements_

    task_size = *sizePtr;
    // Success, with auto-vectorization it doesn't need to be a fixed size, 
    // compared to unrolling loops. Looks like we can cut-down the number of 
    // includes which will shrink the machine code.
    while( task_size > 0 ) {
#define REDUCTION_INNER_LOOP            
#include "interp_body_GENERATED.cpp"
#undef REDUCTION_INNER_LOOP
        iterNext(iter);
        task_size = *sizePtr;   
    }


    return 0;
}

static int
vm_engine_iter_outer_reduce_task(NpyIter *iter, 
                const NumExprObject *params, int *pc_error, char **errorMessage)
{
    NpyIter_IterNextFunc *iterNext;
    npy_intp task_size, *sizePtr;
    char **iterDataPtr;
    npy_intp *iterStrides;

    iterNext = NpyIter_GetIterNext(iter, errorMessage);
    if (iterNext == NULL) {
        return -1;
    }

    sizePtr = NpyIter_GetInnerLoopSizePtr(iter);
    iterDataPtr = NpyIter_GetDataPtrArray(iter);
    iterStrides = NpyIter_GetInnerStrideArray(iter);

    task_size = *sizePtr;
    // First do all the blocks with a compile-time fixed size.
    // This makes a big difference (30-50% on some tests).
    // RAM: Not-so-much with vectorized loops
    
    while( task_size > 0 ) {
#define NO_OUTPUT_BUFFERING
#include "interp_body_GENERATED.cpp"
#undef NO_OUTPUT_BUFFERING
        iterNext(iter);
        task_size = *sizePtr;   
    }
    
    return 0;
}

// Parallel iterator version of VM engine 
// This function fills out the global state and then unlocks the mutexes 
// used to control the threads in module.cpp::th_worker
// prepareThreads() must be called beforehand (or else `self` could not be const).
static int
vm_engine_iter_parallel(NpyIter *iter, const NumExprObject *self,
                        bool need_output_buffering, int *pc_error,
                        char **errorMessage)
{
    if (errorMessage == NULL) return -1;

    // Threads are prepared for execution in prepareThreads()
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
        gs.count_threads--; // RAM: why is this needed?
        pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
    }
    else {
        pthread_cond_broadcast(&gs.count_threads_cv);
    }
    pthread_mutex_unlock(&gs.count_threads_mutex);
    Py_END_ALLOW_THREADS;

    return gs.ret_code;
}

static int
run_interpreter(NumExprObject *self, NpyIter *iter, NpyIter *reduce_iter,
                     bool reduction_outer_loop, bool need_output_buffering,
                     int *pc_error)
{
        
    int returnValue;

    char *errorMessage = NULL;

    *pc_error = -1;
    
    // printf( "run_interpreter #1\n" );
    if ((gs.n_thread == 1) || gs.force_serial) {
        // Can do it as one "task"
        if (reduce_iter == NULL) {
            // Reset the iterator to allocate its buffers
            if(NpyIter_Reset(iter, NULL) != NPY_SUCCEED) return -1;

            // printf( "run_interpreter #2B\n" );
            returnValue = prepareThreads( self, iter, pc_error, &errorMessage );
            if( returnValue != 0 ) return -1;

            // printf( "run_interpreter #3\n" );
            Py_BEGIN_ALLOW_THREADS
            returnValue = vm_engine_iter_task(iter, self, pc_error, &errorMessage);                            
            Py_END_ALLOW_THREADS
            // printf( "run_interpreter #5\n" );
            finishThreads();
        }
        else { // Reduction operation 
            if (reduction_outer_loop) { // Reduction on outer loop 
                char **dataPtr;
                NpyIter_IterNextFunc *iterNext;

                dataPtr = NpyIter_GetDataPtrArray(reduce_iter);
                iterNext = NpyIter_GetIterNext(reduce_iter, NULL);
                if (iterNext == NULL) {
                    return -1;
                }
                                    
                // get_temps_space( safeParams, BLOCK_SIZE1);
                returnValue = prepareThreads( self, iter, pc_error, &errorMessage );
                if( returnValue != 0 ) return -1;

                Py_BEGIN_ALLOW_THREADS
                do {
                    returnValue = NpyIter_ResetBasePointers(iter, dataPtr, &errorMessage);
                    if (returnValue >= 0) {
                        returnValue = vm_engine_iter_outer_reduce_task(iter,
                                                self,
                                                pc_error, &errorMessage);
                    }
                    if (returnValue < 0) {
                        break;
                    }
                } while (iterNext(reduce_iter));
                Py_END_ALLOW_THREADS
                finishThreads();
            }
            else { // Inner reduction    
                char **dataPtr;
                NpyIter_IterNextFunc *iterNext;

                dataPtr = NpyIter_GetDataPtrArray(iter);
                iterNext = NpyIter_GetIterNext(iter, NULL);
                if (iterNext == NULL) {
                    return -1;
                }
                returnValue = prepareThreads( self, iter, pc_error, &errorMessage );
                if( returnValue != 0 ) return -1;
                Py_BEGIN_ALLOW_THREADS
                do {
                    returnValue = NpyIter_ResetBasePointers(reduce_iter, dataPtr,
                                                                    &errorMessage);
                    if (returnValue >= 0) {
                        returnValue = vm_engine_iter_task(reduce_iter, 
                                                self, pc_error, &errorMessage);
                    }
                    if (returnValue < 0) {
                        break;
                    }
                } while (iterNext(iter));
                Py_END_ALLOW_THREADS
                
                finishThreads();
            }
        }
    }
    else {
        // printf( "run_interpreter parallel branch\n" );    
        if (reduce_iter == NULL) {
            returnValue = prepareThreads( self, iter, pc_error, &errorMessage );
            if( returnValue != 0 ) return -1;
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
run_interpreter_const(NumExprObject *params, char *output, int *pc_error)
{
    int returnValue = 0;


    if (params->n_array != 0) {
        return -1;
    }

#define SINGLE_ITEM_CONST_LOOP
#define task_size 1
#define NO_OUTPUT_BUFFERING // Because it's constant
#include "interp_body_GENERATED.cpp"
#undef NO_OUTPUT_BUFFERING
#undef task_size
#undef SINGLE_ITEM_CONST_LOOP
    
    return 0;
}

PyObject *
NumExpr_run(NumExprObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *operands[NPY_MAXARGS];
    PyArray_Descr *dtypes[NPY_MAXARGS];
    PyObject *tmp, *returnArray;
    PyObject *objectRef, *arrayRef;
    npy_uint32 op_flags[NPY_MAXARGS];
    NPY_CASTING casting = NPY_SAFE_CASTING;
    NPY_ORDER order = NPY_KEEPORDER;
    NE_WORD I, J;

    NE_REGISTER arrayCounter = 0, n_input;
    int allocOutput = 0;
    npy_intp maxDims = 0;              // The largest dimensionality in any of the passed arguments
    npy_intp arrayOffset = 0;
    npy_intp* broadcastShape = NULL;   // An array tracking the broadcasted dimensions of the output
    npy_intp* arrayShape = NULL;
    NE_REGISTER retIndex, arg1Index, arg2Index;
    PyArray_Descr* outputType;         // The NumPy array dtype for the output array

    int r, pc_error = 0;
    int reduction_axis = -1;
    npy_intp reduction_size = 1;
    int is_reduction = 0;
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
    if (!gs.init_threads_done || gs.pid != getpid())
        numexpr_set_nthreads(gs.n_thread);
            
    // Don't force serial mode by default
    gs.force_serial = 0;

    n_input = (int)PyTuple_Size(args);

    memset(operands, 0, sizeof(operands));
    memset(dtypes, 0, sizeof(dtypes));
    // Moving away from kwds
    // if (kwds) {
    //     // Parse standard keyword arguments
    //     // User can't change casting here, it would change the program
    //     tmp = PyDict_GetItemString(kwds, "casting"); // borrowed ref
    //     if (tmp != NULL && !PyArray_CastingConverter(tmp, &casting)) {
    //         return PyErr_Format(PyExc_ValueError,
    //             "NumExpr_run(): casting keyword argument is invalid.");
    //     }
    //     Array ordering is not implemented Python-side
    //     tmp = PyDict_GetItemString(kwds, "order"); // borrowed ref
    //     if (tmp != NULL && !PyArray_OrderConverter(tmp, &order)) {
    //         return PyErr_Format(PyExc_ValueError,
    //             "NumExpr_run(): order keyword argument is invalid.");
    //     }
    //     tmp = PyDict_GetItemString(kwds, "#alloc");
    //     if( tmp == NULL ) {
    //         return PyErr_Format(PyExc_ValueError,
    //             "NumExpr_run(): #alloc keyword argument is required.");
    //     }
    //     allocOutput = PyObject_IsTrue( tmp );
    // }

    if ( !allocOutput && self->n_array != n_input) {
        return PyErr_Format(PyExc_ValueError,
            "NumExpr_run(): number of inputs %d doesn't match program ndarrays %d", n_input, self->n_array );
    } else if ( allocOutput && self->n_array != n_input + 1 ) {
        return PyErr_Format(PyExc_ValueError,
            "NumExpr_run(): number of inputs %d doesn't match program ndarrays plus output %d", n_input, self->n_array );
    } else if (n_input >= NPY_MAXARGS) {
        // This is related to the NumPy limit of 32 arguments in an nditer
        return PyErr_Format(PyExc_ValueError,
            "NumExpr_run(): Number of inputs exceeds NPY_MAXARGS that NumPy was compiled with.");
    }
                      
    // Run through all the registers and match the input arguments to that
    // parsed by NumExpr_init
    for (I = 0; I < self->n_reg; I++) {
        // printf( "reg#%d:: arrayCounter=%d, n_input=%d, kind=%d\n", I, arrayCounter, n_input, self->registers[I].kind );

        if( arrayCounter > n_input + allocOutput ) { 
            PyErr_SetString(PyExc_ValueError, 
                "Too many input arrays for program: arrayCounter > n_input" );
            goto fail; 
        }
        if( self->registers[I].kind == KIND_SCALAR || self->registers[I].kind == KIND_TEMP )
            continue;
        
        // if a array is missing
        objectRef = PyTuple_GET_ITEM(args, arrayCounter);
        if( objectRef == Py_None ) { // Unallocated output of KIND_RETURN
            printf( "NumExpr_run got Py_None, assuming unallocated output\n" );
            allocOutput = 1;
            arrayCounter++;
            continue;
        }

        // KIND_RETURN arrays are treated identically to KIND_ARRAY if they are 
        // pre-allocated.
        int typecode = NPYENUM_from_dchar( self->registers[I].dchar );
        // Convert it if it's not an array
        if (!PyArray_Check(objectRef)) {
            printf( "DEBUG: arg[%d] is not an array.\n", I );
            if (typecode == -1) { 
                PyErr_SetString(PyExc_ValueError, 
                    "passed in array with typecode == -1" );
                goto fail; 
            }
            arrayRef = PyArray_FROM_OTF(objectRef, typecode, NPY_ARRAY_NOTSWAPPED);
        }
        else {
            Py_INCREF(objectRef);
            arrayRef = objectRef;
        }
        maxDims = MAX(maxDims, PyArray_NDIM( (PyArrayObject *)arrayRef) );

        operands[arrayCounter] = (PyArrayObject *)arrayRef;
        dtypes[arrayCounter] = PyArray_DescrFromType(typecode);
        
              
        if (operands[arrayCounter] == NULL || dtypes[arrayCounter] == NULL) { 
            PyErr_SetString(PyExc_ValueError, 
                "operands[arrayCounter] == NULL or dtypes[arrayCounter] == NULL" );
            goto fail; 
        }
        
        op_flags[arrayCounter] = NPY_ITER_READONLY|
// #ifdef USE_VML
//                         (ex_uses_vml ? (NPY_ITER_CONTIG|NPY_ITER_ALIGNED) : 0)|
// #endif
#ifndef USE_UNALIGNED_ACCESS
                        NPY_ITER_ALIGNED|
#endif
                        NPY_ITER_NBO;
        
        arrayCounter++;
    }
            
    // Output array allocation (from NumPy ufuncs documentation)
    //
    // Each universal function takes array inputs and produces array outputs by performing the core function 
    // element-wise on the inputs. Standard broadcasting rules are applied so that inputs not sharing exactly the 
    // same shapes can still be usefully operated on. Broadcasting can be understood by four rules:
    // 
    //     All input arrays with ndim smaller than the input array of largest ndim, have 1’s prepended to their shapes.
    //     The size in each dimension of the output shape is the maximum of all the input sizes in that dimension.
    //     An input can be used in the calculation if its size in a particular dimension either matches the output 
    //     size in that dimension, or has value exactly 1.
    //     If an input has a dimension size of 1 in its shape, the first data entry in that dimension will be used 
    //     for all calculations along that dimension. In other words, the stepping machinery of the ufunc will simply 
    //     not step along that dimension (the stride will be 0 for that dimension).
    // 
    // Broadcasting is used throughout NumPy to decide how to handle disparately shaped arrays; for example, all 
    // arithmetic operations (+, -, *, ...) between ndarrays broadcast the arrays before operation.
    // 
    // A set of arrays is called “broadcastable” to the same shape if the above rules produce a valid result, i.e., 
    // one of the following is true:
    // 
    //     The arrays all have exactly the same shape.
    //     The arrays all have the same number of dimensions and the length of each dimensions is either a common length or 1.
    //     The arrays that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.
    if( allocOutput ) {
        arrayCounter = 0;
        
        printf( "MAXDIMS = %d\n", maxDims );
        // There are a couple of options for broadcast tracking:
        // 1.) A 2D array that we operate over with pointer arithmetic.
        // 2.) Add a .virtualShape pointer to the NumExprReg struct.  This would require
        // seperate malloc calls.
        // We'll go with option 1.) for now.
        broadcastShape = (npy_intp*)malloc( self->n_reg * maxDims * sizeof(npy_intp) );
        // Fill in broadcastShape
        for( I = 0; I < self->n_reg; I++ ) {
            arrayOffset = I*maxDims;

            // Fill default value of all 1s
            for( J = 0; J < maxDims; J++ ) {
                broadcastShape[arrayOffset + J] = 1;
            }

            if( self->registers[I].kind == KIND_ARRAY ) {
                // Broadcast appends leading singleton dimension to under-dimension arrays
                arrayShape = PyArray_SHAPE( operands[arrayCounter] );
                arrayOffset += maxDims - PyArray_NDIM(operands[arrayCounter]);
                for( J = 0; J < PyArray_NDIM(operands[arrayCounter]); J++ ) {
                    broadcastShape[arrayOffset + J] = arrayShape[J];
                }
                arrayCounter++;
            }
            else if( self->registers[I].kind == KIND_RETURN ) {
                arrayCounter++;
            }

        }

        // Iterate through program and determine virtual shape of temporaries
        // and eventually the output.
        for( I=0; I < self->program_len; I++ ) {
            retIndex = self->program[I].ret;
            if( (self->registers[retIndex].kind == KIND_TEMP) 
                || (self->registers[retIndex].kind == KIND_RETURN) ) {
                // Compute the virtual broadcast shape of the temporary.  
                // Note the virtual broadcast shape can change as the program
                // advances because the temporaries are re-used.
                arrayOffset = retIndex * maxDims;
                arg1Index = self->program[I].arg1;
                arg2Index = self->program[I].arg2;
                
                if (arg2Index == NULL_REG) { // Case: Casts and similar only have one argument
                    printf( "Prog#%d:: UnaryOp broadcast at #%d to [", I, retIndex );
                    for( J = 0; J < maxDims; J++ ) {
                        broadcastShape[retIndex*maxDims+J] = broadcastShape[arg1Index*maxDims+J];
                        printf( " %d,", broadcastShape[retIndex*maxDims+J] );
                    }
                    printf( "]\n" );
                } else if( self->program[I].arg3 == NULL_REG ) { // Case: BinaryOps have two arguments
                    printf( "Prog#%d:: BinaryOp broadcast at #%d to [", I, retIndex );
                    for( J = 0; J < maxDims; J++ ) {
                        // output size is MAX(arg1,arg2)
                        // TODO: error checking?  Or let nditer handle that?
                        broadcastShape[retIndex*maxDims+J] = MAX( broadcastShape[arg1Index*maxDims+J], broadcastShape[arg2Index*maxDims+J] );
                        printf( " %d,", broadcastShape[retIndex*maxDims+J] );
                    }
                    printf( "]\n" );
                } else { // With three args is the ternary 'where' only
                    for( J = 0; J < maxDims; J++ ) {
                        // output size is MAX(arg2,arg3)
                        broadcastShape[retIndex*maxDims+J] = MAX( broadcastShape[arg2Index*maxDims+J], broadcastShape[self->program[I].arg3*maxDims+J] );
                    }
                }
            }
            
        } 

        // Allocate the output array
        printf( "Allocating output array to index %d, operand index %d\n", self->returnReg, self->returnOperand);
        // The Return type is parsed by the Python-side
        outputType = PyArray_DescrFromType( NPYENUM_from_dchar( self->registers[self->returnReg].dchar ) );
        
        printf( "Broadcast output to type \'%c\', shape : [", outputType->type );
        for( J = 0; J < maxDims; J++ ) {
            printf( " %d,", broadcastShape[retIndex+J] );
        }
        printf( " ]\n" );
        // This should be the arrayCounter, not the registers index!
        operands[self->returnOperand] = (PyArrayObject*)PyArray_SimpleNewFromDescr( maxDims, broadcastShape + retIndex*maxDims, outputType );
        dtypes[self->returnOperand] = outputType;
        op_flags[self->returnOperand] = NPY_ITER_READONLY|
        // #ifdef USE_VML
        //                         (ex_uses_vml ? (NPY_ITER_CONTIG|NPY_ITER_ALIGNED) : 0)|
        // #endif
        #ifndef USE_UNALIGNED_ACCESS
                                    NPY_ITER_ALIGNED|
        #endif
                                    NPY_ITER_NBO;

        free( broadcastShape );
    }

    // printf( "NumExpr_run() #6, arrayCounter = %d\n", arrayCounter ); 
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
    // printf( "NumExpr_run() #7\n" ); 

    

    /*
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

            returnArray = (PyObject *)operands[0];
            Py_INCREF(returnArray);
            goto cleanup_and_exit;
        }
    }
    */

    // printf( "NumExpr_run() #8\n" ); 
    // A case with a single constant output
    if (n_input == 0) {
        char retsig = get_return_sig(self);

        // Allocate the output
        if (operands[self->returnOperand] == NULL) {
            npy_intp dim = 1;
            operands[self->returnOperand] = (PyArrayObject *)PyArray_SimpleNew(0, &dim,
                                        NPYENUM_from_dchar(retsig));
            if (operands[self->returnOperand] == NULL) { 
                PyErr_SetString(PyExc_ValueError, 
                    "operands[self->returnReg] == NULL" );
                goto fail; 
            }
        }
        else {
            PyArrayObject *arrayObj;
            if (PyArray_SIZE(operands[self->returnOperand]) != 1) {
                PyErr_SetString(PyExc_ValueError,
                        "output for a constant expression must have size 1");
                goto fail;
            }
            else if (!PyArray_ISWRITEABLE(operands[self->returnOperand])) {
                PyErr_SetString(PyExc_ValueError,
                        "output is not writeable");
                goto fail;
            }
            Py_INCREF(dtypes[self->returnOperand]);
            arrayObj = (PyArrayObject *)PyArray_FromArray(operands[0], dtypes[0],
                                        NPY_ARRAY_ALIGNED|NPY_ARRAY_UPDATEIFCOPY);
            if (arrayObj == NULL) { 
                PyErr_SetString(PyExc_ValueError, 
                    "arrayObj == NULL" );
                goto fail; 
            }
            Py_DECREF(operands[self->returnOperand]);
            operands[0] = arrayObj;
        }

        r = run_interpreter_const(self, PyArray_BYTES(operands[self->returnOperand]), &pc_error);

        returnArray = (PyObject *)operands[self->returnOperand];
        Py_INCREF(returnArray);
        goto cleanup_and_exit;
    }

    // printf( "NumExpr_run() #9\n" ); 
    // Allocate the iterator or nested iterators
    if (reduction_size == 1) {
        // When there's no reduction, reduction_size is 1 as well
        // printf( "NumExpr_run() #9A, arrayCounter = %d\n", arrayCounter+allocOutput ); 
        iter = NpyIter_AdvancedNew(arrayCounter, operands,
                            NPY_ITER_BUFFERED|
                            NPY_ITER_REDUCE_OK|
                            NPY_ITER_RANGED|
                            NPY_ITER_DELAY_BUFALLOC|
                            NPY_ITER_EXTERNAL_LOOP|
                            NPY_ITER_ZEROSIZE_OK,
                            order, casting,
                            op_flags, dtypes,
                            -1, NULL, NULL,
                            BLOCK_SIZE1);
        // printf( "NumExpr_run() #9A2\n" );                           
        if (iter == NULL) { goto fail; }

    } else {
        // printf( "NumExpr_run() #9B\n" );
        npy_uint32 op_flags_outer[NPY_MAXDIMS];
        // The outer loop is unbuffered 
        
        for (I = 0; I < n_input; ++I) {
            op_flags_outer[I] = NPY_ITER_READONLY;
        }
        op_flags_outer[self->returnOperand] = NPY_ITER_READWRITE|
                                            NPY_ITER_ALLOCATE|
                                            NPY_ITER_NO_BROADCAST;
        // Arbitrary threshold for which is the inner loop...benchmark?
        if (reduction_size < INNER_LOOP_MAX_SIZE) {
            // printf( "NumExpr_run() #9C\n" );
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
            if (iter == NULL) { 
                PyErr_SetString(PyExc_ValueError, 
                    "for case (reduction_size < INNER_LOOP_MAX_SIZE), iter == NULL" );
                goto fail; 
            }

            // If the output was allocated, get it for the second iterator 
            if (operands[self->returnOperand] == NULL) {
                printf( "NumExpr_run: operands[0] == NULL, should never get here any more\n" );
                operands[self->returnOperand] = NpyIter_GetOperandArray(iter)[self->returnOperand];
                Py_INCREF(operands[self->returnOperand]);
            }

            op_axes[self->returnOperand] = &op_axes_reduction_values[self->returnOperand];
            for (I = 0; I < n_input; ++I) {
                op_axes[I] = &op_axes_reduction_values[I];
            }
            op_flags_outer[self->returnOperand] &= ~NPY_ITER_NO_BROADCAST;
            reduce_iter = NpyIter_AdvancedNew(arrayCounter, operands,
                                NPY_ITER_REDUCE_OK,
                                order, casting,
                                op_flags_outer, NULL,
                                1, op_axes, NULL,
                                0);
            if (reduce_iter == NULL) {
                PyErr_SetString(PyExc_ValueError, 
                    "for case (reduction_size < INNER_LOOP_MAX_SIZE), reduce_iter == NULL" );
                goto fail;
            }
        }
        else {
            // printf( "NumExpr_run() #9D\n" );
            PyArray_Descr *dtypes_outer[NPY_MAXDIMS];

            // If the output is being allocated, need to specify its dtype
            dtypes_outer[self->returnOperand] = dtypes[self->returnOperand];
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
                PyErr_SetString(PyExc_ValueError, 
                    "for case (reduction_size >= INNER_LOOP_MAX_SIZE), iter == NULL" );
                goto fail;
            }

            // If the output was allocated, get it for the second iterator
            if (operands[self->returnOperand] == NULL) {
                operands[self->returnOperand] = NpyIter_GetOperandArray(iter)[self->returnOperand];
                Py_INCREF(operands[self->returnOperand]);
            }

            op_axes[self->returnOperand] = &op_axes_reduction_values[self->returnOperand];
            for (I = 0; I < n_input; ++I) {
                op_axes[I] = &op_axes_reduction_values[I];
            }
            op_flags[self->returnOperand] &= ~NPY_ITER_NO_BROADCAST;
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
                PyErr_SetString(PyExc_ValueError, 
                    "for case (reduction_size >= INNER_LOOP_MAX_SIZE), iter == NULL" );
                goto fail;
            }
        }
    }
    // printf( "NumExpr_run() #10\n" ); 
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
    


    // For small calculations, just use 1 thread
    // RAM: this should scale with the number of threads perhaps?  Only use 
    // 1 thread per BLOCK_SIZE1 in the iterator?
    // Also this is still on an element rather than bytesize basis.
    if (NpyIter_GetIterSize(iter) < 2*BLOCK_SIZE1) {
        // printf( "NumExpr_run() FORCING SERIAL MODE\n" ); 
        gs.force_serial = 1;
    }

    // Reductions do not support parallel execution yet
    if (is_reduction) {
        gs.force_serial = 1;
    }
    // printf( "NumExpr_run() #11\n" ); 
    r = run_interpreter(self, iter, reduce_iter,
                             reduction_outer_loop, need_output_buffering,
                             &pc_error);
                        
    // printf( "NumExpr_run() #12\n" ); 
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

    // printf( "NumExpr_run() #13, returnOperand: %d\n", self->returnOperand ); 
    // Get the output from the iterator
    returnArray = (PyObject *)NpyIter_GetOperandArray(iter)[self->returnOperand];
    Py_INCREF(returnArray);
    // printf( "NumExpr_run() #14\n" ); 
    NpyIter_Deallocate(iter);
    // printf( "NumExpr_run() #14B\n" ); 
    if (reduce_iter != NULL) {
        NpyIter_Deallocate(reduce_iter);
    }

cleanup_and_exit:
    // With Python 3.6 and the new malloc arena, the behavoir is quite different
    // and 3.7 may be different still.  So for the moment don't de-reference 
    // these two arrays.
    // printf( "NumExpr_run() De-counting of references disabled!\n" ); 
    // for (I = 0; I < n_input; I++) {
    //     Py_XDECREF(operands[I]);
    //     Py_XDECREF(dtypes[I]);
    // }
    // printf( "NumExpr_run() #16: returnArray = %p\n", returnArray ); 
    return returnArray;

fail:
    printf( "Failed.\n" );
    // for (I = 0; I < n_input; I++) {
    //     Py_XDECREF(operands[I]);
    //     Py_XDECREF(dtypes[I]);
    // }
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