#ifndef NUMEXPR_MODULE_HPP
#define NUMEXPR_MODULE_HPP

// Deal with the clunky numpy import mechanism
// by inverting the logic of the NO_IMPORT_ARRAY symbol.
#define PY_ARRAY_UNIQUE_SYMBOL numexpr_ARRAY_API
#ifndef DO_NUMPY_IMPORT_ARRAY
#  define NO_IMPORT_ARRAY
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// For suppressing depricated char cast warnings
#define CHARP(s) ((char *)(s))

#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>

#include "numexpr_config.hpp"
#include "numexpr_object.hpp"

// thread_data (aka th_params) was merged into global_state, as they both did 
// the same thing.
// ARRAYS that need realloc on thread number change:
// threads, tids, params, stridesArray, iter, reduce_iter
struct global_state {
    // Global variables for threads 
    int n_thread;                    // number of desired threads in pool
    int init_threads_done;           // pool of threads initialized?
    int end_threads;                 // should exisiting threads end?
    // RAM: these fixed length arrays will have to become pointers if we 
    // want to get rid of MAX_THREADS
    pthread_t* threads;              // ARRAY, opaque structure for threads
    int* tids;                       // ARRAY, ID per each thread
    npy_intp gindex;                 // global index for all threads
    int init_sentinels_done;         // Flag, sentinels initialized?
    int giveup;                      // Flag, should parallel code giveup?
    int force_serial;                // Flag, force serial code instead of parallel?
    int pid;                         // the PID for this process

    // Program control and registry structs
    NumExprObject* params;           // ARRAY, copies the programs and registers for each thread.

    // Temporaries in a pre-allocated block
    // A private pool/stack for temporaries so the interpreter does not have to 
    // allocate and deallocate memory with each execution.
    char* tempStack;                 // The pointer to the temporary memory region
    Py_ssize_t tempSize;             // The size of the temporary memory region

    // Syncronization variables 
    pthread_mutex_t global_mutex;    // Previously was a Python threading.Lock()
    pthread_mutex_t count_mutex;
    int count_threads;
    pthread_mutex_t count_threads_mutex;
    pthread_cond_t count_threads_cv;

    // NumPy iterator handles
    npy_intp start;
    npy_intp vlen;
    npy_intp task_size;

    npy_intp* stridesArray;              // ARRAY, one strides array per thread
    NpyIter** iter;                      // ARRAY, one iterator per thread
    NpyIter** reduce_iter;               // ARRAY, when doing nested iteration for a reduction
    bool reduction_outer_loop;           // Flag indicating reduction is the outer loop instead of the inner
    bool need_output_buffering;          // Flag indicating whether output buffering is needed

    // Global return and error handling
    int ret_code;
    int *pc_error;
    char **errorMessage;

    global_state() {
        pthread_mutex_init(&global_mutex, NULL);
        n_thread = DEFAULT_THREADS;
        init_threads_done = 0;
        end_threads = 0;
        pid = 0;
        threads = (pthread_t *)calloc( DEFAULT_THREADS, sizeof(pthread_t) );
        tids = (int *)calloc( DEFAULT_THREADS, sizeof(int) );
        params = (NumExprObject *)calloc( DEFAULT_THREADS, sizeof(NumExprObject) );
        for( int I = 0; I < DEFAULT_THREADS; I++ ) {
            params[I].registers = (NumExprReg *)calloc( NPY_MAXARGS,  sizeof(NumExprReg) );
        }
        stridesArray = (npy_intp *)calloc( DEFAULT_THREADS, sizeof(npy_intp) );
        iter = (NpyIter **)calloc( DEFAULT_THREADS, sizeof(NpyIter*) );
        reduce_iter = (NpyIter **)calloc( DEFAULT_THREADS, sizeof(NpyIter*) );

        tempSize = 0;
        tempStack = NULL;
        task_size = DEFAULT_BLOCK;
    }
};

int numexpr_set_nthreads(int nthreads_new);
Py_ssize_t numexpr_set_tempsize(Py_ssize_t newSize);

#endif // NUMEXPR_MODULE_HPP
