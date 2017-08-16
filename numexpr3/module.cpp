// Numexpr - Fast numerical array expression evaluator for NumPy.
//
//      License: BSD
//      Author:  See AUTHORS.txt
//
//  See LICENSE.txt for details about copyright and rights to use.
//
// module.cpp contains the CPython-specific module exposure.

#define DO_NUMPY_IMPORT_ARRAY

#include "module.hpp"
#include <structmember.h>
#include <vector>

#include "interp_header_GENERATED.hpp"
#include "numexpr_object.hpp"

using namespace std;

// Global state definition. Also includes what used to be in th_params.
global_state gs;

// Do the worker job for a certain thread
void* th_worker(void *tidptr) {
    int tid = *(int *)tidptr;
    // Parameters for threads
    npy_intp start;
    npy_intp vlen;
    npy_intp task_size;
    NpyIter* iter;
    NumExprObject* neObj;

    int* pc_error;
    int ret;
    npy_intp istart, iend;
    char** errorMessage;
    // For output buffering if needed
    vector<char> out_buffer;

    while(1) {
        
        if( tid == 0 ) {
            gs.init_sentinels_done = 0; // sentinels have to be initialised yet
        }

        // Meeting point for all threads (wait for initialization)
        // printf( "TH_WORKER %d:: Acquire init lock. count = %d\n", tid, gs.count_threads );
        pthread_mutex_lock(&gs.count_threads_mutex);
        if( gs.count_threads < gs.n_thread ) {
            gs.count_threads++;
            pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
        }
        else { // Every thread is ready, go...
            // printf( "TH_WORKER %d:: started VM.\n", tid );
            pthread_cond_broadcast(&gs.count_threads_cv);
        }
        // printf( "TH_WORKER %d:: Release init lock. count = %d\n", tid, gs.count_threads );
        pthread_mutex_unlock(&gs.count_threads_mutex);

        // Check if thread has been asked to return
        if( gs.end_threads ) {
            // printf( "TH_WORKER %d:: Killing,\n", tid );
            return(0);
        }

        // Get parameters for this thread before entering the main loop 
        start = gs.start;
        vlen = gs.vlen;
        task_size = gs.task_size;
        neObj = &gs.params[tid];
        pc_error = gs.pc_error;

        // printf( "TH_WORKER %d:: got neObj at %p\n", tid, neObj);

        errorMessage = gs.errorMessage;

        // Loop over blocks
        pthread_mutex_lock(&gs.count_mutex);
        if( !gs.init_sentinels_done ) {
            // Set sentinels and other global variables
            gs.gindex = start;
            istart = gs.gindex;
            iend = istart + task_size;
            if (iend > vlen) {
                iend = vlen;
            }
            gs.init_sentinels_done = 1;  // sentinels have been initialised
            gs.giveup = 0;            // no giveup initially
        } else {
            gs.gindex += task_size;
            istart = gs.gindex;
            iend = istart + task_size;
            if (iend > vlen) {
                iend = vlen;
            }
        }
        // Grab one of the iterators
        iter = gs.iter[tid];
        if( iter == NULL ) {
            gs.ret_code = -1;
            gs.giveup = 1;
        }
        
        // Get temporary space for each thread
        //ret = get_temps_space(params, BLOCK_SIZE1);
        // Prepare the NumExprObject
        // CORRECTION: Do this before PyThreads are released.
        // ret = prepare_thread_mem( tid, neObj );
        // if (ret < 0) {
        //     // Propagate error to main thread 
        //     gs.ret_code = ret;
        //     gs.giveup = 1;
        // }
        pthread_mutex_unlock(&gs.count_mutex);

        while( istart < vlen && !gs.giveup ) {
            // Reset the iterator to the range for this task
            ret = NpyIter_ResetToIterIndexRange(iter, istart, iend,
                                                errorMessage);
            // Execute the task 
            if( ret >= 0 ) {
                ret = vm_engine_iter_task(iter, neObj, pc_error, errorMessage);
            }

            if( ret < 0 ) {
                pthread_mutex_lock(&gs.count_mutex);
                gs.giveup = 1;
                // Propagate error to main thread
                gs.ret_code = ret;
                pthread_mutex_unlock(&gs.count_mutex);
                break;
            }

            pthread_mutex_lock(&gs.count_mutex);
            gs.gindex += task_size;
            istart = gs.gindex;
            iend = istart + task_size;
            if (iend > vlen) {
                iend = vlen;
            }
            pthread_mutex_unlock(&gs.count_mutex);
        }

        // Meeting point for all threads (wait for finalization)
        // printf( "TH_WORKER %d:: Acquire finalization lock. count = %d\n", tid, gs.count_threads );
        pthread_mutex_lock(&gs.count_threads_mutex);
        if (gs.count_threads > 0) { // Not the last thread to join
            gs.count_threads--;
            pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
        } else { // We're the last thread, get out of here.
            // printf( "TH_WORKER %d:: Final thread releasing all\n", tid );
            pthread_cond_broadcast(&gs.count_threads_cv);
        }
        // printf( "TH_WORKER %d:: Release finalization lock. count = %d\n", tid, gs.count_threads );
        pthread_mutex_unlock(&gs.count_threads_mutex);

        // Release resources
        //free_temps_space(params);
        // TODO: keep an array of NumExprObjs for threads instead of creating/destroying structs often.
        //free(params);

    }  // closes while(1)

    // This should never be reached
    return(0);
}
        
// Initialize threads and allocate space for arrays in global_state
int reinit_threads(int n_thread_old) {
    int tid, rc;

    // printf( "Called reinit_threads.\n" );
    // TODO: we also need to cancel the pthread_create 

    // Don't free the first register, this belongs to a NumExprObject.
    // Is this always true?
    for( int I=1; I < n_thread_old; I++ ) {
        free( gs.params[I].registers );
    }

    // Initialize mutex and condition variable objects
    pthread_mutex_init(&gs.count_mutex, NULL);

    // Barrier initialization
    pthread_mutex_init(&gs.count_threads_mutex, NULL);
    pthread_cond_init(&gs.count_threads_cv, NULL);
    gs.count_threads = 0;      // Reset threads counter

    // Allocate memory
    gs.threads = (pthread_t*)realloc( gs.threads, gs.n_thread * sizeof(pthread_t) );
    gs.tids = (int*)realloc( gs.tids, gs.n_thread * sizeof(int*) );


    gs.params = (NumExprObject*)realloc( gs.params, gs.n_thread  * sizeof(NumExprObject) );
    // params[R].program is shared.
    for( int I=1; I < gs.n_thread; I++ ) {
        // Since NPY_MAXARGS=32, we can just allocate a maximum space and only 
        // consume a kB or so of RAM.
        gs.params[I].registers = (NumExprReg*)malloc( NPY_MAXARGS * sizeof(NumExprReg) );
        
    }

    gs.stridesArray = (npy_intp*)realloc( gs.stridesArray, gs.n_thread * sizeof(npy_intp) );
    gs.iter = (NpyIter**)realloc( gs.iter, gs.n_thread  * sizeof(NpyIter*) );
    gs.reduce_iter = (NpyIter**)realloc( gs.reduce_iter, gs.n_thread * sizeof(NpyIter*) );

    // Finally, create the threads
    for (tid = 0; tid < gs.n_thread; tid++) {
        gs.tids[tid] = tid;
        rc = pthread_create(&gs.threads[tid], NULL, th_worker,
                            (void *)&gs.tids[tid]);
        if (rc) {
            fprintf(stderr,
                    "ERROR; return code from pthread_create() is %d\n", rc);
            fprintf(stderr, "\tError detail: %s\n", strerror(rc));
            exit(-1);
        }
    }

    gs.init_threads_done = 1;         // Initialization done!
    gs.pid = (int)getpid();           // save the PID for this process

    return(0);
}
        

int numexpr_set_nthreads(int n_thread_new) {
    // Set the number of threads in numexpr's VM
    int n_thread_old = gs.n_thread;
    int T, rc;
    void *status;

    // printf( "set_nthreads: old: %d, new: %d\n", gs.n_thread, n_thread_new );

    gs.n_thread = n_thread_new;
    /*
    if (n_thread_new > MAX_THREADS) {
        fprintf(stderr,
                "Error.  nthreads cannot be larger than MAX_THREADS (%d)",
                MAX_THREADS);
        return -1;
    }
    */
    if( n_thread_new <= 0 ) {
        fprintf(stderr, "Error.  nthreads must be a positive integer");
        return -1;
    }

    // Only join threads if they are not initialized or if our PID is
    //   different from that in pid var (probably means that we are a
    //   subprocess, and thus threads are non-existent).
    if( n_thread_old > 1 && gs.init_threads_done && gs.pid == getpid() ) {
        // printf( "SET_NTHREADS:: JOINING OLD THREADS\n ");
        // Tell all existing threads to finish 
        gs.end_threads = 1;

        // Tell all threads to start, so they see the gs.end_threads flag.
        pthread_cond_broadcast(&gs.count_threads_cv);

        // Join exiting threads 
        // printf( "SET_NTHREADS:: Try to lock .\n" );
        // Since we're calling pthread_join we don't need need mutex protection.
        //pthread_mutex_lock(&gs.count_threads_mutex);
        for( T=0; T < n_thread_old; T++ ) {
            rc = pthread_join(gs.threads[T], &status);
            if (rc) {
                fprintf(stderr,
                        "ERROR; return code from pthread_join() is %d\n",
                        rc);
                fprintf(stderr, "\tError detail: %s\n", strerror(rc));
                exit(-1);
            }
        }
        gs.init_threads_done = 0;
        gs.end_threads = 0;
        // printf( "SET_NTHREADS:: Unlocking, threads are joined .\n" );
        //pthread_mutex_unlock(&gs.count_threads_mutex);

    } else {
        // printf( "set_n_threads NOT JOING OLD THREADS.\n ");
    }

    // Launch a new pool of threads
    if( gs.n_thread > 1 ) {
        reinit_threads(n_thread_old);
    }
    // if (gs.n_thread > 1 && (!gs.init_threads_done || gs.pid != getpid())) {
    //     reinit_threads(n_thread_old);
    // }

    return n_thread_old;
}

static PyObject*
_set_num_threads(PyObject *self, PyObject *args) {
    int n_threads_new, n_threads_old;
    if (!PyArg_ParseTuple(args, "i", &n_threads_new))
    return NULL;
    n_threads_old = numexpr_set_nthreads(n_threads_new);
    return Py_BuildValue("i", n_threads_old);
}

Py_ssize_t numexpr_set_tempsize(Py_ssize_t newSize) {
    // The bytes of pre-allocated space for temporaries PER THREAD.
    Py_ssize_t oldSize = gs.tempSize;

    pthread_mutex_lock(&gs.count_threads_mutex);
    if( newSize <= 0) {
        // Free space for hibernation
        gs.tempSize = 0;
        free( gs.tempStack );

    } else if( oldSize <= 0 ) {
        // Space was deallocated or never initialized.
        gs.tempSize = newSize;
        gs.tempStack = (char *)malloc(gs.tempSize * gs.n_thread);

    } else {
        // We can use realloc here.
        gs.tempSize = newSize;
        gs.tempStack = (char *)realloc(gs.tempStack, gs.tempSize * gs.n_thread);
    }
    pthread_mutex_unlock(&gs.count_threads_mutex);
    return oldSize;
}

static PyObject*
_set_tempsize(PyObject *self, PyObject *args) {
    Py_ssize_t newSize, oldSize;
    if (!PyArg_ParseTuple(args, "n", &newSize))
    return NULL;
    oldSize = numexpr_set_tempsize(newSize);
    return Py_BuildValue("n", oldSize );
}

#ifdef USE_VML
static PyObject*
_get_vml_version(PyObject *self, PyObject *args) {
    int len=198;
    char buf[198];
    mkl_get_version_string(buf, len);
    return Py_BuildValue("s", buf);
}

static PyObject*
_set_vml_accuracy_mode(PyObject *self, PyObject *args) {
    int mode_in, mode_old;
    if (!PyArg_ParseTuple(args, "i", &mode_in))
    return NULL;
    mode_old = vmlGetMode() & VML_ACCURACY_MASK;
    vmlSetMode((mode_in & VML_ACCURACY_MASK) | VML_ERRMODE_IGNORE );
    return Py_BuildValue("i", mode_old);
}

static PyObject*
_set_vml_num_threads(PyObject *self, PyObject *args) {
    int max_num_threads;
    if (!PyArg_ParseTuple(args, "i", &max_num_threads))
    return NULL;
    mkl_domain_set_num_threads(max_num_threads, MKL_DOMAIN_VML);
    Py_RETURN_NONE;
}

#endif

static PyMethodDef module_methods[] = {
#ifdef USE_VML
    {"_get_vml_version", _get_vml_version, METH_VARARGS,
     "Get the VML/MKL library version."},
    {"_set_vml_accuracy_mode", _set_vml_accuracy_mode, METH_VARARGS,
     "Set accuracy mode for VML functions."},
    {"_set_vml_num_threads", _set_vml_num_threads, METH_VARARGS,
     "Suggests a maximum number of threads to be used in VML operations."},
#endif
    {"_set_num_threads", _set_num_threads, METH_VARARGS,
     "Suggests a maximum number of threads to be used in operations. Returns old value."},
    {"_set_tempsize", _set_tempsize, METH_VARARGS, 
     "Set the stack/pool for temporaries in bytes per thread. Returns old value."},
    {NULL}
};


#ifdef __cplusplus
extern "C" {
#endif

#if PY_MAJOR_VERSION >= 3

// Handle the "global_state" state via moduedef 
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "interpreter",
        NULL,
        -1,                 // sizeof(struct global_state), 
        module_methods,
        NULL,
        NULL,               // module_traverse, 
        NULL,               // module_clear, 
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_interpreter(void)

#else
#define INITERROR return

PyMODINIT_FUNC
initinterpreter()
#endif
{
    PyObject *m, *d;

    // WARNING: PyType_Ready MUST be called to finalize new Python types before
    // a module is created. Official documentation is weak on this point.
    if (PyType_Ready(&NumExprType) < 0)
        INITERROR;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("interpreter", module_methods, NULL);
#endif

    if (m == NULL)
        INITERROR;

    Py_INCREF(&NumExprType);
    PyModule_AddObject(m, "CompiledExec", (PyObject *)&NumExprType);

    import_array();


    // Let's export the block sizes to Python side for benchmarking comparisons
    PyModule_AddIntConstant(m, "__BLOCK_SIZE1__", BLOCK_SIZE1 );
    PyModule_AddIntConstant(m, "__BLOCK_SIZE2__", BLOCK_SIZE2 );

    d = PyDict_New();
    if (!d) INITERROR;

    if (PyModule_AddObject(m, "allaxes", PyLong_FromLong(255)) < 0) INITERROR;
    if (PyModule_AddObject(m, "maxdims", PyLong_FromLong(NPY_MAXDIMS)) < 0) INITERROR;

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

#ifdef __cplusplus
}  // extern "C"
#endif
