// Numexpr - Fast numerical array expression evaluator for NumPy.
//
//      License: BSD
//      Author:  See AUTHORS.txt
//
//  See LICENSE.txt for details about copyright and rights to use.
//
// module.cpp contains the CPython-specific module exposure.

#define DO_NUMPY_IMPORT_ARRAY

#define CHECK_END_COND if(gs.barrier_passed == BARRIER_EXIT) return(0);

#include "module.hpp"
#include "interp_header_GENERATED.hpp"
#include "numexpr_object.hpp"
#include "benchmark.hpp"

using namespace std;

// Global state definition. Also includes what used to be in th_params.
global_state gs;
// Benchmarking data containers
#if defined(_WIN32) && defined(BENCHMARKING)
    LARGE_INTEGER TIMES[512];
    LARGE_INTEGER T_NOW;
    double FREQ;
#elif defined(BENCHMARKING) // Linux
    timespec TIMES[512];
    timespec T_NOW;
#endif


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

    while(1) {
        
        if( tid == 0 ) {
            gs.init_sentinels_done = 0; // sentinels have to be initialised yet
        }

        // Meeting point for all threads (wait for initialization)
        CHECK_END_COND;
        pthread_mutex_lock(&gs.count_threads_mutex);
        if( gs.count_threads < gs.n_thread ) {
            gs.count_threads++;
            // printf( "Count threads start %d\n", gs.count_threads );
            do {
                pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
            } while (gs.barrier_passed == BARRIER_HALT);
        }
        else { // Every thread is ready, go...
            gs.barrier_passed = BARRIER_PASS;
            pthread_cond_broadcast(&gs.count_threads_cv);
        }
        pthread_mutex_unlock(&gs.count_threads_mutex);
        // Get parameters for this thread before entering the main loop 
        start = gs.start;
        vlen = gs.vlen;
        task_size = gs.task_size;
        neObj = &gs.params[tid];
        pc_error = gs.pc_error;

        errorMessage = gs.errorMessage;

        // Loop over blocks
        CHECK_END_COND;
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
        pthread_mutex_unlock(&gs.count_mutex);
        DIFF_TIME(150+tid);

        BENCH_TIME(200+tid);
        while( istart < vlen && !gs.giveup ) {
            // Reset the iterator to the range for this task
            ret = NpyIter_ResetToIterIndexRange(iter, istart, iend,
                                                errorMessage);
            // Execute the task 
            if( ret >= 0 ) {
                
                ret = vm_engine_iter_task(iter, neObj, tid, pc_error, errorMessage);
                
            }

            if( ret < 0 ) {
                CHECK_END_COND;
                pthread_mutex_lock(&gs.count_mutex);
                gs.giveup = 1;
                // Propagate error to main thread
                gs.ret_code = ret;
                pthread_mutex_unlock(&gs.count_mutex);
                break;
            }
            CHECK_END_COND;
            pthread_mutex_lock(&gs.count_mutex);
            gs.gindex += task_size;
            istart = gs.gindex;
            iend = istart + task_size;
            if (iend > vlen) {
                iend = vlen;
            }
            pthread_mutex_unlock(&gs.count_mutex);
        }
        DIFF_TIME(200+tid);
        BENCH_TIME(250+tid);
        // Meeting point for all threads (wait for finalization)
        CHECK_END_COND;
        pthread_mutex_lock(&gs.count_threads_mutex);
        if (gs.count_threads > 0) { // Not the last thread to join
            gs.count_threads--;
            // printf( "Count threads finalize %d\n", gs.count_threads );
            do {
                pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
            } while (gs.barrier_passed == BARRIER_PASS);
        } else { // We're the last thread, get out of here.
            gs.barrier_passed = BARRIER_HALT;
            pthread_cond_broadcast(&gs.count_threads_cv);
        }
        pthread_mutex_unlock(&gs.count_threads_mutex);
        DIFF_TIME(250+tid);
    }  // closes while(1)

    // This should never be reached
    return(0);
}
        
// Initialize threads and allocate space for arrays in global_state
int reinit_threads(int n_thread_old) {
    int tid, rc;

    gs.count_threads = 0;      // Reset threads counter
    gs.barrier_passed = BARRIER_HALT;

    // Allocate memory
    gs.threads = (pthread_t*)realloc( gs.threads, gs.n_thread * sizeof(pthread_t) );
    gs.tids = (int*)realloc( gs.tids, gs.n_thread * sizeof(int*) );

    gs.params = (NumExprObject*)realloc( gs.params, gs.n_thread  * sizeof(NumExprObject) );
    // params[R].program is shared.

    // Allocate an arena for the registers
    // Since NPY_MAXARGS=32, we can just allocate a maximum space and only 
    // consume a kB or so of RAM.
    gs.registerArena = (char *)realloc( gs.registerArena, gs.n_thread * NPY_MAXARGS * sizeof(NumExprReg) );
    for( int I=0; I < gs.n_thread; I++ ) {
        gs.params[I].registers = (NumExprReg*)(gs.registerArena + I*NPY_MAXARGS*sizeof(NumExprReg) );
    }

    gs.stridesArray = (npy_intp*)realloc( gs.stridesArray, gs.n_thread * sizeof(npy_intp) );
    gs.iter = (NpyIter**)realloc( gs.iter, gs.n_thread  * sizeof(NpyIter*) );
    gs.reduce_iter = (NpyIter**)realloc( gs.reduce_iter, gs.n_thread * sizeof(NpyIter*) );

    // Finally, create the threads
    for (tid = 0; tid < gs.n_thread; tid++) {
        gs.tids[tid] = tid;
        rc = pthread_create(&gs.threads[tid], NULL, th_worker, (void *)&gs.tids[tid]);

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

    // printf("numexpr_set_nthreads #1, old =%d, new =%d\n", n_thread_old, n_thread_new );
    if( n_thread_new <= 0 ) {
        fprintf(stderr, "Error. `nthreads` must be a positive integer");
        return -1;
    }
    if( n_thread_new == n_thread_old ) {
        return n_thread_old;
    }

    // Only join threads if they are not initialized or if our PID is
    //   different from that in pid var (probably means that we are a
    //   subprocess, and thus threads are non-existent).
    if( gs.init_threads_done && gs.pid == getpid() ) {
        // Tell all existing threads to finish 
        gs.barrier_passed = BARRIER_EXIT;
        pthread_mutex_lock(&gs.count_threads_mutex);
        // Ensure that workers never wait on the condition variable
        gs.count_threads = n_thread_old; 
        // Wake all workers
        pthread_cond_broadcast(&gs.count_threads_cv);
        pthread_mutex_unlock(&gs.count_threads_mutex);

        for( T=0; T < n_thread_old; T++ ) {
            // Join exiting threads 
            rc = pthread_join(gs.threads[T], &status);
            if (rc) {
                printf( "JOIN ERROR %d\n", rc );
                fprintf(stderr, "ERROR; return code from pthread_join() is %d\n", rc);
                fprintf(stderr, "    Error detail: %s\n", strerror(rc));
                exit(-1);
            }
            
        }
        pthread_mutex_unlock(&gs.count_threads_mutex);
        pthread_mutex_unlock(&gs.count_mutex);
        gs.count_threads = 0;
        gs.n_thread = n_thread_new;
        gs.init_threads_done = 0;
        gs.barrier_passed = BARRIER_HALT;
    } 
    else {
        gs.n_thread = n_thread_new;
    }

    // Launch a new pool of threads
    reinit_threads(n_thread_old);
    
    return n_thread_old;
}
PyDoc_STRVAR(SetNumThreads__doc__,
"Sets a maximum number of threads to be used in operations. Returns old value.\n");
static PyObject*
PySet_num_threads(PyObject *self, PyObject *args) {
    int n_threads_new, n_threads_old;
    if (!PyArg_ParseTuple(args, "i", &n_threads_new))
        return NULL;

    pthread_mutex_lock(&gs.global_mutex);
    n_threads_old = numexpr_set_nthreads(n_threads_new);
    pthread_mutex_unlock(&gs.global_mutex);
    return Py_BuildValue("i", n_threads_old);
}

PyDoc_STRVAR(GetNumThreads__doc__,
"Gets a maximum number of threads to be used in operations.\n");
static PyObject*
PyGet_num_threads(PyObject *self, PyObject *args) {
        int n_thread;

        pthread_mutex_lock(&gs.global_mutex);
        n_thread = gs.n_thread;
        pthread_mutex_unlock(&gs.global_mutex);
        return Py_BuildValue("i", n_thread);
    }
    

Py_ssize_t numexpr_set_tempsize(Py_ssize_t newSize) {

    // printf( "DEBUG: setting temporary space to %d bytes.\n", newSize );
    // The bytes of pre-allocated space for temporaries PER THREAD.
    Py_ssize_t oldSize = gs.tempSize;
    if( newSize <= 0) {
        // Free space for hibernation
        gs.tempSize = 0;
        free( gs.tempArena );

    } else if( oldSize <= 0 ) {
        // Space was deallocated or never initialized.
        gs.tempSize = newSize;
        gs.tempArena = (char *)malloc(gs.tempSize);

    } else {
        // We can use realloc here.
        gs.tempSize = newSize;
        gs.tempArena = (char *)realloc(gs.tempArena, gs.tempSize);
    }
    return oldSize;
}

PyDoc_STRVAR(SetTempSize__doc__,
"set_tempsize(int size) -- Sets the size of the temporary array arena. If you \
expect to continuously increase the number of temporaries this can avoid \
realloc's of the arena. Set to zero to free all temporary array resources.\n");
static PyObject*
PySet_tempsize(PyObject *self, PyObject *args) {
    Py_ssize_t newSize, oldSize;
    if (!PyArg_ParseTuple(args, "n", &newSize))
    return NULL;

    pthread_mutex_lock(&gs.global_mutex);
    oldSize = numexpr_set_tempsize(newSize);
    pthread_mutex_unlock(&gs.global_mutex);

    return Py_BuildValue("n", oldSize );
}

#ifdef USE_VML
PyDoc_STRVAR(GetVMLVersion__doc__,
"Get the VML/MKL library version.\n");
static PyObject*
PyGet_vml_version(PyObject *self, PyObject *args) {
    int len=198;
    char buf[198];
    mkl_get_version_string(buf, len);
    return Py_BuildValue("s", buf);
}

PyDoc_STRVAR(SetVMLAccuracy__doc__, 
"Set accuracy mode for VML functions.\n");
static PyObject*
PySet_vml_accuracy_mode(PyObject *self, PyObject *args) {
    int mode_in, mode_old;
    if (!PyArg_ParseTuple(args, "i", &mode_in))
    return NULL;
    mode_old = vmlGetMode() & VML_ACCURACY_MASK;
    vmlSetMode((mode_in & VML_ACCURACY_MASK) | VML_ERRMODE_IGNORE );
    return Py_BuildValue("i", mode_old);
}

PyDoc_STRVAR(SetVMLNumThreads__doc__, 
"Suggests a maximum number of threads to be used in VML operations.\n");
static PyObject*
PySet_vml_num_threads(PyObject *self, PyObject *args) {
    int max_num_threads;
    if (!PyArg_ParseTuple(args, "i", &max_num_threads))
    return NULL;
    mkl_domain_set_num_threads(max_num_threads, MKL_DOMAIN_VML);
    Py_RETURN_NONE;
}

#endif

static PyMethodDef module_methods[] = {
#ifdef USE_VML
    {"_get_vml_version",       PyGet_vml_version,       METH_VARARGS, GetVMLVersion__doc__   },
    {"_set_vml_accuracy_mode", PySet_vml_accuracy_mode, METH_VARARGS, SetVMLAccuracy__doc__  },
    {"_set_vml_num_threads",   PySet_vml_num_threads,   METH_VARARGS, SetVMLNumThreads__doc__},
#endif
    {"set_num_threads",       PySet_num_threads,       METH_VARARGS, SetNumThreads__doc__   },
    {"set_tempsize",          PySet_tempsize,          METH_VARARGS, SetTempSize__doc__     },
    {"get_num_threads",       PyGet_num_threads,       METH_VARARGS, GetNumThreads__doc__   },
    {NULL}
};

#ifdef __cplusplus
extern "C" {
#endif

// #if PY_MAJOR_VERSION >= 3

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

PyObject*
PyInit_interpreter(void)
// #else  // Python 2.7
// #define INITERROR return

// PyMODINIT_FUNC
// initinterpreter()
// #endif
{
    // PyObject *m, *d;
    PyObject *m;

    // WARNING: PyType_Ready MUST be called to finalize new Python types before
    // a module is created. Official documentation is weak on this point.
    if (PyType_Ready(&NumExprType) < 0)
        INITERROR;

// #if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
// #else
    // m = Py_InitModule3("interpreter", module_methods, NULL);
// #endif

    if (m == NULL)
        INITERROR;

    Py_INCREF(&NumExprType);
    PyModule_AddObject(m, "CompiledExec", (PyObject *)&NumExprType);

    import_array();

    // Let's export the block sizes to Python side for benchmarking comparisons
    PyModule_AddIntConstant(m, "__BLOCK_SIZE1__", BLOCK_SIZE1 );
    PyModule_AddIntConstant(m, "__BLOCK_SIZE2__", BLOCK_SIZE2 );

    PyModule_AddIntConstant(m, "MAX_ARGS", NPY_MAXARGS );
    PyModule_AddIntConstant(m, "MAX_DIMS", NPY_MAXDIMS );

    
#ifdef BENCHMARKING
    PyModule_AddIntConstant(m, "BENCHMARKING", 1 );  
    #if defined(_WIN32) // On Windows x64: LARGE_INTEGER is a union and LARGE_INTEGER.QuadPart is int64. 
        npy_intp dims = BCOUNT;
        QueryPerformanceFrequency( &T_NOW );
        PyModule_AddIntConstant(m, "cpu_freq", T_NOW.QuadPart );
        PyObject* bench_times = PyArray_SimpleNewFromData( 1, &dims, NPY_INT64, TIMES );
    #else  // On Linux: struct timespec { long ts_sec; long ts_nsec };
        // So the array is 2*BCOUNT long and will need to be sliced on the Python end
        npy_intp dims = 2*BCOUNT;
        PyObject* bench_times = PyArray_SimpleNewFromData( 1, &dims, NPY_INT64, TIMES );
    #endif
    PyModule_AddObject(m, "bench_times", bench_times ); // Make NumPy arrays from the timing arrays.
#else // No BENCHMARKING
    PyModule_AddIntConstant(m, "BENCHMARKING", 0 );  
#endif

    // The OpTable is loaded via pickle now.
    // d = PyDict_New();
    // if (!d) INITERROR;

// #if PY_MAJOR_VERSION >= 3
    return m;
// #endif
}

#ifdef __cplusplus
}  // extern "C"
#endif
