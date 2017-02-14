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

// Global state. The file interp_header_GENERATED.hpp also has some global state
// in its 'th_params' variable
global_state gs;

// Do the worker job for a certain thread
void *th_worker(void *tidptr)
{
    int tid = *(int *)tidptr;
    // Parameters for threads
    npy_intp start;
    npy_intp vlen;
    npy_intp block_size;
    NpyIter *iter;
    NumExprObject *params;

    int *pc_error;
    int ret;
    //npy_uint16 n_ndarray, n_reg, n_scalar, n_temp;
    //size_t memsize;
    //char **mem;
    //npy_intp *memsteps;
    npy_intp istart, iend;
    char **errorMessage;
    // For output buffering if needed
    vector<char> out_buffer;

    while (1) {
<<<<<<< HEAD:numexpr/module.cpp

        /* Sentinels have to be initialised yet */
        gs.init_sentinels_done = 0;
=======
        
        if (tid==0) {
            gs.init_sentinels_done = 0; // sentinels have to be initialised yet
        }
>>>>>>> 3b5260be1d8bdf82b50269b47799b508c6715348:numexpr3/module.cpp

        // Meeting point for all threads (wait for initialization)
        pthread_mutex_lock(&gs.count_threads_mutex);
        if (gs.count_threads < gs.n_thread) {
            gs.count_threads++;
            pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
        }
        else {
            pthread_cond_broadcast(&gs.count_threads_cv);
        }
        pthread_mutex_unlock(&gs.count_threads_mutex);

        // Check if thread has been asked to return
        if (gs.end_threads) {
            return(0);
        }

        // Get parameters for this thread before entering the main loop 
        start = th_params.start;
        vlen = th_params.vlen;
        block_size = th_params.block_size;
        params = NumExprObject_copy_threadsafe( th_params.params );
        pc_error = th_params.pc_error;

        // If output buffering is needed, allocate it
        // RAM: is this thread-safe???  params is the same thing 
        if (th_params.need_output_buffering) {
            out_buffer.resize( GET_RETURN_REG(params).itemsize * BLOCK_SIZE1);
            params->outBuffer = (char *)(&out_buffer[0]);
        } else {
            params->outBuffer = NULL;
        }

        // Populate private data for each thread
        // RAM: unnecessary if we just safe_copy params
        // n_ndarray = params.n_ndarray;
        //n_scalar = params.n_scalar;
        //n_temp = params.n_temp;
        //n_reg = params.n_reg;
        
        // RAM ok this is all different.  
        // RAM: allocate space for temporaries for each thread
        //memsize = (1 + n_reg) * sizeof(npy_intp);
        // malloc seems thread safe for POSIX, but for Win?
        //mem = (char **)malloc(memsize);
        //memcpy(mem, params.mem, memsize);
        //memsize = copy_strides(params);

        errorMessage = th_params.errorMessage;

        //params.mem = mem;

        // Loop over blocks
        pthread_mutex_lock(&gs.count_mutex);
        if (!gs.init_sentinels_done) {
            // Set sentinels and other global variables
            gs.gindex = start;
            istart = gs.gindex;
            iend = istart + block_size;
            if (iend > vlen) {
                iend = vlen;
            }
            gs.init_sentinels_done = 1;  // sentinels have been initialised
            gs.giveup = 0;            // no giveup initially
        } else {
            gs.gindex += block_size;
            istart = gs.gindex;
            iend = istart + block_size;
            if (iend > vlen) {
                iend = vlen;
            }
        }
        // Grab one of the iterators
        iter = th_params.iter[tid];
        if (iter == NULL) {
            th_params.ret_code = -1;
            gs.giveup = 1;
        }
        //
        //memsteps = th_params.memsteps[tid];
        // Get temporary space for each thread
        ret = get_temps_space(params, BLOCK_SIZE1);
        if (ret < 0) {
            // Propagate error to main thread 
            th_params.ret_code = ret;
            gs.giveup = 1;
        }
        pthread_mutex_unlock(&gs.count_mutex);

        while (istart < vlen && !gs.giveup) {
            // Reset the iterator to the range for this task
            ret = NpyIter_ResetToIterIndexRange(iter, istart, iend,
                                                errorMessage);
            // Execute the task 
            if (ret >= 0) {
                ret = vm_engine_iter_task(iter, params, pc_error, errorMessage);
            }

            if (ret < 0) {
                pthread_mutex_lock(&gs.count_mutex);
                gs.giveup = 1;
                // Propagate error to main thread
                th_params.ret_code = ret;
                pthread_mutex_unlock(&gs.count_mutex);
                break;
            }

            pthread_mutex_lock(&gs.count_mutex);
            gs.gindex += block_size;
            istart = gs.gindex;
            iend = istart + block_size;
            if (iend > vlen) {
                iend = vlen;
            }
            pthread_mutex_unlock(&gs.count_mutex);
        }

        // Meeting point for all threads (wait for finalization)
        pthread_mutex_lock(&gs.count_threads_mutex);
        if (gs.count_threads > 0) {
            gs.count_threads--;
            pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
        }
        else {
            pthread_cond_broadcast(&gs.count_threads_cv);
        }
        pthread_mutex_unlock(&gs.count_threads_mutex);

        // Release resources */
        free_temps_space(params);
        // TODO: keep an array of NumExprObjs for threads instead of creating/destroying structs often.
        free(params);

    }  // closes while(1)

    // This should never be reached
    return(0);
}
        
// Initialize threads 
int init_threads(void)
{
    int tid, rc;

    // Initialize mutex and condition variable objects
    pthread_mutex_init(&gs.count_mutex, NULL);
    pthread_mutex_init(&gs.parallel_mutex, NULL);

    // Barrier initialization
    pthread_mutex_init(&gs.count_threads_mutex, NULL);
    pthread_cond_init(&gs.count_threads_cv, NULL);
    gs.count_threads = 0;      // Reset threads counter

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
        
// Set the number of threads in numexpr's VM
int numexpr_set_nthreads(int n_thread_new)
{
    int n_thread_old = gs.n_thread;
    int t, rc;
    void *status;

    if (n_thread_new > MAX_THREADS) {
        fprintf(stderr,
                "Error.  nthreads cannot be larger than MAX_THREADS (%d)",
                MAX_THREADS);
        return -1;
    }
    else if (n_thread_new <= 0) {
        fprintf(stderr, "Error.  nthreads must be a positive integer");
        return -1;
    }

    // Only join threads if they are not initialized or if our PID is
    //   different from that in pid var (probably means that we are a
    //   subprocess, and thus threads are non-existent).
    if (gs.n_thread > 1 && gs.init_threads_done && gs.pid == getpid()) {
        // Tell all existing threads to finish 
        gs.end_threads = 1;
        pthread_mutex_lock(&gs.count_threads_mutex);
        if (gs.count_threads < gs.n_thread) {
            gs.count_threads++;
            pthread_cond_wait(&gs.count_threads_cv, &gs.count_threads_mutex);
        }
        else {
            pthread_cond_broadcast(&gs.count_threads_cv);
        }
        pthread_mutex_unlock(&gs.count_threads_mutex);

        // Join exiting threads 
        for (t=0; t<gs.n_thread; t++) {
            rc = pthread_join(gs.threads[t], &status);
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
    }

    // Launch a new pool of threads (if necessary)
    gs.n_thread = n_thread_new;
    if (gs.n_thread > 1 && (!gs.init_threads_done || gs.pid != getpid())) {
        init_threads();
    }

    return n_thread_old;
}


#ifdef USE_VML

static PyObject *
_get_vml_version(PyObject *self, PyObject *args)
{
    int len=198;
    char buf[198];
    mkl_get_version_string(buf, len);
    return Py_BuildValue("s", buf);
}

static PyObject *
_set_vml_accuracy_mode(PyObject *self, PyObject *args)
{
    int mode_in, mode_old;
    if (!PyArg_ParseTuple(args, "i", &mode_in))
    return NULL;
    mode_old = vmlGetMode() & VML_ACCURACY_MASK;
    vmlSetMode((mode_in & VML_ACCURACY_MASK) | VML_ERRMODE_IGNORE );
    return Py_BuildValue("i", mode_old);
}

static PyObject *
_set_vml_num_threads(PyObject *self, PyObject *args)
{
    int max_num_threads;
    if (!PyArg_ParseTuple(args, "i", &max_num_threads))
    return NULL;
    mkl_domain_set_num_threads(max_num_threads, MKL_DOMAIN_VML);
    Py_RETURN_NONE;
}

#endif



static PyObject *
_set_num_threads(PyObject *self, PyObject *args)
{
    int n_threads_new, n_threads_old;
    if (!PyArg_ParseTuple(args, "i", &n_threads_new))
    return NULL;
    n_threads_old = numexpr_set_nthreads(n_threads_new);
    return Py_BuildValue("i", n_threads_old);
}

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
     "Suggests a maximum number of threads to be used in operations."},
    {NULL}
};

/* RAM: add_symbol was for opcodes and isn't used anymore.
static int
add_symbol(PyObject *d, const char *sname, int name, const char* routine_name)
{
    PyObject *o, *s;
    int r;

    if (!sname) {
        return 0;
    }

    o = PyLong_FromLong(name);
    s = PyBytes_FromString(sname);
    if (!s) {
        PyErr_SetString(PyExc_RuntimeError, routine_name);
        return -1;
    }
    r = PyDict_SetItem(d, s, o);
    Py_XDECREF(o);
    return r;
}
*/

#ifdef __cplusplus
extern "C" {
#endif

#if PY_MAJOR_VERSION >= 3

<<<<<<< HEAD:numexpr/module.cpp
/* XXX: handle the "global_state" state via moduledef */
=======
// Handle the "global_state" state via moduedef 
>>>>>>> 3b5260be1d8bdf82b50269b47799b508c6715348:numexpr3/module.cpp
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
    PyModule_AddObject(m, "NumExpr", (PyObject *)&NumExprType);

    import_array();

    //Let's export the block sizes to Python side for benchmarking comparisons
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
