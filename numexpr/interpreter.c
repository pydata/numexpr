#include "Python.h"
#include "structmember.h"
#include "numpy/noprefix.h"
#include "numpy/arrayscalars.h"
#include "math.h"
#include "string.h"
#include "assert.h"

#if defined(_WIN32)
  #include "win32/pthread.h"
  #include <process.h>
  #define getpid _getpid
#else
  #include <pthread.h>
  #include "unistd.h"
#endif

#include "complex_functions.inc"

#ifdef SCIPY_MKL_H
#define USE_VML
#endif

#ifdef USE_VML
#include "mkl_vml.h"
#include "mkl_service.h"
#endif

#ifdef _WIN32
#define inline __inline
#include "missing_posix_functions.inc"
#include "msvc_function_stubs.inc"
#endif

#define L1_SIZE 32*1024         /* The average L1 cache size */

#ifdef USE_VML
/* The values below have been tuned for a nowadays Core2 processor */
/* Note: with VML functions a larger block size (e.g. 4096) allows to make use
 * of the automatic multithreading capabilities of the VML library */
#define BLOCK_SIZE1 4096
#define BLOCK_SIZE2 32
#else
/* The values below have been tuned for a nowadays Core2 processor */
/* Note: without VML available a smaller block size is best, specially
 * for the strided and unaligned cases.  Recent implementation of
 * multithreading make it clear that larger block sizes benefit
 * performance (although it seems like we don't need very large sizes
 * like VML yet). */
#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 16
#endif


/* The maximum number of threads (for some static arrays) */
#define MAX_THREADS 256

/* Global variables for threads */
int nthreads = 1;                /* number of desired threads in pool */
int init_threads_done = 0;       /* pool of threads initialized? */
int end_threads = 0;             /* should exisiting threads end? */
pthread_t threads[MAX_THREADS];  /* opaque structure for threads */
int tids[MAX_THREADS];           /* ID per each thread */
intp gindex;                     /* global index for all threads */
int init_sentinels_done;         /* sentinels initialized? */
int giveup;                      /* should parallel code giveup? */
int force_serial;                /* force serial code instead of parallel? */
int pid = 0;                     /* the PID for this process */

/* Syncronization variables */
pthread_mutex_t count_mutex;
int count_threads;
pthread_mutex_t count_threads_mutex;
pthread_cond_t count_threads_cv;



/* This file and interp_body should really be generated from a description of
   the opcodes -- there's too much repetition here for manually editing */


enum OpCodes {
#define OPCODE(n, e, ...) e = n,
#include "opcodes.inc"
#undef OPCODE
};

/* bit of a misnomer; includes the return value. */
#define max_args 4

static char op_signature_table[][max_args] = {
#define Tb 'b'
#define Ti 'i'
#define Tl 'l'
#define Tf 'f'
#define Td 'd'
#define Tc 'c'
#define Ts 's'
#define Tn 'n'
#define T0 0
#define OPCODE(n, e, ex, rt, a1, a2, a3) {rt, a1, a2, a3},
#include "opcodes.inc"
#undef OPCODE
#undef Tb
#undef Ti
#undef Tl
#undef Tf
#undef Td
#undef Tc
#undef Ts
#undef Tn
#undef T0
};

/* returns the sig of the nth op, '\0' if no more ops -1 on failure */
static int
op_signature(int op, unsigned int n) {
    if (n >= max_args) {
        return 0;
    }
    if (op < 0 || op > OP_END) {
        return -1;
    }
    return op_signature_table[op][n];
}



/*
   To add a function to the lookup table, add to FUNC_CODES (first
   group is 1-arg functions, second is 2-arg functions), also to
   functions_f or functions_ff as appropriate. Finally, use add_func
   down below to add to funccodes. Functions with more arguments
   aren't implemented at present, but should be easy; just copy the 1-
   or 2-arg case.

   Some functions (for example, sqrt) are repeated in this table that
   are opcodes, but there's no problem with that as the compiler
   selects opcodes over functions, and this makes it easier to compare
   opcode vs. function speeds.
*/

enum FuncFFCodes {
#define FUNC_FF(fop, ...) fop,
#include "functions.inc"
#undef FUNC_FF
};

typedef float (*FuncFFPtr)(float);

#ifdef _WIN32
FuncFFPtr functions_ff[] = {
#define FUNC_FF(fop, s, f, f_win32, ...) f_win32,
#include "functions.inc"
#undef FUNC_FF
};
#else
FuncFFPtr functions_ff[] = {
#define FUNC_FF(fop, s, f, ...) f,
#include "functions.inc"
#undef FUNC_FF
};
#endif

#ifdef USE_VML
typedef void (*FuncFFPtr_vml)(int, const float*, float*);
FuncFFPtr_vml functions_ff_vml[] = {
#define FUNC_FF(fop, s, f, f_win32, f_vml) f_vml,
#include "functions.inc"
#undef FUNC_FF
};
#endif

enum FuncFFFCodes {
#define FUNC_FFF(fop, ...) fop,
#include "functions.inc"
#undef FUNC_FFF
};

typedef float (*FuncFFFPtr)(float, float);

#ifdef _WIN32
FuncFFFPtr functions_fff[] = {
#define FUNC_FFF(fop, s, f, f_win32, ...) f_win32,
#include "functions.inc"
#undef FUNC_FFF
};
#else
FuncFFFPtr functions_fff[] = {
#define FUNC_FFF(fop, s, f, ...) f,
#include "functions.inc"
#undef FUNC_FFF
};
#endif

#ifdef USE_VML
/* fmod not available in VML */
static void vsfmod(int n, const float* x1, const float* x2, float* dest)
{
    int j;
    for(j=0; j < n; j++) {
	dest[j] = fmod(x1[j], x2[j]);
    };
};

typedef void (*FuncFFFPtr_vml)(int, const float*, const float*, float*);
FuncFFFPtr_vml functions_fff_vml[] = {
#define FUNC_FFF(fop, s, f, f_win32, f_vml) f_vml,
#include "functions.inc"
#undef FUNC_FFF
};
#endif


enum FuncDDCodes {
#define FUNC_DD(fop, ...) fop,
#include "functions.inc"
#undef FUNC_DD
};

typedef double (*FuncDDPtr)(double);

FuncDDPtr functions_dd[] = {
#define FUNC_DD(fop, s, f, ...) f,
#include "functions.inc"
#undef FUNC_DD
};

#ifdef USE_VML
typedef void (*FuncDDPtr_vml)(int, const double*, double*);
FuncDDPtr_vml functions_dd_vml[] = {
#define FUNC_DD(fop, s, f, f_vml) f_vml,
#include "functions.inc"
#undef FUNC_DD
};
#endif

enum FuncDDDCodes {
#define FUNC_DDD(fop, ...) fop,
#include "functions.inc"
#undef FUNC_DDD
};

typedef double (*FuncDDDPtr)(double, double);

FuncDDDPtr functions_ddd[] = {
#define FUNC_DDD(fop, s, f, ...) f,
#include "functions.inc"
#undef FUNC_DDD
};

#ifdef USE_VML
/* fmod not available in VML */
static void vdfmod(int n, const double* x1, const double* x2, double* dest)
{
    int j;
    for(j=0; j < n; j++) {
	dest[j] = fmod(x1[j], x2[j]);
    };
};

typedef void (*FuncDDDPtr_vml)(int, const double*, const double*, double*);
FuncDDDPtr_vml functions_ddd_vml[] = {
#define FUNC_DDD(fop, s, f, f_vml) f_vml,
#include "functions.inc"
#undef FUNC_DDD
};
#endif


enum FuncCCCodes {
#define FUNC_CC(fop, ...) fop,
#include "functions.inc"
#undef FUNC_CC
};


typedef void (*FuncCCPtr)(cdouble*, cdouble*);

FuncCCPtr functions_cc[] = {
#define FUNC_CC(fop, s, f, ...) f,
#include "functions.inc"
#undef FUNC_CC
};

#ifdef USE_VML
/* complex expm1 not available in VML */
static void vzExpm1(int n, const MKL_Complex16* x1, MKL_Complex16* dest)
{
    int j;
    vzExp(n, x1, dest);
    for (j=0; j<n; j++) {
	dest[j].real -= 1.0;
    };
};

static void vzLog1p(int n, const MKL_Complex16* x1, MKL_Complex16* dest)
{
    int j;
    for (j=0; j<n; j++) {
	dest[j].real = x1[j].real + 1;
	dest[j].imag = x1[j].imag;
    };
    vzLn(n, dest, dest);
};

/* Use this instead of native vzAbs in VML as it seems to work badly */
static void vzAbs_(int n, const MKL_Complex16* x1, MKL_Complex16* dest)
{
    int j;
    for (j=0; j<n; j++) {
        dest[j].real = sqrt(x1[j].real*x1[j].real + x1[j].imag*x1[j].imag);
	dest[j].imag = 0;
    };
};

typedef void (*FuncCCPtr_vml)(int, const MKL_Complex16[], MKL_Complex16[]);

FuncCCPtr_vml functions_cc_vml[] = {
#define FUNC_CC(fop, s, f, f_vml) f_vml,
#include "functions.inc"
#undef FUNC_CC
};
#endif


enum FuncCCCCodes {
#define FUNC_CCC(fop, ...) fop,
#include "functions.inc"
#undef FUNC_CCC
};

typedef void (*FuncCCCPtr)(cdouble*, cdouble*, cdouble*);

FuncCCCPtr functions_ccc[] = {
#define FUNC_CCC(fop, s, f) f,
#include "functions.inc"
#undef FUNC_CCC
};

typedef struct
{
    PyObject_HEAD
    PyObject *signature;    /* a python string */
    PyObject *tempsig;
    PyObject *constsig;
    PyObject *fullsig;
    PyObject *program;      /* a python string */
    PyObject *constants;    /* a tuple of int/float/complex */
    PyObject *input_names;  /* tuple of strings */
    char **mem;             /* pointers to registers */
    char *rawmem;           /* a chunks of raw memory for storing registers */
    intp *memsteps;
    intp *memsizes;
    int  rawmemsize;
    int  n_inputs;
    int  n_constants;
    int  n_temps;
} NumExprObject;

static void
NumExpr_dealloc(NumExprObject *self)
{
    Py_XDECREF(self->signature);
    Py_XDECREF(self->tempsig);
    Py_XDECREF(self->constsig);
    Py_XDECREF(self->fullsig);
    Py_XDECREF(self->program);
    Py_XDECREF(self->constants);
    Py_XDECREF(self->input_names);
    PyMem_Del(self->mem);
    PyMem_Del(self->rawmem);
    PyMem_Del(self->memsteps);
    PyMem_Del(self->memsizes);
    self->ob_type->tp_free((PyObject*)self);
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

        INIT_WITH(signature, PyString_FromString(""));
        INIT_WITH(tempsig, PyString_FromString(""));
        INIT_WITH(constsig, PyString_FromString(""));
        INIT_WITH(fullsig, PyString_FromString(""));
        INIT_WITH(program, PyString_FromString(""));
        INIT_WITH(constants, PyTuple_New(0));
        Py_INCREF(Py_None);
        self->input_names = Py_None;
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

static char
get_return_sig(PyObject* program) {
    int sig;
    char last_opcode;
    int end = PyString_Size(program);
    do {
        end -= 4;
        if (end < 0) return 'X';
    }
    while ((last_opcode = PyString_AS_STRING(program)[end]) == OP_NOOP);
    sig = op_signature(last_opcode, 0);
    if (sig <= 0) {
        return 'X';
    } else {
        return (char)sig;
    }
}

static int
size_from_char(char c)
{
    switch (c) {
        case 'b': return sizeof(char);
        case 'i': return sizeof(int);
        case 'l': return sizeof(long long);
        case 'f': return sizeof(float);
        case 'd': return sizeof(double);
        case 'c': return 2*sizeof(double);
        case 's': return 0;  /* strings are ok but size must be computed */
        default:
            PyErr_SetString(PyExc_TypeError, "signature value not in 'bilfdcs'");
            return -1;
    }
}

static int
typecode_from_char(char c)
{
    switch (c) {
        case 'b': return PyArray_BOOL;
        case 'i': return PyArray_INT;
        case 'l': return PyArray_LONGLONG;
        case 'f': return PyArray_FLOAT;
        case 'd': return PyArray_DOUBLE;
        case 'c': return PyArray_CDOUBLE;
        case 's': return PyArray_STRING;
        default:
            PyErr_SetString(PyExc_TypeError, "signature value not in 'bilfdcs'");
            return -1;
    }
}

static int
last_opcode(PyObject *program_object) {
    intp n;
    unsigned char *program;
    PyString_AsStringAndSize(program_object, (char **)&program, &n);
    return program[n-4];
}

static int
get_reduction_axis(PyObject* program) {
    int end = PyString_Size(program);
    int axis = ((unsigned char *)PyString_AS_STRING(program))[end-1];
    if (axis != 255 && axis >= MAX_DIMS)
        axis = MAX_DIMS - axis;
    return axis;
}



static int
check_program(NumExprObject *self)
{
    unsigned char *program;
    intp prog_len, n_buffers, n_inputs;
    int pc, arg, argloc, argno, sig;
    char *fullsig, *signature;

    if (PyString_AsStringAndSize(self->program, (char **)&program,
                                 &prog_len) < 0) {
        PyErr_Format(PyExc_RuntimeError, "invalid program: can't read program");
        return -1;
    }
    if (prog_len % 4 != 0) {
        PyErr_Format(PyExc_RuntimeError, "invalid program: prog_len mod 4 != 0");
        return -1;
    }
    if (PyString_AsStringAndSize(self->fullsig, (char **)&fullsig,
                                 &n_buffers) < 0) {
        PyErr_Format(PyExc_RuntimeError, "invalid program: can't read fullsig");
        return -1;
    }
    if (PyString_AsStringAndSize(self->signature, (char **)&signature,
                                 &n_inputs) < 0) {
        PyErr_Format(PyExc_RuntimeError, "invalid program: can't read signature");
        return -1;
    }
    if (n_buffers > 255) {
        PyErr_Format(PyExc_RuntimeError, "invalid program: too many buffers");
        return -1;
    }
    for (pc = 0; pc < prog_len; pc += 4) {
        unsigned int op = program[pc];
        if (op == OP_NOOP) {
            continue;
        }
        if ((op >= OP_REDUCTION) && pc != prog_len-4) {
                PyErr_Format(PyExc_RuntimeError,
                    "invalid program: reduction operations must occur last");
                return -1;
        }
        for (argno = 0; ; argno++) {
            sig = op_signature(op, argno);
            if (sig == -1) {
                PyErr_Format(PyExc_RuntimeError, "invalid program: illegal opcode at %i (%d)", pc, op);
                return -1;
            }
            if (sig == 0) break;
            if (argno < 3) {
                argloc = pc+argno+1;
            }
            if (argno >= 3) {
                if (pc + 1 >= prog_len) {
                    PyErr_Format(PyExc_RuntimeError, "invalid program: double opcode (%c) at end (%i)", pc, sig);
                    return -1;
                }
                argloc = pc+argno+2;
            }
            arg = program[argloc];

            if (sig != 'n' && ((arg >= n_buffers) || (arg < 0))) {
                PyErr_Format(PyExc_RuntimeError, "invalid program: buffer out of range (%i) at %i", arg, argloc);
                return -1;
            }
            if (sig == 'n') {
                if (op == OP_FUNC_FFN) {
                    if (arg < 0 || arg >= FUNC_FF_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_FFFN) {
                    if (arg < 0 || arg >= FUNC_FFF_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_DDN) {
                    if (arg < 0 || arg >= FUNC_DD_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_DDDN) {
                    if (arg < 0 || arg >= FUNC_DDD_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_CCN) {
                    if (arg < 0 || arg >= FUNC_CC_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_CCCN) {
                    if (arg < 0 || arg >= FUNC_CCC_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op >= OP_REDUCTION) {
                    ;
                } else {
                    PyErr_Format(PyExc_RuntimeError, "invalid program: internal checker errror processing %i", argloc);
                    return -1;
                }
            /* The next is to avoid problems with the ('i','l') duality,
               specially in 64-bit platforms */
            } else if (((sig == 'l') && (fullsig[arg] == 'i')) ||
                       ((sig == 'i') && (fullsig[arg] == 'l'))) {
              ;
            } else if (sig != fullsig[arg]) {
                PyErr_Format(PyExc_RuntimeError,
                "invalid : opcode signature doesn't match buffer (%c vs %c) at %i", sig, fullsig[arg], argloc);
                return -1;
            }
        }
    }
    return 0;
}



static int
NumExpr_init(NumExprObject *self, PyObject *args, PyObject *kwds)
{
    int i, j, mem_offset;
    int n_inputs, n_constants, n_temps;
    PyObject *signature = NULL, *tempsig = NULL, *constsig = NULL;
    PyObject *fullsig = NULL, *program = NULL, *constants = NULL;
    PyObject *input_names = NULL, *o_constants = NULL;
    int *itemsizes = NULL;
    char **mem = NULL, *rawmem = NULL;
    intp *memsteps;
    intp *memsizes;
    int rawmemsize;
    static char *kwlist[] = {"signature", "tempsig",
                             "program",  "constants",
                             "input_names", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "SSS|OO", kwlist,
                                     &signature,
                                     &tempsig,
                                     &program, &o_constants,
                                     &input_names)) {
        return -1;
    }

    n_inputs = PyString_Size(signature);
    n_temps = PyString_Size(tempsig);

    if (o_constants) {
        if (!PySequence_Check(o_constants) ) {
                PyErr_SetString(PyExc_TypeError, "constants must be a sequence");
                return -1;
        }
        n_constants = PySequence_Length(o_constants);
        if (!(constants = PyTuple_New(n_constants)))
            return -1;
        if (!(constsig = PyString_FromStringAndSize(NULL, n_constants))) {
            Py_DECREF(constants);
            return -1;
        }
        if (!(itemsizes = PyMem_New(int, n_constants))) {
            Py_DECREF(constants);
            return -1;
        }
        for (i = 0; i < n_constants; i++) {
            PyObject *o;
            if (!(o = PySequence_GetItem(o_constants, i))) { /* new reference */
                Py_DECREF(constants);
                Py_DECREF(constsig);
                PyMem_Del(itemsizes);
                return -1;
            }
            PyTuple_SET_ITEM(constants, i, o); /* steals reference */
            if (PyBool_Check(o)) {
                PyString_AS_STRING(constsig)[i] = 'b';
                itemsizes[i] = size_from_char('b');
                continue;
            }
            if (PyInt_Check(o)) {
                PyString_AS_STRING(constsig)[i] = 'i';
                itemsizes[i] = size_from_char('i');
                continue;
            }
            if (PyLong_Check(o)) {
                PyString_AS_STRING(constsig)[i] = 'l';
                itemsizes[i] = size_from_char('l');
                continue;
            }
            /* The Float32 scalars are the only ones that should reach here */
            if (PyArray_IsScalar(o, Float32)) {
                PyString_AS_STRING(constsig)[i] = 'f';
                itemsizes[i] = size_from_char('f');
                continue;
            }
            if (PyFloat_Check(o)) {
                /* Python float constants are double precision by default */
                PyString_AS_STRING(constsig)[i] = 'd';
                itemsizes[i] = size_from_char('d');
                continue;
            }
            if (PyComplex_Check(o)) {
                PyString_AS_STRING(constsig)[i] = 'c';
                itemsizes[i] = size_from_char('c');
                continue;
            }
            if (PyString_Check(o)) {
                PyString_AS_STRING(constsig)[i] = 's';
                itemsizes[i] = PyString_GET_SIZE(o);
                continue;
            }
            PyErr_SetString(PyExc_TypeError, "constants must be of type bool/int/long/float/double/complex/str");
            Py_DECREF(constsig);
            Py_DECREF(constants);
            PyMem_Del(itemsizes);
            return -1;
        }
    } else {
        n_constants = 0;
        if (!(constants = PyTuple_New(0)))
            return -1;
        if (!(constsig = PyString_FromString(""))) {
            Py_DECREF(constants);
            return -1;
        }
    }

    fullsig = PyString_FromFormat("%c%s%s%s", get_return_sig(program),
        PyString_AS_STRING(signature), PyString_AS_STRING(constsig),
        PyString_AS_STRING(tempsig));
    if (!fullsig) {
        Py_DECREF(constants);
        Py_DECREF(constsig);
        PyMem_Del(itemsizes);
        return -1;
    }

    if (!input_names) {
        input_names = Py_None;
    }

    /* Compute the size of registers. We leave temps out (will be
       malloc'ed later on). */
    rawmemsize = 0;
    for (i = 0; i < n_constants; i++)
        rawmemsize += itemsizes[i];
    rawmemsize *= BLOCK_SIZE1;

    mem = PyMem_New(char *, 1 + n_inputs + n_constants + n_temps);
    rawmem = PyMem_New(char, rawmemsize);
    memsteps = PyMem_New(intp, 1 + n_inputs + n_constants + n_temps);
    memsizes = PyMem_New(intp, 1 + n_inputs + n_constants + n_temps);
    if (!mem || !rawmem || !memsteps || !memsizes) {
        Py_DECREF(constants);
        Py_DECREF(constsig);
        Py_DECREF(fullsig);
        PyMem_Del(itemsizes);
        PyMem_Del(mem);
        PyMem_Del(rawmem);
        PyMem_Del(memsteps);
        PyMem_Del(memsizes);
        return -1;
    }
    /*
       0                                                  -> output
       [1, n_inputs+1)                                    -> inputs
       [n_inputs+1, n_inputs+n_consts+1)                  -> constants
       [n_inputs+n_consts+1, n_inputs+n_consts+n_temps+1) -> temps
    */
    /* Fill in 'mem' and 'rawmem' for constants */
    mem_offset = 0;
    for (i = 0; i < n_constants; i++) {
        char c = PyString_AS_STRING(constsig)[i];
        int size = itemsizes[i];
        mem[i+n_inputs+1] = rawmem + mem_offset;
        mem_offset += BLOCK_SIZE1 * size;
        memsteps[i+n_inputs+1] = memsizes[i+n_inputs+1] = size;
        /* fill in the constants */
        if (c == 'b') {
            char *bmem = (char*)mem[i+n_inputs+1];
            char value = (char)PyInt_AS_LONG(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                bmem[j] = value;
            }
        } else if (c == 'i') {
            int *imem = (int*)mem[i+n_inputs+1];
            int value = (int)PyInt_AS_LONG(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                imem[j] = value;
            }
        } else if (c == 'l') {
            long long *lmem = (long long*)mem[i+n_inputs+1];
            long long value = PyLong_AsLongLong(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                lmem[j] = value;
            }
        } else if (c == 'f') {
            /* In this particular case the constant is in a NumPy scalar
             and in a regular Python object */
            float *fmem = (float*)mem[i+n_inputs+1];
            float value = PyArrayScalar_VAL(PyTuple_GET_ITEM(constants, i),
                                            Float);
            for (j = 0; j < BLOCK_SIZE1; j++) {
                fmem[j] = value;
            }
        } else if (c == 'd') {
            double *dmem = (double*)mem[i+n_inputs+1];
            double value = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < BLOCK_SIZE1; j++) {
                dmem[j] = value;
            }
        } else if (c == 'c') {
            double *cmem = (double*)mem[i+n_inputs+1];
            Py_complex value = PyComplex_AsCComplex(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < 2*BLOCK_SIZE1; j+=2) {
                cmem[j] = value.real;
                cmem[j+1] = value.imag;
            }
        } else if (c == 's') {
            char *smem = (char*)mem[i+n_inputs+1];
            char *value = PyString_AS_STRING(PyTuple_GET_ITEM(constants, i));
            for (j = 0; j < size*BLOCK_SIZE1; j+=size) {
                memcpy(smem + j, value, size);
            }
        }
    }
    /* This is no longer needed since no unusual item sizes appear
       in temporaries (there are no string temporaries). */
    PyMem_Del(itemsizes);

    /* Fill in 'memsteps' and 'memsizes' for temps */
    for (i = 0; i < n_temps; i++) {
        char c = PyString_AS_STRING(tempsig)[i];
        int size = size_from_char(c);
        memsteps[i+n_inputs+n_constants+1] = size;
        memsizes[i+n_inputs+n_constants+1] = size;
    }
    /* See if any errors occured (e.g., in size_from_char) or if mem_offset is wrong */
    if (PyErr_Occurred() || mem_offset != rawmemsize) {
        if (mem_offset != rawmemsize) {
            PyErr_Format(PyExc_RuntimeError, "mem_offset does not match rawmemsize");
        }
        Py_DECREF(constants);
        Py_DECREF(constsig);
        Py_DECREF(fullsig);
        PyMem_Del(mem);
        PyMem_Del(rawmem);
        PyMem_Del(memsteps);
        PyMem_Del(memsizes);
        return -1;
    }


    #define REPLACE_OBJ(arg) \
    {PyObject *tmp = self->arg; \
     self->arg = arg; \
     Py_XDECREF(tmp);}
    #define INCREF_REPLACE_OBJ(arg) {Py_INCREF(arg); REPLACE_OBJ(arg);}
    #define REPLACE_MEM(arg) {PyMem_Del(self->arg); self->arg=arg;}

    INCREF_REPLACE_OBJ(signature);
    INCREF_REPLACE_OBJ(tempsig);
    REPLACE_OBJ(constsig);
    REPLACE_OBJ(fullsig);
    INCREF_REPLACE_OBJ(program);
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

    #undef REPLACE_OBJ
    #undef INCREF_REPLACE_OBJ
    #undef REPLACE_MEM

    return check_program(self);
}

static PyMemberDef NumExpr_members[] = {
    {"signature", T_OBJECT_EX, offsetof(NumExprObject, signature), READONLY, NULL},
    {"constsig", T_OBJECT_EX, offsetof(NumExprObject, constsig), READONLY, NULL},
    {"tempsig", T_OBJECT_EX, offsetof(NumExprObject, tempsig), READONLY, NULL},
    {"fullsig", T_OBJECT_EX, offsetof(NumExprObject, fullsig), READONLY, NULL},

    {"program", T_OBJECT_EX, offsetof(NumExprObject, program), READONLY, NULL},
    {"constants", T_OBJECT_EX, offsetof(NumExprObject, constants),
     READONLY, NULL},
    {"input_names", T_OBJECT, offsetof(NumExprObject, input_names), 0, NULL},
    {NULL},
};


struct index_data {
    int count;
    int size;
    int findex;
    intp *shape;
    intp *strides;
    int *index;
    char *buffer;
};

struct vm_params {
    int prog_len;
    unsigned char *program;
    int n_inputs;
    int n_constants;
    int n_temps;
    unsigned int r_end;
    char *output;
    char **inputs;
    char **mem;
    intp *memsteps;
    intp *memsizes;
    struct index_data *index_data;
};

/* Structure for parameters in worker threads */
struct thread_data {
    intp start;
    intp vlen;
    intp block_size;
    struct vm_params params;
    int ret_code;
    int *pc_error;
} th_params;


static inline unsigned int
flat_index(struct index_data *id, unsigned int j) {
    int i, k = id->count - 1;
    unsigned int findex = id->findex;
    if (k < 0) return 0;
    if (findex == -1) {
        findex = 0;
        for (i = 0; i < id->count; i++)
            findex += id->strides[i] * id->index[i];
    }
    id->index[k] += 1;
    if (id->index[k] >= id->shape[k]) {
        while (id->index[k] >= id->shape[k]) {
            id->index[k] -= id->shape[k];
            if (k < 1) break;
            id->index[--k] += 1;
        }
        id->findex = -1;
    } else {
        id->findex = findex + id->strides[k];
    }
    return findex;
}


#define DO_BOUNDS_CHECK 1

#if DO_BOUNDS_CHECK
#define BOUNDS_CHECK(arg) if ((arg) >= params.r_end) { \
        *pc_error = pc;                                                 \
        return -2;                                                      \
    }
#else
#define BOUNDS_CHECK(arg)
#endif

int
stringcmp(const char *s1, const char *s2, intp maxlen1, intp maxlen2)
{
    intp maxlen, nextpos;
    /* Point to this when the end of a string is found,
       to simulate infinte trailing NUL characters. */
    const char null = 0;

    maxlen = (maxlen1 > maxlen2) ? maxlen1 : maxlen2;
    for (nextpos = 1;  nextpos <= maxlen;  nextpos++) {
        if (*s1 < *s2)
            return -1;
        if (*s1 > *s2)
            return +1;
        s1 = (nextpos >= maxlen1) ? &null : s1+1;
        s2 = (nextpos >= maxlen2) ? &null : s2+1;
    }
    return 0;
}

/* Get space for VM temporary registers */
int
get_temps_space(struct vm_params params, char **mem, size_t block_size)
{
    int r, k = 1 + params.n_inputs + params.n_constants;

    for (r = k; r < k + params.n_temps; r++) {
        mem[r] = malloc(block_size * params.memsizes[r]);
        if (mem[r] == NULL) {
            return -1;
        }
    }
    return 0;
}

/* Free space for VM temporary registers */
void
free_temps_space(struct vm_params params, char **mem)
{
    int r, k = 1 + params.n_inputs + params.n_constants;

    for (r = k; r < k + params.n_temps; r++) {
        free(mem[r]);
    }
}

/* Serial version of VM engine */
static inline int
vm_engine_serial(intp start, intp vlen, intp block_size,
                 struct vm_params params, int *pc_error)
{
    intp index;
    char **mem = params.mem;
    get_temps_space(params, mem, block_size);
    for (index = start; index < vlen; index += block_size) {
#define BLOCK_SIZE block_size
#include "interp_body.c"
#undef BLOCK_SIZE
    }
    free_temps_space(params, mem);
    return 0;
}

/* Serial version of VM engine (specific for BLOCK_SIZE1) */
static inline int
vm_engine_serial1(intp start, intp vlen,
                  struct vm_params params, int *pc_error)
{
    intp index;
    char **mem = params.mem;
    get_temps_space(params, mem, BLOCK_SIZE1);
    for (index = start; index < vlen; index += BLOCK_SIZE1) {
#define BLOCK_SIZE BLOCK_SIZE1
#include "interp_body.c"
#undef BLOCK_SIZE
    }
    free_temps_space(params, mem);
    return 0;
}

/* Parallel version of VM engine */
static inline int
vm_engine_parallel(intp start, intp vlen, intp block_size,
                   struct vm_params params, int *pc_error)
{
    /* Populate parameters for worker threads */
    th_params.start = start;
    th_params.vlen = vlen;
    th_params.block_size = block_size;
    th_params.params = params;
    th_params.ret_code = 0;
    th_params.pc_error = pc_error;

    /* Synchronization point for all threads (wait for initialization) */
    pthread_mutex_lock(&count_threads_mutex);
    if (count_threads < nthreads) {
        count_threads++;
        pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
    }
    else {
        pthread_cond_broadcast(&count_threads_cv);
    }
    pthread_mutex_unlock(&count_threads_mutex);

    /* Synchronization point for all threads (wait for finalization) */
    pthread_mutex_lock(&count_threads_mutex);
    if (count_threads > 0) {
        count_threads--;
        pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
    }
    else {
        pthread_cond_broadcast(&count_threads_cv);
    }
    pthread_mutex_unlock(&count_threads_mutex);

    return th_params.ret_code;
}

/* VM engine for each thread (specific for BLOCK_SIZE1) */
static inline int
vm_engine_thread1(char **mem, intp index,
                  struct vm_params params, int *pc_error)
{
#define BLOCK_SIZE BLOCK_SIZE1
#include "interp_body.c"
#undef BLOCK_SIZE
    return 0;
}

/* VM engine for each threadi (general) */
static inline int
vm_engine_thread(char **mem, intp index, intp block_size,
                  struct vm_params params, int *pc_error)
{
#define BLOCK_SIZE block_size
#include "interp_body.c"
#undef BLOCK_SIZE
    return 0;
}

/* Do the worker job for a certain thread */
void *th_worker(void *tids)
{
    /* int tid = *(int *)tids; */
    intp index;                 /* private copy of gindex */
    /* Parameters for threads */
    intp start;
    intp vlen;
    intp block_size;
    struct vm_params params;
    int *pc_error;
    int ret;
    int n_inputs;
    int n_constants;
    int n_temps;
    size_t memsize;
    char **mem;

    while (1) {

        init_sentinels_done = 0;     /* sentinels have to be initialised yet */

        /* Meeting point for all threads (wait for initialization) */
        pthread_mutex_lock(&count_threads_mutex);
        if (count_threads < nthreads) {
            count_threads++;
            pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
        }
        else {
            pthread_cond_broadcast(&count_threads_cv);
        }
        pthread_mutex_unlock(&count_threads_mutex);

        /* Check if thread has been asked to return */
        if (end_threads) {
            return(0);
        }

        /* Get parameters for this thread before entering the main loop */
        start = th_params.start;
        vlen = th_params.vlen;
        block_size = th_params.block_size;
        params = th_params.params;
        pc_error = th_params.pc_error;

        /* Populate private data for each thread */
        n_inputs = params.n_inputs;
        n_constants = params.n_constants;
        n_temps = params.n_temps;
        memsize = (1+n_inputs+n_constants+n_temps) * sizeof(char *);
        /* XXX malloc seems thread safe for POSIX, but for Win? */
        mem = malloc(memsize);
        memcpy(mem, params.mem, memsize);
        /* Get temporary space for each thread */
        ret = get_temps_space(params, mem, block_size);
        if (ret < 0) {
            pthread_mutex_lock(&count_mutex);
            giveup = 1;
            /* Propagate error to main thread */
            th_params.ret_code = ret;
            pthread_mutex_unlock(&count_mutex);
        }

        /* Loop over blocks */
        pthread_mutex_lock(&count_mutex);
        if (!init_sentinels_done) {
            /* Set sentinels and other global variables */
            gindex = start;
            index = gindex;
            init_sentinels_done = 1;    /* sentinels have been initialised */
            giveup = 0;            /* no giveup initially */
        } else {
            gindex += block_size;
            index = gindex;
        }
        pthread_mutex_unlock(&count_mutex);
        while (index < vlen && !giveup) {
            if (block_size == BLOCK_SIZE1) {
                ret = vm_engine_thread1(mem, index, params, pc_error);
            }
            else {
                ret = vm_engine_thread(mem, index, block_size,
                                       params, pc_error);
            }
            if (ret < 0) {
                pthread_mutex_lock(&count_mutex);
                giveup = 1;
                /* Propagate error to main thread */
                th_params.ret_code = ret;
                pthread_mutex_unlock(&count_mutex);
            }
            pthread_mutex_lock(&count_mutex);
            gindex += block_size;
            index = gindex;
            pthread_mutex_unlock(&count_mutex);
        }

        /* Meeting point for all threads (wait for finalization) */
        pthread_mutex_lock(&count_threads_mutex);
        if (count_threads > 0) {
            count_threads--;
            pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
        }
        else {
            pthread_cond_broadcast(&count_threads_cv);
        }
        pthread_mutex_unlock(&count_threads_mutex);

        /* Release resources */
        free_temps_space(params, mem);
        free(mem);

    }  /* closes while(1) */

    /* This should never be reached, but anyway */
    return(0);
}

/* Compute expresion in [start:vlen], if possible with threads */
static inline int
vm_engine_block(intp start, intp vlen, intp block_size,
                struct vm_params params, int *pc_error)
{
    int r;

    /* From now on, we can release the GIL */
    Py_BEGIN_ALLOW_THREADS;
    /* Run the serial version when nthreads is 1 or when the total
       length to compute is small */
    if ((nthreads == 1) || (vlen <= L1_SIZE) || force_serial) {
        if (block_size == BLOCK_SIZE1) {
            r = vm_engine_serial1(start, vlen, params, pc_error);
        }
        else {
            r = vm_engine_serial(start, vlen, block_size, params, pc_error);
        }
    }
    else {
        r = vm_engine_parallel(start, vlen, block_size, params, pc_error);
    }
    /* Get the GIL again */
    Py_END_ALLOW_THREADS;
    return r;
}

static inline int
vm_engine_rest(intp start, intp blen,
               struct vm_params params, int *pc_error)
{
    intp index = start;
    intp block_size = blen - start;
    char **mem = params.mem;
    get_temps_space(params, mem, block_size);
#define BLOCK_SIZE block_size
#include "interp_body.c"
#undef BLOCK_SIZE
    free_temps_space(params, mem);
    return 0;
}

static int
run_interpreter(NumExprObject *self, intp len, char *output, char **inputs,
                struct index_data *index_data, int *pc_error)
{
    int r;
    intp plen;
    intp blen1, blen2;
    struct vm_params params;

    *pc_error = -1;
    if (PyString_AsStringAndSize(self->program, (char **)&(params.program),
                                 &plen) < 0) {
        return -1;
    }
    params.prog_len = plen;
    params.output = output;
    params.inputs = inputs;
    params.index_data = index_data;
    params.n_inputs = self->n_inputs;
    params.n_constants = self->n_constants;
    params.n_temps = self->n_temps;
    params.mem = self->mem;
    params.memsteps = self->memsteps;
    params.memsizes = self->memsizes;
    params.r_end = PyString_Size(self->fullsig);

    blen1 = len - len % BLOCK_SIZE1;
    r = vm_engine_block(0, blen1, BLOCK_SIZE1, params, pc_error);
    if (r < 0) return r;
    if (len != blen1) {
        blen2 = len - len % BLOCK_SIZE2;
        /* A block is generally too small for threading to be an advantage */
        r = vm_engine_serial(blen1, blen2, BLOCK_SIZE2, params, pc_error);
        if (r < 0) return r;
        if (len != blen2) {
            r = vm_engine_rest(blen2, len, params, pc_error);
            if (r < 0) return r;
        }
    }

    return 0;
}

/* Initialize threads */
int init_threads(void)
{
    int tid, rc;

    /* Initialize mutex and condition variable objects */
    pthread_mutex_init(&count_mutex, NULL);

    /* Barrier initialization */
    pthread_mutex_init(&count_threads_mutex, NULL);
    pthread_cond_init(&count_threads_cv, NULL);
    count_threads = 0;      /* Reset threads counter */

    /* Finally, create the threads */
    for (tid = 0; tid < nthreads; tid++) {
        tids[tid] = tid;
        rc = pthread_create(&threads[tid], NULL, th_worker,
                            (void *)&tids[tid]);
        if (rc) {
            fprintf(stderr,
                    "ERROR; return code from pthread_create() is %d\n", rc);
            fprintf(stderr, "\tError detail: %s\n", strerror(rc));
            exit(-1);
        }
    }

    init_threads_done = 1;                 /* Initialization done! */
    pid = (int)getpid();                   /* save the PID for this process */

    return(0);
}

/* Set the number of threads in numexpr's VM */
int numexpr_set_nthreads(int nthreads_new)
{
    int nthreads_old = nthreads;
    int t, rc;
    void *status;

    if (nthreads_new > MAX_THREADS) {
        fprintf(stderr,
                "Error.  nthreads cannot be larger than MAX_THREADS (%d)",
                MAX_THREADS);
        return -1;
    }
    else if (nthreads_new <= 0) {
        fprintf(stderr, "Error.  nthreads must be a positive integer");
        return -1;
    }

    /* Only join threads if they are not initialized or if our PID is
       different from that in pid var (probably means that we are a
       subprocess, and thus threads are non-existent). */
    if (nthreads > 1 && init_threads_done && pid == getpid()) {
        /* Tell all existing threads to finish */
        end_threads = 1;
        pthread_mutex_lock(&count_threads_mutex);
        if (count_threads < nthreads) {
            count_threads++;
            pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
        }
        else {
            pthread_cond_broadcast(&count_threads_cv);
        }
        pthread_mutex_unlock(&count_threads_mutex);

        /* Join exiting threads */
        for (t=0; t<nthreads; t++) {
            rc = pthread_join(threads[t], &status);
            if (rc) {
                fprintf(stderr,
                        "ERROR; return code from pthread_join() is %d\n",
                        rc);
                fprintf(stderr, "\tError detail: %s\n", strerror(rc));
                exit(-1);
            }
        }
        init_threads_done = 0;
        end_threads = 0;
    }

    /* Launch a new pool of threads (if necessary) */
    nthreads = nthreads_new;
    if (nthreads > 1 && (!init_threads_done || pid != getpid())) {
        init_threads();
    }

    return nthreads_old;
}

/* Free possible memory temporaries and thread resources */
void numexpr_free_resources(void)
{
    int t, rc;
    void *status;

    /* Finish the possible thread pool */
    if (nthreads > 1 && init_threads_done) {
        /* Tell all existing threads to finish */
        end_threads = 1;
        pthread_mutex_lock(&count_threads_mutex);
        if (count_threads < nthreads) {
            count_threads++;
            pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
        }
        else {
            pthread_cond_broadcast(&count_threads_cv);
        }
        pthread_mutex_unlock(&count_threads_mutex);

        /* Join exiting threads */
        for (t=0; t<nthreads; t++) {
            rc = pthread_join(threads[t], &status);
            if (rc) {
                fprintf(stderr,
                        "ERROR; return code from pthread_join() is %d\n", rc);
                fprintf(stderr, "\tError detail: %s\n", strerror(rc));
                exit(-1);
            }
        }

        /* Release mutex and condition variable objects */
        pthread_mutex_destroy(&count_mutex);
        pthread_mutex_destroy(&count_threads_mutex);
        pthread_cond_destroy(&count_threads_cv);

        init_threads_done = 0;
        end_threads = 0;
    }
}

/* keyword arguments are ignored! */
static PyObject *
NumExpr_run(NumExprObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *output = NULL, *a_inputs = NULL;
    struct index_data *inddata = NULL;
    unsigned int n_inputs, n_dimensions = 0;
    intp shape[MAX_DIMS];
    int i, j, r, pc_error;
    intp size;
    char **inputs = NULL;
    intp strides[MAX_DIMS]; /* clean up XXX */

    /* Check whether we need to restart threads */
    if (!init_threads_done || pid != getpid()) {
        numexpr_set_nthreads(nthreads);
    }

    /* Don't force serial mode by default */
    force_serial = 0;

    n_inputs = PyTuple_Size(args);
    if (PyString_Size(self->signature) != n_inputs) {
        return PyErr_Format(PyExc_ValueError,
                            "number of inputs doesn't match program");
    }
    if (kwds && PyObject_Length(kwds) > 0) {
        return PyErr_Format(PyExc_ValueError,
                            "keyword arguments are not accepted");
    }

    /* This is overkill - we shouldn't need to allocate all of this space,
       but this makes it easier figure out */
    a_inputs = PyTuple_New(3*n_inputs);
    if (!a_inputs) goto cleanup_and_exit;

    inputs = PyMem_New(char *, n_inputs);
    if (!inputs) goto cleanup_and_exit;

    inddata = PyMem_New(struct index_data, n_inputs+1);
    if (!inddata) goto cleanup_and_exit;
    for (i = 0; i < n_inputs+1; i++)
        inddata[i].count = 0;

    /* First, make sure everything is some sort of array so that we can work
       with their shapes. Count dimensions concurrently. */

    for (i = 0; i < n_inputs; i++) {
        PyObject *o = PyTuple_GET_ITEM(args, i); /* borrowed ref */
        PyObject *a;
        char c = PyString_AS_STRING(self->signature)[i];
        int typecode = typecode_from_char(c);
        if (typecode == -1) goto cleanup_and_exit;
        /* Convert it just in case of a non-swapped array */
        if (!PyArray_Check(o) || PyArray_TYPE(o) != PyArray_STRING) {
            a = PyArray_FROM_OTF(o, typecode, NOTSWAPPED);
        } else {
            Py_INCREF(PyArray_DESCR(o));  /* typecode is not enough */
            a = PyArray_FromAny(o, PyArray_DESCR(o), 0, 0, NOTSWAPPED, NULL);
        }
        if (!a) goto cleanup_and_exit;
        PyTuple_SET_ITEM(a_inputs, i, a);  /* steals reference */
        if (PyArray_NDIM(a) > n_dimensions)
            n_dimensions = PyArray_NDIM(a);
    }

    /* Broadcast all of the inputs to determine the output shape (this will
       require some modifications if we later allow a final reduction
       operation). If an array has too few dimensions it's shape is padded
       with ones fromthe left. All array dimensions must match, or be one. */

    for (i = 0; i < n_dimensions; i++)
        shape[i] = 1;
    for (i = 0; i < n_inputs; i++) {
        PyObject *a = PyTuple_GET_ITEM(a_inputs, i);
        unsigned int ndims = PyArray_NDIM(a);
        int delta = n_dimensions - ndims;
        for (j = 0; j < ndims; j++) {
            unsigned int n = PyArray_DIM(a, j);
            if (n == 1 || n == shape[delta+j]) continue;
            if (shape[delta+j] == 1)
                shape[delta+j] = n;
            else {
                PyErr_SetString(PyExc_ValueError,
                                "cannot broadcast inputs to common shape");
                goto cleanup_and_exit;
            }
        }
    }
    size = PyArray_MultiplyList(shape, n_dimensions);

    /* Broadcast indices of all of the arrays. We could improve efficiency
       by keeping track of what needs to be broadcast above */

    for (i = 0; i < n_inputs; i++) {
        PyObject *a = PyTuple_GET_ITEM(a_inputs, i);
        PyObject *b;
        intp strides[MAX_DIMS];
        int delta = n_dimensions - PyArray_NDIM(a);
        if (PyArray_NDIM(a)) {
            for (j = 0; j < n_dimensions; j++)
                strides[j] = (j < delta || PyArray_DIM(a, j-delta) == 1) ?
                                0 : PyArray_STRIDE(a, j-delta);
            Py_INCREF(PyArray_DESCR(a));
            b = PyArray_NewFromDescr(a->ob_type,
                                       PyArray_DESCR(a),
                                       n_dimensions, shape,
                                       strides, PyArray_DATA(a), 0, a);
            if (!b) goto cleanup_and_exit;
        } else { /* Leave scalars alone */
            b = a;
            Py_INCREF(b);
        }
        /* Store b so that it stays alive till we're done */
        PyTuple_SET_ITEM(a_inputs, i+n_inputs, b);
    }


    for (i = 0; i < n_inputs; i++) {
        PyObject *a = PyTuple_GET_ITEM(a_inputs, i+n_inputs);
        PyObject *b;
        char c = PyString_AS_STRING(self->signature)[i];
        int typecode = typecode_from_char(c);
        if (PyArray_NDIM(a) == 0) {
            /* Broadcast scalars */
            intp dims[1] = {BLOCK_SIZE1};
            Py_INCREF(PyArray_DESCR(a));
            b = PyArray_SimpleNewFromDescr(1, dims, PyArray_DESCR(a));
            if (!b) goto cleanup_and_exit;
            self->memsteps[i+1] = 0;
            self->memsizes[i+1] = PyArray_ITEMSIZE(a);
            PyTuple_SET_ITEM(a_inputs, i+2*n_inputs, b);  /* steals reference */
            inputs[i] = PyArray_DATA(b);
            if (typecode == PyArray_BOOL) {
                char value = ((char*)PyArray_DATA(a))[0];
                for (j = 0; j < BLOCK_SIZE1; j++)
                    ((char*)PyArray_DATA(b))[j] = value;
            } else if (typecode == PyArray_INT) {
                int value = ((int*)PyArray_DATA(a))[0];
                for (j = 0; j < BLOCK_SIZE1; j++)
                    ((int*)PyArray_DATA(b))[j] = value;
            } else if (typecode == PyArray_LONGLONG) {
                long long value = ((long long*)PyArray_DATA(a))[0];
                for (j = 0; j < BLOCK_SIZE1; j++)
                    ((long long*)PyArray_DATA(b))[j] = value;
            } else if (typecode == PyArray_FLOAT) {
                float value = ((float*)PyArray_DATA(a))[0];
                for (j = 0; j < BLOCK_SIZE1; j++)
                    ((float*)PyArray_DATA(b))[j] = value;
            } else if (typecode == PyArray_DOUBLE) {
                double value = ((double*)PyArray_DATA(a))[0];
                for (j = 0; j < BLOCK_SIZE1; j++)
                    ((double*)PyArray_DATA(b))[j] = value;
            } else if (typecode == PyArray_CDOUBLE) {
                double rvalue = ((double*)PyArray_DATA(a))[0];
                double ivalue = ((double*)PyArray_DATA(a))[1];
                for (j = 0; j < 2*BLOCK_SIZE1; j+=2) {
                    ((double*)PyArray_DATA(b))[j] = rvalue;
                    ((double*)PyArray_DATA(b))[j+1] = ivalue;
                }
            } else if (typecode == PyArray_STRING) {
                int itemsize = PyArray_ITEMSIZE(a);
                char *value = (char*)(PyArray_DATA(a));
                for (j = 0; j < itemsize*BLOCK_SIZE1; j+=itemsize)
                    memcpy((char*)(PyArray_DATA(b)) + j, value, itemsize);
            } else {
                PyErr_SetString(PyExc_RuntimeError, "illegal typecode value");
                goto cleanup_and_exit;
            }
        } else {
            /* Check that discontiguous strides appear only on the last
               dimension. If not, the arrays should be copied.
               Furthermore, such arrays can appear when doing
               broadcasting above, so this check really needs to be
               here, and not in Python space. */
            intp inner_size;
            for (j = PyArray_NDIM(a)-2; j >= 0; j--) {
                inner_size = PyArray_STRIDE(a, j+1) * PyArray_DIM(a, j+1);
                if (PyArray_STRIDE(a, j) != inner_size) {
                    intp dims[1] = {BLOCK_SIZE1};
                    inddata[i+1].count = PyArray_NDIM(a);
                    inddata[i+1].findex = -1;
                    inddata[i+1].size = PyArray_ITEMSIZE(a);
                    inddata[i+1].shape = PyArray_DIMS(a);
                    inddata[i+1].strides = PyArray_STRIDES(a);
                    inddata[i+1].buffer = PyArray_BYTES(a);
                    inddata[i+1].index = PyMem_New(int, inddata[i+1].count);
                    for (j = 0; j < inddata[i+1].count; j++)
                        inddata[i+1].index[j] = 0;
                    Py_INCREF(PyArray_DESCR(a));
                    a = PyArray_SimpleNewFromDescr(1, dims, PyArray_DESCR(a));
                    /* steals reference below */
                    PyTuple_SET_ITEM(a_inputs, i+2*n_inputs, a);
                    /* Broadcasting code only seems to work well for
                       serial code (don't know exactly why) */
                    force_serial = 1;
                    break;
                }
            }

            self->memsteps[i+1] = PyArray_STRIDE(a, PyArray_NDIM(a)-1);
            self->memsizes[i+1] = PyArray_ITEMSIZE(a);
            inputs[i] = PyArray_DATA(a);

        }
    }

    if (last_opcode(self->program) > OP_REDUCTION) {
        /* A reduction can not result in a string,
           so we don't need to worry about item sizes here. */
        char retsig = get_return_sig(self->program);
        int axis = get_reduction_axis(self->program);
        /* Reduction ops only works with 1 thread */
        force_serial = 1;
        self->memsteps[0] = 0; /*size_from_char(retsig);*/
        if (axis == 255) {
            intp dims[1];
            for (i = 0; i < n_dimensions; i++)
                strides[i] = 0;
            output = PyArray_SimpleNew(0, dims, typecode_from_char(retsig));
            if (!output) goto cleanup_and_exit;
        } else {
            intp dims[MAX_DIMS];
            if (axis < 0)
                axis = n_dimensions + axis;
            if (axis < 0 || axis >= n_dimensions) {
                PyErr_SetString(PyExc_ValueError, "axis out of range");
                goto cleanup_and_exit;
            }
            for (i = j = 0; i < n_dimensions; i++) {
                if (i != axis) {
                    dims[j] = shape[i];
                    j += 1;
                }
            }
            output = PyArray_SimpleNew(n_dimensions-1, dims,
                                       typecode_from_char(retsig));
            if (!output) goto cleanup_and_exit;
            for (i = j = 0; i < n_dimensions; i++) {
                if (i != axis) {
                    strides[i] = PyArray_STRIDES(output)[j];
                    j += 1;
                } else {
                    strides[i] = 0;
                }
            }


        }
        /* TODO optimize strides -- in this and other inddata cases, strides and
           shape can be tweaked to minimize the amount of looping */
        inddata[0].count = n_dimensions;
        inddata[0].findex = -1;
        inddata[0].size = PyArray_ITEMSIZE(output);
        inddata[0].shape = shape;
        inddata[0].strides = strides;
        inddata[0].buffer = PyArray_BYTES(output);
        inddata[0].index = PyMem_New(int, n_dimensions);
        for (j = 0; j < inddata[0].count; j++)
            inddata[0].index[j] = 0;

        if (last_opcode(self->program) >= OP_SUM &&
            last_opcode(self->program) < OP_PROD) {
                PyObject *zero = PyInt_FromLong(0);
                PyArray_FillWithScalar((PyArrayObject *)output, zero);
                Py_DECREF(zero);
        } else {
                PyObject *one = PyInt_FromLong(1);
                PyArray_FillWithScalar((PyArrayObject *)output, one);
                Py_DECREF(one);
        }
    }
    else {
        char retsig = get_return_sig(self->program);
        if (retsig != 's') {
            self->memsteps[0] = self->memsizes[0] = size_from_char(retsig);
            output = PyArray_SimpleNew(
                n_dimensions, shape, typecode_from_char(retsig));
        } else {
            /* Since the *only* supported operation returning a string
             * is a copy, the size of returned strings
             * can be directly gotten from the first (and only)
             * input/constant/temporary. */
            PyArray_Descr *descr;
            if (n_inputs > 0) {  /* input, like in 'a' where a -> 'foo' */
                descr = PyArray_DESCR(PyTuple_GET_ITEM(a_inputs, 1));
            Py_INCREF(descr);
            } else {  /* constant, like in '"foo"' */
                descr = PyArray_DescrFromType(PyArray_STRING);
                descr->elsize = self->memsizes[1];
            }  /* no string temporaries, so no third case  */
            self->memsteps[0] = self->memsizes[0] = self->memsizes[1];
            output = PyArray_SimpleNewFromDescr(n_dimensions, shape, descr);
        }
        if (!output) goto cleanup_and_exit;
    }


    r = run_interpreter(self, size, PyArray_DATA(output), inputs, inddata,
                        &pc_error);

    if (r < 0) {
        Py_XDECREF(output);
        output = NULL;
        if (r == -1) {
            PyErr_SetString(PyExc_RuntimeError,
                            "an error occurred while running the program");
        } else if (r == -2) {
            PyErr_Format(PyExc_RuntimeError,
                         "bad argument at pc=%d", pc_error);
        } else if (r == -3) {
            PyErr_Format(PyExc_RuntimeError,
                         "bad opcode at pc=%d", pc_error);
        } else {
            PyErr_SetString(PyExc_RuntimeError,
                            "unknown error occurred while running the program");
        }
    }
cleanup_and_exit:
    Py_XDECREF(a_inputs);
    PyMem_Del(inputs);
    if (inddata) {
        for (i = 0; i < n_inputs+1; i++) {
            if (inddata[i].count) {
                PyMem_Del(inddata[i].index);
            }
        }
    }
    PyMem_Del(inddata);
    return output;
}

static PyMethodDef NumExpr_methods[] = {
    {"run", (PyCFunction) NumExpr_run, METH_VARARGS|METH_KEYWORDS, NULL},
    {NULL, NULL}
};

static PyTypeObject NumExprType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
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
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
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


#ifdef USE_VML

static PyObject *
_get_vml_version(PyObject *self, PyObject *args)
{
    int len=198;
    char buf[198];
    MKLGetVersionString(buf, len);
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
    mkl_domain_set_num_threads(max_num_threads, MKL_VML);
    Py_RETURN_NONE;
}

#endif

static PyObject *
_set_num_threads(PyObject *self, PyObject *args)
{
    int num_threads, nthreads_old;
    if (!PyArg_ParseTuple(args, "i", &num_threads))
	return NULL;
    nthreads_old = numexpr_set_nthreads(num_threads);
    return Py_BuildValue("i", nthreads_old);
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

static int
add_symbol(PyObject *d, const char *sname, int name, const char* routine_name)
{
    PyObject *o, *s;
    int r;

    if (!sname) {
        return 0;
    }

    o = PyInt_FromLong(name);
    s = PyString_FromString(sname);
    if (!s) {
        PyErr_SetString(PyExc_RuntimeError, routine_name);
        return -1;
    }
    r = PyDict_SetItem(d, s, o);
    Py_XDECREF(o);
    return r;
}

void
initinterpreter(void)
{
    PyObject *m, *d;

    if (PyType_Ready(&NumExprType) < 0)
        return;

    m = Py_InitModule3("interpreter", module_methods, NULL);
    if (m == NULL)
        return;

    Py_INCREF(&NumExprType);
    PyModule_AddObject(m, "NumExpr", (PyObject *)&NumExprType);

    import_array();

    d = PyDict_New();
    if (!d) return;

#define OPCODE(n, name, sname, ...)                              \
    if (add_symbol(d, sname, name, "add_op") < 0) { return; }
#include "opcodes.inc"
#undef OPCODE

    if (PyModule_AddObject(m, "opcodes", d) < 0) return;

    d = PyDict_New();
    if (!d) return;

#define add_func(name, sname)                           \
    if (add_symbol(d, sname, name, "add_func") < 0) { return; }
#define FUNC_FF(name, sname, ...)  add_func(name, sname);
#define FUNC_FFF(name, sname, ...) add_func(name, sname);
#define FUNC_DD(name, sname, ...)  add_func(name, sname);
#define FUNC_DDD(name, sname, ...) add_func(name, sname);
#define FUNC_CC(name, sname, ...)  add_func(name, sname);
#define FUNC_CCC(name, sname, ...) add_func(name, sname);
#include "functions.inc"
#undef FUNC_CCC
#undef FUNC_CC
#undef FUNC_DDD
#undef FUNC_DD
#undef FUNC_DD
#undef FUNC_FFF
#undef FUNC_FF
#undef add_func

    if (PyModule_AddObject(m, "funccodes", d) < 0) return;

    if (PyModule_AddObject(m, "allaxes", PyInt_FromLong(255)) < 0) return;
    if (PyModule_AddObject(m, "maxdims", PyInt_FromLong(MAX_DIMS)) < 0) return;

}


/*
Local Variables:
   c-basic-offset: 4
End:
*/
