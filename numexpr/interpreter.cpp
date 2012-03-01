/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/


#include "Python.h"
#include "structmember.h"
#include "numpy/noprefix.h"
#include "numpy/arrayscalars.h"
#include "numpy/ndarrayobject.h"
#include "numpy/npy_cpu.h"
#include "math.h"
#include "string.h"
#include "assert.h"

#include "numexpr_config.hpp"

#include "complex_functions.hpp"

#ifdef SCIPY_MKL_H
#define USE_VML
#endif

#ifdef USE_VML
#include "mkl_vml.h"
#include "mkl_service.h"
#endif

#ifdef _WIN32
  #define inline __inline
  #ifndef __MINGW32__
    #include "missing_posix_functions.hpp"
  #endif
  #include "msvc_function_stubs.hpp"
#endif

/* x86 platform works with unaligned reads and writes */
#if (defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64))
#  define USE_UNALIGNED_ACCESS 1
#endif

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

/* The maximum number of threads (for some static arrays).
 * Choose this large enough for most monsters out there.
   Keep in sync this with the number in __init__.py. */
#define MAX_THREADS 4096

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
#include "opcodes.hpp"
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
#include "opcodes.hpp"
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
#include "functions.hpp"
#undef FUNC_FF
};

typedef float (*FuncFFPtr)(float);

#ifdef _WIN32
FuncFFPtr functions_ff[] = {
#define FUNC_FF(fop, s, f, f_win32, ...) f_win32,
#include "functions.hpp"
#undef FUNC_FF
};
#else
FuncFFPtr functions_ff[] = {
#define FUNC_FF(fop, s, f, ...) f,
#include "functions.hpp"
#undef FUNC_FF
};
#endif

#ifdef USE_VML
typedef void (*FuncFFPtr_vml)(int, const float*, float*);
FuncFFPtr_vml functions_ff_vml[] = {
#define FUNC_FF(fop, s, f, f_win32, f_vml) f_vml,
#include "functions.hpp"
#undef FUNC_FF
};
#endif

enum FuncFFFCodes {
#define FUNC_FFF(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_FFF
};

typedef float (*FuncFFFPtr)(float, float);

#ifdef _WIN32
FuncFFFPtr functions_fff[] = {
#define FUNC_FFF(fop, s, f, f_win32, ...) f_win32,
#include "functions.hpp"
#undef FUNC_FFF
};
#else
FuncFFFPtr functions_fff[] = {
#define FUNC_FFF(fop, s, f, ...) f,
#include "functions.hpp"
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
#include "functions.hpp"
#undef FUNC_FFF
};
#endif


enum FuncDDCodes {
#define FUNC_DD(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_DD
};

typedef double (*FuncDDPtr)(double);

FuncDDPtr functions_dd[] = {
#define FUNC_DD(fop, s, f, ...) f,
#include "functions.hpp"
#undef FUNC_DD
};

#ifdef USE_VML
typedef void (*FuncDDPtr_vml)(int, const double*, double*);
FuncDDPtr_vml functions_dd_vml[] = {
#define FUNC_DD(fop, s, f, f_vml) f_vml,
#include "functions.hpp"
#undef FUNC_DD
};
#endif

enum FuncDDDCodes {
#define FUNC_DDD(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_DDD
};

typedef double (*FuncDDDPtr)(double, double);

FuncDDDPtr functions_ddd[] = {
#define FUNC_DDD(fop, s, f, ...) f,
#include "functions.hpp"
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
#include "functions.hpp"
#undef FUNC_DDD
};
#endif


enum FuncCCCodes {
#define FUNC_CC(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_CC
};


typedef void (*FuncCCPtr)(cdouble*, cdouble*);

FuncCCPtr functions_cc[] = {
#define FUNC_CC(fop, s, f, ...) f,
#include "functions.hpp"
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
#include "functions.hpp"
#undef FUNC_CC
};
#endif


enum FuncCCCCodes {
#define FUNC_CCC(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_CCC
};

typedef void (*FuncCCCPtr)(cdouble*, cdouble*, cdouble*);

FuncCCCPtr functions_ccc[] = {
#define FUNC_CCC(fop, s, f) f,
#include "functions.hpp"
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
    Py_ssize_t end = PyString_Size(program);
    char *program_str = PyString_AS_STRING(program);

    do {
        end -= 4;
        if (end < 0) return 'X';
        last_opcode = program_str[end];
    }
    while (last_opcode == OP_NOOP);

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
    Py_ssize_t end = PyString_Size(program);
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

    n_inputs = (int)PyString_Size(signature);
    n_temps = (int)PyString_Size(tempsig);

    if (o_constants) {
        if (!PySequence_Check(o_constants) ) {
                PyErr_SetString(PyExc_TypeError, "constants must be a sequence");
                return -1;
        }
        n_constants = (int)PySequence_Length(o_constants);
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
                itemsizes[i] = (int)PyString_GET_SIZE(o);
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
    char **errmsg;
    /* One memsteps array per thread */
    intp *memsteps[MAX_THREADS];
    /* One iterator per thread */
    NpyIter *iter[MAX_THREADS];
    /* When doing nested iteration for a reduction */
    NpyIter *reduce_iter[MAX_THREADS];
    /* Flag indicating reduction is the outer loop instead of the inner */
    int reduction_outer_loop;
} th_params;


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
        mem[r] = (char *)malloc(block_size * params.memsizes[r]);
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

/* Serial/parallel task iterator version of the VM engine */
static int
vm_engine_iter_task(NpyIter *iter, intp *memsteps, struct vm_params params, int *pc_error, char **errmsg)
{
    char **mem = params.mem;
    NpyIter_IterNextFunc *iternext;
    intp block_size, *size_ptr;
    char **iter_dataptr;
    intp *iter_strides;

    iternext = NpyIter_GetIterNext(iter, errmsg);
    if (iternext == NULL) {
        return -1;
    }

    size_ptr = NpyIter_GetInnerLoopSizePtr(iter);
    iter_dataptr = NpyIter_GetDataPtrArray(iter);
    iter_strides = NpyIter_GetInnerStrideArray(iter);

    /*
     * First do all the blocks with a compile-time fixed size.
     * This makes a big difference (30-50% on some tests).
     */
    block_size = *size_ptr;
    while (block_size == BLOCK_SIZE1) {
#define REDUCTION_INNER_LOOP
#define BLOCK_SIZE BLOCK_SIZE1
#include "interp_body.cpp"
#undef BLOCK_SIZE
#undef REDUCTION_INNER_LOOP
        iternext(iter);
        block_size = *size_ptr;
    }

    /* Then finish off the rest */
    if (block_size > 0) do {
#define REDUCTION_INNER_LOOP
#define BLOCK_SIZE block_size
#include "interp_body.cpp"
#undef BLOCK_SIZE
#undef REDUCTION_INNER_LOOP
    } while (iternext(iter));

    return 0;
}

static int
vm_engine_iter_outer_reduce_task(NpyIter *iter, intp *memsteps, struct vm_params params, int *pc_error, char **errmsg)
{
    char **mem = params.mem;
    NpyIter_IterNextFunc *iternext;
    intp block_size, *size_ptr;
    char **iter_dataptr;
    intp *iter_strides;

    iternext = NpyIter_GetIterNext(iter, errmsg);
    if (iternext == NULL) {
        return -1;
    }

    size_ptr = NpyIter_GetInnerLoopSizePtr(iter);
    iter_dataptr = NpyIter_GetDataPtrArray(iter);
    iter_strides = NpyIter_GetInnerStrideArray(iter);

    /*
     * First do all the blocks with a compile-time fixed size.
     * This makes a big difference (30-50% on some tests).
     */
    block_size = *size_ptr;
    while (block_size == BLOCK_SIZE1) {
#define BLOCK_SIZE BLOCK_SIZE1
#include "interp_body.cpp"
#undef BLOCK_SIZE
        iternext(iter);
        block_size = *size_ptr;
    }

    /* Then finish off the rest */
    if (block_size > 0) do {
#define BLOCK_SIZE block_size
#include "interp_body.cpp"
#undef BLOCK_SIZE
    } while (iternext(iter));

    return 0;
}

/* Parallel iterator version of VM engine */
static int
vm_engine_iter_parallel(NpyIter *iter, struct vm_params params, int *pc_error,
                        char **errmsg)
{
    int i;
    npy_intp numblocks, taskfactor;

    if (errmsg == NULL) {
        return -1;
    }

    /* Populate parameters for worker threads */
    NpyIter_GetIterIndexRange(iter, &th_params.start, &th_params.vlen);
    /*
     * Try to make it so each thread gets 16 tasks.  This is a compromise
     * between 1 task per thread and one block per task.
     */
    taskfactor = 16*BLOCK_SIZE1*nthreads;
    numblocks = (th_params.vlen - th_params.start + taskfactor - 1) /
                            taskfactor;
    th_params.block_size = numblocks * BLOCK_SIZE1;

    th_params.params = params;
    th_params.ret_code = 0;
    th_params.pc_error = pc_error;
    th_params.errmsg = errmsg;
    th_params.iter[0] = iter;
    /* Make one copy for each additional thread */
    for (i = 1; i < nthreads; ++i) {
        th_params.iter[i] = NpyIter_Copy(iter);
        if (th_params.iter[i] == NULL) {
            --i;
            for (; i > 0; --i) {
                NpyIter_Deallocate(th_params.iter[i]);
            }
            return -1;
        }
    }
    th_params.memsteps[0] = params.memsteps;
    /* Make one copy of memsteps for each additional thread */
    for (i = 1; i < nthreads; ++i) {
        th_params.memsteps[i] = PyMem_New(intp,
                    1 + params.n_inputs + params.n_constants + params.n_temps);
        if (th_params.memsteps[i] == NULL) {
            --i;
            for (; i > 0; --i) {
                PyMem_Del(th_params.memsteps[i]);
            }
            for (i = 0; i < nthreads; ++i) {
                NpyIter_Deallocate(th_params.iter[i]);
            }
            return -1;
        }
        memcpy(th_params.memsteps[i], th_params.memsteps[0],
                sizeof(intp) *
                (1 + params.n_inputs + params.n_constants + params.n_temps));
    }

    Py_BEGIN_ALLOW_THREADS;

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

    Py_END_ALLOW_THREADS;

    /* Deallocate all the iterator and memsteps copies */
    for (i = 1; i < nthreads; ++i) {
        NpyIter_Deallocate(th_params.iter[i]);
        PyMem_Del(th_params.memsteps[i]);
    }

    return th_params.ret_code;
}

/* Do the worker job for a certain thread */
void *th_worker(void *tidptr)
{
    int tid = *(int *)tidptr;
    /* Parameters for threads */
    intp start;
    intp vlen;
    intp block_size;
    NpyIter *iter;
    struct vm_params params;
    int *pc_error;
    int ret;
    int n_inputs;
    int n_constants;
    int n_temps;
    size_t memsize;
    char **mem;
    intp *memsteps;
    intp istart, iend;
    char **errmsg;

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
        mem = (char **)malloc(memsize);
        memcpy(mem, params.mem, memsize);

        errmsg = th_params.errmsg;

        params.mem = mem;

        /* Loop over blocks */
        pthread_mutex_lock(&count_mutex);
        if (!init_sentinels_done) {
            /* Set sentinels and other global variables */
            gindex = start;
            istart = gindex;
            iend = istart + block_size;
            if (iend > vlen) {
                iend = vlen;
            }
            init_sentinels_done = 1;  /* sentinels have been initialised */
            giveup = 0;            /* no giveup initially */
        } else {
            gindex += block_size;
            istart = gindex;
            iend = istart + block_size;
            if (iend > vlen) {
                iend = vlen;
            }
        }
        /* Grab one of the iterators */
        iter = th_params.iter[tid];
        if (iter == NULL) {
            th_params.ret_code = -1;
            giveup = 1;
        }
        memsteps = th_params.memsteps[tid];
        /* Get temporary space for each thread */
        ret = get_temps_space(params, mem, BLOCK_SIZE1);
        if (ret < 0) {
            /* Propagate error to main thread */
            th_params.ret_code = ret;
            giveup = 1;
        }
        pthread_mutex_unlock(&count_mutex);

        while (istart < vlen && !giveup) {
            /* Reset the iterator to the range for this task */
            ret = NpyIter_ResetToIterIndexRange(iter, istart, iend,
                                                errmsg);
            /* Execute the task */
            if (ret >= 0) {
                ret = vm_engine_iter_task(iter, memsteps, params, pc_error, errmsg);
            }

            if (ret < 0) {
                pthread_mutex_lock(&count_mutex);
                giveup = 1;
                /* Propagate error to main thread */
                th_params.ret_code = ret;
                pthread_mutex_unlock(&count_mutex);
                break;
            }

            pthread_mutex_lock(&count_mutex);
            gindex += block_size;
            istart = gindex;
            iend = istart + block_size;
            if (iend > vlen) {
                iend = vlen;
            }
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

static int
run_interpreter(NumExprObject *self, NpyIter *iter, NpyIter *reduce_iter,
                     int reduction_outer_loop, int *pc_error)
{
    int r;
    Py_ssize_t plen;
    struct vm_params params;
    char *errmsg = NULL;

    *pc_error = -1;
    if (PyString_AsStringAndSize(self->program, (char **)&(params.program),
                                 &plen) < 0) {
        return -1;
    }

    params.prog_len = (int)plen;
    params.output = NULL;
    params.inputs = NULL;
    params.index_data = NULL;
    params.n_inputs = self->n_inputs;
    params.n_constants = self->n_constants;
    params.n_temps = self->n_temps;
    params.mem = self->mem;
    params.memsteps = self->memsteps;
    params.memsizes = self->memsizes;
    params.r_end = (int)PyString_Size(self->fullsig);

    if ((nthreads == 1) || force_serial) {
        /* Can do it as one "task" */
        if (reduce_iter == NULL) {
            /* Reset the iterator to allocate its buffers */
            if(NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
                return -1;
            }
            get_temps_space(params, params.mem, BLOCK_SIZE1);
            Py_BEGIN_ALLOW_THREADS;
            r = vm_engine_iter_task(iter, params.memsteps,
                                        params, pc_error, &errmsg);
            Py_END_ALLOW_THREADS;
            free_temps_space(params, params.mem);
        }
        else {
            if (reduction_outer_loop) {
                char **dataptr;
                NpyIter_IterNextFunc *iternext;

                dataptr = NpyIter_GetDataPtrArray(reduce_iter);
                iternext = NpyIter_GetIterNext(reduce_iter, NULL);
                if (iternext == NULL) {
                    return -1;
                }

                get_temps_space(params, params.mem, BLOCK_SIZE1);
                Py_BEGIN_ALLOW_THREADS;
                do {
                    r = NpyIter_ResetBasePointers(iter, dataptr, &errmsg);
                    if (r >= 0) {
                        r = vm_engine_iter_outer_reduce_task(iter,
                                                params.memsteps, params,
                                                pc_error, &errmsg);
                    }
                    if (r < 0) {
                        break;
                    }
                } while (iternext(reduce_iter));
                Py_END_ALLOW_THREADS;
                free_temps_space(params, params.mem);
            }
            else {
                char **dataptr;
                NpyIter_IterNextFunc *iternext;

                dataptr = NpyIter_GetDataPtrArray(iter);
                iternext = NpyIter_GetIterNext(iter, NULL);
                if (iternext == NULL) {
                    return -1;
                }

                get_temps_space(params, params.mem, BLOCK_SIZE1);
                Py_BEGIN_ALLOW_THREADS;
                do {
                    r = NpyIter_ResetBasePointers(reduce_iter, dataptr,
                                                                    &errmsg);
                    if (r >= 0) {
                        r = vm_engine_iter_task(reduce_iter, params.memsteps,
                                                params, pc_error, &errmsg);
                    }
                    if (r < 0) {
                        break;
                    }
                } while (iternext(iter));
                Py_END_ALLOW_THREADS;
                free_temps_space(params, params.mem);
            }
        }
    }
    else {
        if (reduce_iter == NULL) {
            r = vm_engine_iter_parallel(iter, params, pc_error, &errmsg);
        }
        else {
            errmsg = "Parallel engine doesn't support reduction yet";
            r = -1;
        }
    }

    if (r < 0 && errmsg != NULL) {
        PyErr_SetString(PyExc_RuntimeError, errmsg);
    }

    return 0;
}

static int
run_interpreter_const(NumExprObject *self, char *output, int *pc_error)
{
    struct vm_params params;
    Py_ssize_t plen;
    char **mem;
    intp *memsteps;

    *pc_error = -1;
    if (PyString_AsStringAndSize(self->program, (char **)&(params.program),
                                 &plen) < 0) {
        return -1;
    }
    if (self->n_inputs != 0) {
        return -1;
    }
    params.prog_len = (int)plen;
    params.output = output;
    params.inputs = NULL;
    params.index_data = NULL;
    params.n_inputs = self->n_inputs;
    params.n_constants = self->n_constants;
    params.n_temps = self->n_temps;
    params.mem = self->mem;
    memsteps = self->memsteps;
    params.memsizes = self->memsizes;
    params.r_end = (int)PyString_Size(self->fullsig);

    mem = params.mem;
    get_temps_space(params, mem, 1);
#define SINGLE_ITEM_CONST_LOOP
#define BLOCK_SIZE 1
#include "interp_body.cpp"
#undef BLOCK_SIZE
#undef SINGLE_ITEM_CONST_LOOP
    free_temps_space(params, mem);

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

static PyObject *
NumExpr_run(NumExprObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *operands[NPY_MAXARGS];
    PyArray_Descr *dtypes[NPY_MAXARGS], **dtypes_tmp;
    PyObject *tmp, *ret;
    npy_uint32 op_flags[NPY_MAXARGS];
    NPY_CASTING casting = NPY_SAFE_CASTING;
    NPY_ORDER order = NPY_KEEPORDER;
    unsigned int i, n_inputs;
    int r, pc_error = 0;
    int reduction_axis = -1;
    npy_intp reduction_size = 1;
    int ex_uses_vml = 0, is_reduction = 0, reduction_outer_loop = 0;

    /* To specify axes when doing a reduction */
    int op_axes_values[NPY_MAXARGS][NPY_MAXDIMS],
         op_axes_reduction_values[NPY_MAXARGS];
    int *op_axes_ptrs[NPY_MAXDIMS];
    int oa_ndim = 0;
    int **op_axes = NULL;

    NpyIter *iter = NULL, *reduce_iter = NULL;

    /* Check whether we need to restart threads */
    if (!init_threads_done || pid != getpid()) {
        numexpr_set_nthreads(nthreads);
    }

    /* Don't force serial mode by default */
    force_serial = 0;

    /* Check whether there's a reduction as the final step */
    is_reduction = last_opcode(self->program) > OP_REDUCTION;

    n_inputs = (int)PyTuple_Size(args);
    if (PyString_Size(self->signature) != n_inputs) {
        return PyErr_Format(PyExc_ValueError,
                            "number of inputs doesn't match program");
    }
    else if (n_inputs+1 > NPY_MAXARGS) {
        return PyErr_Format(PyExc_ValueError,
                            "too many inputs");
    }

    memset(operands, 0, sizeof(operands));
    memset(dtypes, 0, sizeof(dtypes));

    if (kwds) {
        tmp = PyDict_GetItemString(kwds, "casting"); /* borrowed ref */
        if (tmp != NULL && !PyArray_CastingConverter(tmp, &casting)) {
            return NULL;
        }
        tmp = PyDict_GetItemString(kwds, "order"); /* borrowed ref */
        if (tmp != NULL && !PyArray_OrderConverter(tmp, &order)) {
            return NULL;
        }
        tmp = PyDict_GetItemString(kwds, "ex_uses_vml"); /* borrowed ref */
        if (tmp == NULL) {
            return PyErr_Format(PyExc_ValueError,
                                "ex_uses_vml parameter is required");
        }
        if (tmp == Py_True) {
            ex_uses_vml = 1;
        }
            /* borrowed ref */
        operands[0] = (PyArrayObject *)PyDict_GetItemString(kwds, "out");
        if (operands[0] != NULL) {
            if ((PyObject *)operands[0] == Py_None) {
                operands[0] = NULL;
            }
            else if (!PyArray_Check(operands[0])) {
                return PyErr_Format(PyExc_ValueError,
                                    "out keyword parameter is not an array");
            }
            else {
                Py_INCREF(operands[0]);
            }
        }
    }

    for (i = 0; i < n_inputs; i++) {
        PyObject *o = PyTuple_GET_ITEM(args, i); /* borrowed ref */
        PyObject *a;
        char c = PyString_AS_STRING(self->signature)[i];
        int typecode = typecode_from_char(c);
        /* Convert it if it's not an array */
        if (!PyArray_Check(o)) {
            if (typecode == -1) goto fail;
            a = PyArray_FROM_OTF(o, typecode, NOTSWAPPED);
        }
        else {
            Py_INCREF(o);
            a = o;
        }
        operands[i+1] = (PyArrayObject *)a;
        dtypes[i+1] = PyArray_DescrFromType(typecode);

        if (operands[i+1] == NULL || dtypes[i+1] == NULL) {
            goto fail;
        }
        op_flags[i+1] = NPY_ITER_READONLY|
#ifdef USE_VML
                        (ex_uses_vml ? (NPY_ITER_CONTIG|NPY_ITER_ALIGNED) : 0)|
#endif
#ifndef USE_UNALIGNED_ACCESS
                        NPY_ITER_ALIGNED|
#endif
                        NPY_ITER_NBO
                        ;
    }

    if (is_reduction) {
        /* A reduction can not result in a string,
           so we don't need to worry about item sizes here. */
        char retsig = get_return_sig(self->program);
        reduction_axis = get_reduction_axis(self->program);

        /* Need to set up op_axes for the non-reduction part */
        if (reduction_axis != 255) {
            /* Get the number of broadcast dimensions */
            for (i = 0; i < n_inputs; ++i) {
                int ndim = PyArray_NDIM(operands[i+1]);
                if (ndim > oa_ndim) {
                    oa_ndim = ndim;
                }
            }
            if (reduction_axis < 0 || reduction_axis >= oa_ndim) {
                PyErr_Format(PyExc_ValueError,
                        "reduction axis is out of bounds");
                goto fail;
            }
            /* Fill in the op_axes */
            op_axes_ptrs[0] = NULL;
            op_axes_reduction_values[0] = -1;
            for (i = 0; i < n_inputs; ++i) {
                int j = 0, idim, ndim = PyArray_NDIM(operands[i+1]);
                for (idim = 0; idim < oa_ndim-ndim; ++idim) {
                    if (idim != reduction_axis) {
                        op_axes_values[i+1][j++] = -1;
                    }
                    else {
                        op_axes_reduction_values[i+1] = -1;
                    }
                }
                for (idim = oa_ndim-ndim; idim < oa_ndim; ++idim) {
                    if (idim != reduction_axis) {
                        op_axes_values[i+1][j++] = idim-(oa_ndim-ndim);
                    }
                    else {
                        intp size = PyArray_DIM(operands[i+1],
                                                    idim-(oa_ndim-ndim));
                        if (size > reduction_size) {
                            reduction_size = size;
                        }
                        op_axes_reduction_values[i+1] = idim-(oa_ndim-ndim);
                    }
                }
                op_axes_ptrs[i+1] = op_axes_values[i+1];
            }
            /* op_axes has one less than the broadcast dimensions */
            --oa_ndim;
            if (oa_ndim > 0) {
                op_axes = op_axes_ptrs;
            }
            else {
                reduction_size = 1;
            }
        }
        /* A full reduction can be done without nested iteration */
        if (oa_ndim == 0) {
            if (operands[0] == NULL) {
                intp dim = 1;
                operands[0] = (PyArrayObject *)PyArray_SimpleNew(0, &dim,
                                            typecode_from_char(retsig));
                if (!operands[0])
                    goto fail;
            } else if (PyArray_SIZE(operands[0]) != 1) {
                PyErr_Format(PyExc_ValueError,
                        "out argument must have size 1 for a full reduction");
                goto fail;
            }
        }

        dtypes[0] = PyArray_DescrFromType(typecode_from_char(retsig));

        op_flags[0] = NPY_ITER_READWRITE|
                      NPY_ITER_ALLOCATE|
                      /* Copy, because it can't buffer the reduction */
                      NPY_ITER_UPDATEIFCOPY|
                      NPY_ITER_NBO|
#ifndef USE_UNALIGNED_ACCESS
                      NPY_ITER_ALIGNED|
#endif
                      (oa_ndim == 0 ? 0 : NPY_ITER_NO_BROADCAST);
    }
    else {
        char retsig = get_return_sig(self->program);
        if (retsig != 's') {
            dtypes[0] = PyArray_DescrFromType(typecode_from_char(retsig));
        } else {
            /* Since the *only* supported operation returning a string
             * is a copy, the size of returned strings
             * can be directly gotten from the first (and only)
             * input/constant/temporary. */
            if (n_inputs > 0) {  /* input, like in 'a' where a -> 'foo' */
                dtypes[0] = PyArray_DESCR(operands[1]);
                Py_INCREF(dtypes[0]);
            } else {  /* constant, like in '"foo"' */
                dtypes[0] = PyArray_DescrNewFromType(PyArray_STRING);
                dtypes[0]->elsize = (int)self->memsizes[1];
            }  /* no string temporaries, so no third case  */
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

    /* Check for empty arrays in expression */
    if (n_inputs > 0) {
        char retsig = get_return_sig(self->program);

        /* Check length for all inputs */
        int zeroi, zerolen = 0;
        for (i=0; i < n_inputs; i++) {
            if (PyArray_SIZE(operands[i+1]) == 0) {
                zerolen = 1;
                zeroi = i+1;
                break;
            }
        }

        if (zerolen != 0) {
            /* Allocate the output */
            int ndim = PyArray_NDIM(operands[zeroi]);
            intp *dims = PyArray_DIMS(operands[zeroi]);
            operands[0] = (PyArrayObject *)PyArray_SimpleNew(ndim, dims,
                                              typecode_from_char(retsig));
            if (operands[0] == NULL) {
                goto fail;
            }

            ret = (PyObject *)operands[0];
            Py_INCREF(ret);
            goto cleanup_and_exit;
        }
    }


    /* A case with a single constant output */
    if (n_inputs == 0) {
        char retsig = get_return_sig(self->program);

        /* Allocate the output */
        if (operands[0] == NULL) {
            intp dim = 1;
            operands[0] = (PyArrayObject *)PyArray_SimpleNew(0, &dim,
                                        typecode_from_char(retsig));
            if (operands[0] == NULL) {
                goto fail;
            }
        }
        else {
            PyArrayObject *a;
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
            a = (PyArrayObject *)PyArray_FromArray(operands[0], dtypes[0],
                                        NPY_ALIGNED|NPY_UPDATEIFCOPY);
            if (a == NULL) {
                goto fail;
            }
            Py_DECREF(operands[0]);
            operands[0] = a;
        }

        r = run_interpreter_const(self, PyArray_BYTES(operands[0]), &pc_error);

        ret = (PyObject *)operands[0];
        Py_INCREF(ret);
        goto cleanup_and_exit;
    }


    /* Allocate the iterator or nested iterators */
    if (reduction_size == 1) {
        /* When there's no reduction, reduction_size is 1 as well */
        iter = NpyIter_AdvancedNew(n_inputs+1, operands,
                            NPY_ITER_BUFFERED|
                            NPY_ITER_REDUCE_OK|
                            NPY_ITER_RANGED|
                            NPY_ITER_DELAY_BUFALLOC|
                            NPY_ITER_EXTERNAL_LOOP,
                            order, casting,
                            op_flags, dtypes,
                            0, NULL, NULL,
                            BLOCK_SIZE1);
        if (iter == NULL) {
            goto fail;
        }
    } else {
        npy_uint32 op_flags_outer[NPY_MAXDIMS];
        /* The outer loop is unbuffered */
        op_flags_outer[0] = NPY_ITER_READWRITE|
                            NPY_ITER_ALLOCATE|
                            NPY_ITER_NO_BROADCAST;
        for (i = 0; i < n_inputs; ++i) {
            op_flags_outer[i+1] = NPY_ITER_READONLY;
        }
        /* Arbitrary threshold for which is the inner loop...benchmark? */
        if (reduction_size < 64) {
            reduction_outer_loop = 1;
            iter = NpyIter_AdvancedNew(n_inputs+1, operands,
                                NPY_ITER_BUFFERED|
                                NPY_ITER_RANGED|
                                NPY_ITER_DELAY_BUFALLOC|
                                NPY_ITER_EXTERNAL_LOOP,
                                order, casting,
                                op_flags, dtypes,
                                oa_ndim, op_axes, NULL,
                                BLOCK_SIZE1);
            if (iter == NULL) {
                goto fail;
            }

            /* If the output was allocated, get it for the second iterator */
            if (operands[0] == NULL) {
                operands[0] = NpyIter_GetOperandArray(iter)[0];
                Py_INCREF(operands[0]);
            }

            op_axes[0] = &op_axes_reduction_values[0];
            for (i = 0; i < n_inputs; ++i) {
                op_axes[i+1] = &op_axes_reduction_values[i+1];
            }
            op_flags_outer[0] &= ~NPY_ITER_NO_BROADCAST;
            reduce_iter = NpyIter_AdvancedNew(n_inputs+1, operands,
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

            /* If the output is being allocated, need to specify its dtype */
            dtypes_outer[0] = dtypes[0];
            for (i = 0; i < n_inputs; ++i) {
                dtypes_outer[i+1] = NULL;
            }
            iter = NpyIter_AdvancedNew(n_inputs+1, operands,
                                NPY_ITER_RANGED,
                                order, casting,
                                op_flags_outer, dtypes_outer,
                                oa_ndim, op_axes, NULL,
                                0);
            if (iter == NULL) {
                goto fail;
            }

            /* If the output was allocated, get it for the second iterator */
            if (operands[0] == NULL) {
                operands[0] = NpyIter_GetOperandArray(iter)[0];
                Py_INCREF(operands[0]);
            }

            op_axes[0] = &op_axes_reduction_values[0];
            for (i = 0; i < n_inputs; ++i) {
                op_axes[i+1] = &op_axes_reduction_values[i+1];
            }
            op_flags[0] &= ~NPY_ITER_NO_BROADCAST;
            reduce_iter = NpyIter_AdvancedNew(n_inputs+1, operands,
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

    /* Initialize the output to the reduction unit */
    if (is_reduction) {
        PyArrayObject *a = NpyIter_GetOperandArray(iter)[0];
        if (last_opcode(self->program) >= OP_SUM &&
            last_opcode(self->program) < OP_PROD) {
                PyObject *zero = PyInt_FromLong(0);
                PyArray_FillWithScalar(a, zero);
                Py_DECREF(zero);
        } else {
                PyObject *one = PyInt_FromLong(1);
                PyArray_FillWithScalar(a, one);
                Py_DECREF(one);
        }
    }

    /* Get the sizes of all the operands */
    dtypes_tmp = NpyIter_GetDescrArray(iter);
    for (i = 0; i < n_inputs+1; ++i) {
        self->memsizes[i] = dtypes_tmp[i]->elsize;
    }

    /* For small calculations, just use 1 thread */
    if (NpyIter_GetIterSize(iter) < 2*BLOCK_SIZE1) {
        force_serial = 1;
    }

    /* Reductions do not support parallel execution yet */
    if (is_reduction) {
        force_serial = 1;
    }

    r = run_interpreter(self, iter, reduce_iter,
                             reduction_outer_loop, &pc_error);

    if (r < 0) {
        if (r == -1) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError,
                                "an error occurred while running the program");
            }
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
        goto fail;
    }

    /* Get the output from the iterator */
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[0];
    Py_INCREF(ret);

    NpyIter_Deallocate(iter);
    if (reduce_iter != NULL) {
        NpyIter_Deallocate(reduce_iter);
    }
cleanup_and_exit:
    for (i = 0; i < n_inputs+1; i++) {
        Py_XDECREF(operands[i]);
        Py_XDECREF(dtypes[i]);
    }

    return ret;
fail:
    for (i = 0; i < n_inputs+1; i++) {
        Py_XDECREF(operands[i]);
        Py_XDECREF(dtypes[i]);
    }
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }
    if (reduce_iter != NULL) {
        NpyIter_Deallocate(reduce_iter);
    }

    return NULL;
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

PyMODINIT_FUNC
initinterpreter()
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
#include "opcodes.hpp"
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
#include "functions.hpp"
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
