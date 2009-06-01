#include "Python.h"
#include "structmember.h"
#include "numpy/noprefix.h"
#include "numpy/arrayscalars.h"
#include "math.h"
#include "string.h"
#include "assert.h"

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

#ifdef USE_VML
/* The values below have been tuned for a nowadays Core2 processor */
/* Note: with VML functions a larger block size (e.g. 4096) allows to make use
 * of the automatic multithreading capabilities of the VML library */
#define BLOCK_SIZE1 4096
#define BLOCK_SIZE2 32
#else
/* The values below have been tuned for a nowadays Core2 processor */
/* Note: without VML available a smaller block size is best, specially
 * for the strided and unaligned cases.  However, this may change
 * when/if numexpr would support multithreading for the non-VML case. */
#define BLOCK_SIZE1 256
#define BLOCK_SIZE2 8
#endif

/* This file and interp_body should really be generated from a description of
   the opcodes -- there's too much repetition here for manually editing */


enum OpCodes {
    OP_NOOP = 0,

    OP_COPY_BB,

    OP_INVERT_BB,
    OP_AND_BBB,
    OP_OR_BBB,

    OP_EQ_BBB,
    OP_NE_BBB,

    OP_GT_BII,
    OP_GE_BII,
    OP_EQ_BII,
    OP_NE_BII,

    OP_GT_BLL,
    OP_GE_BLL,
    OP_EQ_BLL,
    OP_NE_BLL,

    OP_GT_BFF,
    OP_GE_BFF,
    OP_EQ_BFF,
    OP_NE_BFF,

    OP_GT_BDD,
    OP_GE_BDD,
    OP_EQ_BDD,
    OP_NE_BDD,

    OP_GT_BSS,
    OP_GE_BSS,
    OP_EQ_BSS,
    OP_NE_BSS,

    OP_COPY_II,
    OP_ONES_LIKE_II,
    OP_NEG_II,
    OP_ADD_III,
    OP_SUB_III,
    OP_MUL_III,
    OP_DIV_III,
    OP_POW_III,
    OP_MOD_III,
    OP_WHERE_IBII,

    OP_CAST_LI,
    OP_COPY_LL,
    OP_ONES_LIKE_LL,
    OP_NEG_LL,
    OP_ADD_LLL,
    OP_SUB_LLL,
    OP_MUL_LLL,
    OP_DIV_LLL,
    OP_POW_LLL,
    OP_MOD_LLL,
    OP_WHERE_LBLL,

    OP_CAST_FI,
    OP_CAST_FL,
    OP_COPY_FF,
    OP_ONES_LIKE_FF,
    OP_NEG_FF,
    OP_ADD_FFF,
    OP_SUB_FFF,
    OP_MUL_FFF,
    OP_DIV_FFF,
    OP_POW_FFF,
    OP_MOD_FFF,
    OP_SQRT_FF,
    OP_WHERE_FBFF,
    OP_FUNC_FF,
    OP_FUNC_FFF,

    OP_CAST_DI,
    OP_CAST_DL,
    OP_CAST_DF,
    OP_COPY_DD,
    OP_ONES_LIKE_DD,
    OP_NEG_DD,
    OP_ADD_DDD,
    OP_SUB_DDD,
    OP_MUL_DDD,
    OP_DIV_DDD,
    OP_POW_DDD,
    OP_MOD_DDD,
    OP_SQRT_DD,
    OP_WHERE_DBDD,
    OP_FUNC_DD,
    OP_FUNC_DDD,

    OP_EQ_BCC,
    OP_NE_BCC,

    OP_CAST_CI,
    OP_CAST_CL,
    OP_CAST_CF,
    OP_CAST_CD,
    OP_ONES_LIKE_CC,
    OP_COPY_CC,
    OP_NEG_CC,
    OP_ADD_CCC,
    OP_SUB_CCC,
    OP_MUL_CCC,
    OP_DIV_CCC,
    OP_WHERE_CBCC,
    OP_FUNC_CC,
    OP_FUNC_CCC,

    OP_REAL_DC,
    OP_IMAG_DC,
    OP_COMPLEX_CDD,

    OP_COPY_SS,

    OP_REDUCTION,

    OP_SUM,
    OP_SUM_IIN,
    OP_SUM_LLN,
    OP_SUM_FFN,
    OP_SUM_DDN,
    OP_SUM_CCN,

    OP_PROD,
    OP_PROD_IIN,
    OP_PROD_LLN,
    OP_PROD_FFN,
    OP_PROD_DDN,
    OP_PROD_CCN

};

/* returns the sig of the nth op, '\0' if no more ops -1 on failure */
static int
op_signature(int op, int n) {
    switch (op) {
        case OP_NOOP:
            break;
        case OP_COPY_BB:
            if (n == 0 || n == 1) return 'b';
            break;
        case OP_INVERT_BB:
            if (n == 0 || n == 1) return 'b';
            break;
        case OP_AND_BBB:
        case OP_OR_BBB:
        case OP_EQ_BBB:
        case OP_NE_BBB:
            if (n == 0 || n == 1 || n == 2) return 'b';
            break;
        case OP_GT_BII:
        case OP_GE_BII:
        case OP_EQ_BII:
        case OP_NE_BII:
            if (n == 0) return 'b';
            if (n == 1 || n == 2) return 'i';
            break;
        case OP_GT_BLL:
        case OP_GE_BLL:
        case OP_EQ_BLL:
        case OP_NE_BLL:
            if (n == 0) return 'b';
            if (n == 1 || n == 2) return 'l';
            break;
        case OP_GT_BFF:
        case OP_GE_BFF:
        case OP_EQ_BFF:
        case OP_NE_BFF:
            if (n == 0) return 'b';
            if (n == 1 || n == 2) return 'f';
            break;
        case OP_GT_BDD:
        case OP_GE_BDD:
        case OP_EQ_BDD:
        case OP_NE_BDD:
            if (n == 0) return 'b';
            if (n == 1 || n == 2) return 'd';
            break;
        case OP_EQ_BCC:
        case OP_NE_BCC:
            if (n == 0) return 'b';
            if (n == 1 || n == 2) return 'c';
            break;

        case OP_GT_BSS:
        case OP_GE_BSS:
        case OP_EQ_BSS:
        case OP_NE_BSS:
            if (n == 0) return 'b';
            if (n == 1 || n == 2) return 's';
            break;
        case OP_COPY_II:
        case OP_ONES_LIKE_II:
        case OP_NEG_II:
            if (n == 0 || n == 1) return 'i';
            break;
        case OP_ADD_III:
        case OP_SUB_III:
        case OP_MUL_III:
        case OP_DIV_III:
        case OP_MOD_III:
        case OP_POW_III:
            if (n == 0 || n == 1 || n == 2) return 'i';
            break;
        case OP_WHERE_IBII:
            if (n == 0 || n == 2 || n == 3) return 'i';
            if (n == 1) return 'b';
            break;
        case OP_CAST_LI:
            if (n == 0) return 'l';
            if (n == 1) return 'i';
            break;
        case OP_COPY_LL:
        case OP_ONES_LIKE_LL:
        case OP_NEG_LL:
            if (n == 0 || n == 1) return 'l';
            break;
        case OP_ADD_LLL:
        case OP_SUB_LLL:
        case OP_MUL_LLL:
        case OP_DIV_LLL:
        case OP_MOD_LLL:
        case OP_POW_LLL:
            if (n == 0 || n == 1 || n == 2) return 'l';
            break;
        case OP_WHERE_LBLL:
            if (n == 0 || n == 2 || n == 3) return 'l';
            if (n == 1) return 'b';
            break;

        case OP_CAST_FI:
            if (n == 0) return 'f';
            if (n == 1) return 'i';
            break;
        case OP_CAST_FL:
            if (n == 0) return 'f';
            if (n == 1) return 'l';
            break;
        case OP_COPY_FF:
        case OP_ONES_LIKE_FF:
        case OP_NEG_FF:
        case OP_SQRT_FF:
            if (n == 0 || n == 1) return 'f';
            break;
        case OP_ADD_FFF:
        case OP_SUB_FFF:
        case OP_MUL_FFF:
        case OP_DIV_FFF:
        case OP_POW_FFF:
        case OP_MOD_FFF:
            if (n == 0 || n == 1 || n == 2) return 'f';
            break;
        case OP_WHERE_FBFF:
            if (n == 0 || n == 2 || n == 3) return 'f';
            if (n == 1) return 'b';
            break;
        case OP_FUNC_FF:
            if (n == 0 || n == 1) return 'f';
            if (n == 2) return 'n';
            break;
        case OP_FUNC_FFF:
            if (n == 0 || n == 1 || n == 2) return 'f';
            if (n == 3) return 'n';
            break;

        case OP_CAST_DI:
            if (n == 0) return 'd';
            if (n == 1) return 'i';
            break;
        case OP_CAST_DL:
            if (n == 0) return 'd';
            if (n == 1) return 'l';
            break;
        case OP_CAST_DF:
            if (n == 0) return 'd';
            if (n == 1) return 'f';
            break;
        case OP_COPY_DD:
        case OP_ONES_LIKE_DD:
        case OP_NEG_DD:
        case OP_SQRT_DD:
            if (n == 0 || n == 1) return 'd';
            break;
        case OP_ADD_DDD:
        case OP_SUB_DDD:
        case OP_MUL_DDD:
        case OP_DIV_DDD:
        case OP_POW_DDD:
        case OP_MOD_DDD:
            if (n == 0 || n == 1 || n == 2) return 'd';
            break;
        case OP_WHERE_DBDD:
            if (n == 0 || n == 2 || n == 3) return 'd';
            if (n == 1) return 'b';
            break;
        case OP_FUNC_DD:
            if (n == 0 || n == 1) return 'd';
            if (n == 2) return 'n';
            break;
        case OP_FUNC_DDD:
            if (n == 0 || n == 1 || n == 2) return 'd';
            if (n == 3) return 'n';
            break;

        case OP_CAST_CI:
            if (n == 0) return 'c';
            if (n == 1) return 'i';
            break;
        case OP_CAST_CL:
            if (n == 0) return 'c';
            if (n == 1) return 'l';
            break;
        case OP_CAST_CF:
            if (n == 0) return 'c';
            if (n == 1) return 'f';
            break;
        case OP_CAST_CD:
            if (n == 0) return 'c';
            if (n == 1) return 'd';
            break;
        case OP_COPY_CC:
        case OP_ONES_LIKE_CC:
        case OP_NEG_CC:
            if (n == 0 || n == 1) return 'c';
            break;
        case OP_ADD_CCC:
        case OP_SUB_CCC:
        case OP_MUL_CCC:
        case OP_DIV_CCC:
            if (n == 0 || n == 1 || n == 2) return 'c';
            break;
        case OP_WHERE_CBCC:
            if (n == 0 || n == 2 || n == 3) return 'c';
            if (n == 1) return 'b';
            break;
        case OP_FUNC_CC:
            if (n == 0 || n == 1) return 'c';
            if (n == 2) return 'n';
            break;
        case OP_FUNC_CCC:
            if (n == 0 || n == 1 || n == 2) return 'c';
            if (n == 3) return 'n';
            break;
        case OP_REAL_DC:
        case OP_IMAG_DC:
            if (n == 0) return 'd';
            if (n == 1) return 'c';
            break;
        case OP_COMPLEX_CDD:
            if (n == 0) return 'c';
            if (n == 1 || n == 2) return 'd';
            break;
        case OP_COPY_SS:
            if (n == 0 || n == 1) return 's';
            break;
        case OP_PROD_IIN:
        case OP_SUM_IIN:
            if (n == 0 || n == 1) return 'i';
            if (n == 2) return 'n';
            break;
        case OP_PROD_LLN:
        case OP_SUM_LLN:
            if (n == 0 || n == 1) return 'l';
            if (n == 2) return 'n';
            break;
        case OP_PROD_FFN:
        case OP_SUM_FFN:
            if (n == 0 || n == 1) return 'f';
            if (n == 2) return 'n';
            break;
        case OP_PROD_DDN:
        case OP_SUM_DDN:
            if (n == 0 || n == 1) return 'd';
            if (n == 2) return 'n';
            break;
        case OP_PROD_CCN:
        case OP_SUM_CCN:
            if (n == 0 || n == 1) return 'c';
            if (n == 2) return 'n';
            break;
        default:
            return -1;
            break;
    }
    return 0;
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
    FUNC_SQRT_FF = 0,
    FUNC_SIN_FF,
    FUNC_COS_FF,
    FUNC_TAN_FF,
    FUNC_ARCSIN_FF,
    FUNC_ARCCOS_FF,
    FUNC_ARCTAN_FF,
    FUNC_SINH_FF,
    FUNC_COSH_FF,
    FUNC_TANH_FF,
    FUNC_ARCSINH_FF,
    FUNC_ARCCOSH_FF,
    FUNC_ARCTANH_FF,

    FUNC_LOG_FF,
    FUNC_LOG1P_FF,
    FUNC_LOG10_FF,
    FUNC_EXP_FF,
    FUNC_EXPM1_FF,

    FUNC_FF_LAST
};

typedef float (*FuncFFPtr)(float);

/* The order of this array must match the FuncFFCodes enum above */
#ifdef _WIN32
FuncFFPtr functions_ff[] = {
    sqrtf2,
    sinf2,
    cosf2,
    tanf2,
    asinf2,
    acosf2,
    atanf2,
    sinhf2,
    coshf2,
    tanhf2,
    asinhf2,
    acoshf2,
    atanhf2,
    logf2,
    log1pf2,
    log10f2,
    expf2,
    expm1f2,
};
#else
FuncFFPtr functions_ff[] = {
    sqrtf,
    sinf,
    cosf,
    tanf,
    asinf,
    acosf,
    atanf,
    sinhf,
    coshf,
    tanhf,
    asinhf,
    acoshf,
    atanhf,
    logf,
    log1pf,
    log10f,
    expf,
    expm1f,
};
#endif  // #ifdef _WIN32

#ifdef USE_VML
typedef void (*FuncFFPtr_vml)(int, const float*, float*);
FuncFFPtr_vml functions_ff_vml[] = {
    vsSqrt,
    vsSin,
    vsCos,
    vsTan,
    vsAsin,
    vsAcos,
    vsAtan,
    vsSinh,
    vsCosh,
    vsTanh,
    vsAsinh,
    vsAcosh,
    vsAtanh,
    vsLn,
    vsLog1p,
    vsLog10,
    vsExp,
    vsExpm1,
};
#endif

enum FuncFFFCodes {
    FUNC_FMOD_FFF = 0,
    FUNC_ARCTAN2_FFF,

    FUNC_FFF_LAST
};

typedef float (*FuncFFFPtr)(float, float);

#ifdef _WIN32
FuncFFFPtr functions_fff[] = {
    fmodf2,
    atan2f2,
};
#else
FuncFFFPtr functions_fff[] = {
    fmodf,
    atan2f,
};
#endif  // #ifdef _WIN32

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
    vsfmod,
    vsAtan2,
};
#endif


enum FuncDDCodes {
    FUNC_SQRT_DD = 0,
    FUNC_SIN_DD,
    FUNC_COS_DD,
    FUNC_TAN_DD,
    FUNC_ARCSIN_DD,
    FUNC_ARCCOS_DD,
    FUNC_ARCTAN_DD,
    FUNC_SINH_DD,
    FUNC_COSH_DD,
    FUNC_TANH_DD,
    FUNC_ARCSINH_DD,
    FUNC_ARCCOSH_DD,
    FUNC_ARCTANH_DD,

    FUNC_LOG_DD,
    FUNC_LOG1P_DD,
    FUNC_LOG10_DD,
    FUNC_EXP_DD,
    FUNC_EXPM1_DD,

    FUNC_DD_LAST
};

typedef double (*FuncDDPtr)(double);

/* The order of this array must match the FuncDDCodes enum above */
FuncDDPtr functions_dd[] = {
    sqrt,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    log,
    log1p,
    log10,
    exp,
    expm1,
};

#ifdef USE_VML
typedef void (*FuncDDPtr_vml)(int, const double*, double*);
FuncDDPtr_vml functions_dd_vml[] = {
    vdSqrt,
    vdSin,
    vdCos,
    vdTan,
    vdAsin,
    vdAcos,
    vdAtan,
    vdSinh,
    vdCosh,
    vdTanh,
    vdAsinh,
    vdAcosh,
    vdAtanh,
    vdLn,
    vdLog1p,
    vdLog10,
    vdExp,
    vdExpm1,
};
#endif

enum FuncDDDCodes {
    FUNC_FMOD_DDD = 0,
    FUNC_ARCTAN2_DDD,

    FUNC_DDD_LAST
};

typedef double (*FuncDDDPtr)(double, double);

FuncDDDPtr functions_ddd[] = {
    fmod,
    atan2,
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
    vdfmod,
    vdAtan2,
};
#endif


enum FuncCCCodes {
    FUNC_SQRT_CC = 0,
    FUNC_SIN_CC,
    FUNC_COS_CC,
    FUNC_TAN_CC,
    FUNC_ARCSIN_CC,
    FUNC_ARCCOS_CC,
    FUNC_ARCTAN_CC,
    FUNC_SINH_CC,
    FUNC_COSH_CC,
    FUNC_TANH_CC,
    FUNC_ARCSINH_CC,
    FUNC_ARCCOSH_CC,
    FUNC_ARCTANH_CC,

    FUNC_LOG_CC,
    FUNC_LOG1P_CC,
    FUNC_LOG10_CC,
    FUNC_EXP_CC,
    FUNC_EXPM1_CC,

    FUNC_CC_LAST
};


typedef void (*FuncCCPtr)(cdouble*, cdouble*);

/* The order of this array must match the FuncCCCodes enum above */
FuncCCPtr functions_cc[] = {
    nc_sqrt,
    nc_sin,
    nc_cos,
    nc_tan,
    nc_asin,
    nc_acos,
    nc_atan,
    nc_sinh,
    nc_cosh,
    nc_tanh,
    nc_asinh,
    nc_acosh,
    nc_atanh,
    nc_log,
    nc_log1p,
    nc_log10,
    nc_exp,
    nc_expm1,
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

typedef void (*FuncCCPtr_vml)(int, const MKL_Complex16[], MKL_Complex16[]);

/* The order of this array must match the FuncCCCodes enum above */
FuncCCPtr_vml functions_cc_vml[] = {
    vzSqrt,
    vzSin,
    vzCos,
    vzTan,
    vzAsin,
    vzAcos,
    vzAtan,
    vzSinh,
    vzCosh,
    vzTanh,
    vzAsinh,
    vzAcosh,
    vzAtanh,
    vzLn,
    vzLog1p, //poor approximation
    vzLog10,
    vzExp,
    vzExpm1, //poor approximation
};
#endif


enum FuncCCCCodes {
    FUNC_POW_CCC = 0,

    FUNC_CCC_LAST
};

typedef void (*FuncCCCPtr)(cdouble*, cdouble*, cdouble*);

FuncCCCPtr functions_ccc[] = {
    nc_pow,
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
size_from_sig(PyObject *o)
{
    intp size = 0;
    char *s = PyString_AsString(o);
    if (!s) return -1;
    for (; *s != '\0'; s++) {
        int x = size_from_char(*s);
        if (x == -1) return -1;
        size += x;
    }
    return size;
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
    Py_ssize_t n;
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
    Py_ssize_t prog_len, n_buffers, n_inputs;
    int rno, pc, arg, argloc, argno, sig;
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
    for (rno = n_inputs+1; rno < n_buffers; rno++) {
        char *bufend = self->mem[rno] + BLOCK_SIZE1 * size_from_char(fullsig[rno]);
        if ( (bufend - self->rawmem) > self->rawmemsize) {
            PyErr_Format(PyExc_RuntimeError, "invalid program: too many buffers");
            return -1;
        }
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
                PyErr_Format(PyExc_RuntimeError, "invalid program: illegal opcode at %i (%c)", pc, op);
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
                if (op == OP_FUNC_FF) {
                    if (arg < 0 || arg >= FUNC_FF_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_FFF) {
                    if (arg < 0 || arg >= FUNC_FFF_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_DD) {
                    if (arg < 0 || arg >= FUNC_DD_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_DDD) {
                    if (arg < 0 || arg >= FUNC_DDD_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_CC) {
                    if (arg < 0 || arg >= FUNC_CC_LAST) {
                        PyErr_Format(PyExc_RuntimeError, "invalid program: funccode out of range (%i) at %i", arg, argloc);
                        return -1;
                    }
                } else if (op == OP_FUNC_CCC) {
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
    int n_constants, n_inputs, n_temps;
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

    /* Compute the size of registers. */
    rawmemsize = 0;
    for (i = 0; i < n_constants; i++)
        rawmemsize += itemsizes[i];
    rawmemsize += size_from_sig(tempsig);  /* no string temporaries */
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

    /* Fill in 'mem' for temps */
    for (i = 0; i < n_temps; i++) {
        char c = PyString_AS_STRING(tempsig)[i];
        int size = size_from_char(c);
        /* XXX: This check is quite useless, since using a string temporary
           still causes a crash when freeing rawmem.  Why? */
        if (c == 's') {
            PyErr_SetString(PyExc_NotImplementedError,
                            "string temporaries are not supported");
            break;
        }
        mem[i+n_inputs+n_constants+1] = rawmem + mem_offset;
        mem_offset += BLOCK_SIZE1 * size;
        memsteps[i+n_inputs+n_constants+1] = memsizes[i+n_inputs+n_constants+1] = size;
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
    unsigned int n_inputs;
    unsigned int r_end;
    char *output;
    char **inputs;
    char **mem;
    intp *memsteps;
    intp *memsizes;
    struct index_data *index_data;
};

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

static inline int
vm_engine_1(int start, int blen, struct vm_params params, int *pc_error)
{
    unsigned int index;
    for (index = start; index < blen; index += BLOCK_SIZE1) {
#define VECTOR_SIZE BLOCK_SIZE1
#include "interp_body.c"
#undef VECTOR_SIZE
    }
    return 0;
}

static inline int
vm_engine_2(int start, int blen, struct vm_params params, int *pc_error)
{
    unsigned int index;
    for (index = start; index < blen; index += BLOCK_SIZE2) {
#define VECTOR_SIZE BLOCK_SIZE2
#include "interp_body.c"
#undef VECTOR_SIZE
    }
    return 0;
}

static inline int
vm_engine_rest(int start, int blen, struct vm_params params, int *pc_error)
{
    unsigned int index = start;
    unsigned int rest = blen - start;
#define VECTOR_SIZE rest
#include "interp_body.c"
#undef VECTOR_SIZE
    return 0;
}

static int
run_interpreter(NumExprObject *self, int len, char *output, char **inputs,
                struct index_data *index_data, int *pc_error)
{
    int r;
    Py_ssize_t plen;
    unsigned int blen1, blen2;
    struct vm_params params;

    *pc_error = -1;
    if (PyString_AsStringAndSize(self->program, (char **)&(params.program),
                                 &plen) < 0) {
        return -1;
    }
    params.prog_len = plen;
    if ((params.n_inputs = PyObject_Length(self->signature)) == -1)
        return -1;

    params.output = output;
    params.inputs = inputs;
    params.index_data = index_data;
    params.mem = self->mem;
    params.memsteps = self->memsteps;
    params.memsizes = self->memsizes;
    params.r_end = PyString_Size(self->fullsig);
    blen1 = len - len % BLOCK_SIZE1;
    r = vm_engine_1(0, blen1, params, pc_error);
    if (r < 0) return r;
    if (len != blen1) {
        blen2 = len - len % BLOCK_SIZE2;
        r = vm_engine_2(blen1, blen2, params, pc_error);
        if (r < 0) return r;
        if (len != blen2) {
            r = vm_engine_rest(blen2, len, params, pc_error);
            if (r < 0) return r;
        }
    }
    return 0;
}

/* keyword arguments are ignored! */
static PyObject *
NumExpr_run(NumExprObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *output = NULL, *a_inputs = NULL;
    struct index_data *inddata = NULL;
    unsigned int n_inputs, n_dimensions = 0;
    intp shape[MAX_DIMS];
    int i, j, size, r, pc_error;
    char **inputs = NULL;
    intp strides[MAX_DIMS]; /* clean up XXX */

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

static PyMethodDef module_methods[] = {
#ifdef USE_VML
    {"_get_vml_version", _get_vml_version, METH_VARARGS,
     "Get the VML/MKL library version."},
    {"_set_vml_accuracy_mode", _set_vml_accuracy_mode, METH_VARARGS,
     "Set accuracy mode for VML functions."},
    {"_set_vml_num_threads", _set_vml_num_threads, METH_VARARGS,
     "Suggests a maximum number of threads to be used in VML operations."},
#endif
    {NULL}
};

void
initinterpreter(void)
{
    PyObject *m, *d, *o;
    int r;

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

#define add_op(sname, name) o = PyInt_FromLong(name);   \
    r = PyDict_SetItemString(d, sname, o);              \
    Py_XDECREF(o);                                      \
    if (r < 0) {PyErr_SetString(PyExc_RuntimeError, "add_op"); return;}
    add_op("noop", OP_NOOP);

    add_op("copy_bb", OP_COPY_BB);
    add_op("invert_bb", OP_INVERT_BB);
    add_op("and_bbb", OP_AND_BBB);
    add_op("or_bbb", OP_OR_BBB);

    add_op("eq_bbb", OP_EQ_BBB);
    add_op("ne_bbb", OP_NE_BBB);

    add_op("gt_bii", OP_GT_BII);
    add_op("ge_bii", OP_GE_BII);
    add_op("eq_bii", OP_EQ_BII);
    add_op("ne_bii", OP_NE_BII);

    add_op("gt_bll", OP_GT_BLL);
    add_op("ge_bll", OP_GE_BLL);
    add_op("eq_bll", OP_EQ_BLL);
    add_op("ne_bll", OP_NE_BLL);

    add_op("gt_bff", OP_GT_BFF);
    add_op("ge_bff", OP_GE_BFF);
    add_op("eq_bff", OP_EQ_BFF);
    add_op("ne_bff", OP_NE_BFF);

    add_op("gt_bdd", OP_GT_BDD);
    add_op("ge_bdd", OP_GE_BDD);
    add_op("eq_bdd", OP_EQ_BDD);
    add_op("ne_bdd", OP_NE_BDD);

    add_op("gt_bss", OP_GT_BSS);
    add_op("ge_bss", OP_GE_BSS);
    add_op("eq_bss", OP_EQ_BSS);
    add_op("ne_bss", OP_NE_BSS);

    add_op("ones_like_ii", OP_ONES_LIKE_II);
    add_op("copy_ii", OP_COPY_II);
    add_op("neg_ii", OP_NEG_II);
    add_op("add_iii", OP_ADD_III);
    add_op("sub_iii", OP_SUB_III);
    add_op("mul_iii", OP_MUL_III);
    add_op("div_iii", OP_DIV_III);
    add_op("pow_iii", OP_POW_III);
    add_op("mod_iii", OP_MOD_III);
    add_op("where_ibii", OP_WHERE_IBII);

    add_op("cast_li", OP_CAST_LI);
    add_op("ones_like_ll", OP_ONES_LIKE_LL);
    add_op("copy_ll", OP_COPY_LL);
    add_op("neg_ll", OP_NEG_LL);
    add_op("add_lll", OP_ADD_LLL);
    add_op("sub_lll", OP_SUB_LLL);
    add_op("mul_lll", OP_MUL_LLL);
    add_op("div_lll", OP_DIV_LLL);
    add_op("pow_lll", OP_POW_LLL);
    add_op("mod_lll", OP_MOD_LLL);
    add_op("where_lbll", OP_WHERE_LBLL);

    add_op("cast_fi", OP_CAST_FI);
    add_op("cast_fl", OP_CAST_FL);
    add_op("copy_ff", OP_COPY_FF);
    add_op("ones_like_ff", OP_ONES_LIKE_FF);
    add_op("neg_ff", OP_NEG_FF);
    add_op("add_fff", OP_ADD_FFF);
    add_op("sub_fff", OP_SUB_FFF);
    add_op("mul_fff", OP_MUL_FFF);
    add_op("div_fff", OP_DIV_FFF);
    add_op("pow_fff", OP_POW_FFF);
    add_op("mod_fff", OP_MOD_FFF);
    add_op("sqrt_ff", OP_SQRT_FF);
    add_op("where_fbff", OP_WHERE_FBFF);
    add_op("func_ff", OP_FUNC_FF);
    add_op("func_fff", OP_FUNC_FFF);

    add_op("cast_di", OP_CAST_DI);
    add_op("cast_dl", OP_CAST_DL);
    add_op("cast_df", OP_CAST_DF);
    add_op("copy_dd", OP_COPY_DD);
    add_op("ones_like_dd", OP_ONES_LIKE_DD);
    add_op("neg_dd", OP_NEG_DD);
    add_op("add_ddd", OP_ADD_DDD);
    add_op("sub_ddd", OP_SUB_DDD);
    add_op("mul_ddd", OP_MUL_DDD);
    add_op("div_ddd", OP_DIV_DDD);
    add_op("pow_ddd", OP_POW_DDD);
    add_op("mod_ddd", OP_MOD_DDD);
    add_op("sqrt_dd", OP_SQRT_DD);
    add_op("where_dbdd", OP_WHERE_DBDD);
    add_op("func_dd", OP_FUNC_DD);
    add_op("func_ddd", OP_FUNC_DDD);

    add_op("eq_bcc", OP_EQ_BCC);
    add_op("ne_bcc", OP_NE_BCC);

    add_op("cast_ci", OP_CAST_CI);
    add_op("cast_cl", OP_CAST_CL);
    add_op("cast_cf", OP_CAST_CF);
    add_op("cast_cd", OP_CAST_CD);
    add_op("copy_cc", OP_COPY_CC);
    add_op("ones_like_cc", OP_ONES_LIKE_CC);
    add_op("neg_cc", OP_NEG_CC);
    add_op("add_ccc", OP_ADD_CCC);
    add_op("sub_ccc", OP_SUB_CCC);
    add_op("mul_ccc", OP_MUL_CCC);
    add_op("div_ccc", OP_DIV_CCC);
    add_op("where_cbcc", OP_WHERE_CBCC);
    add_op("func_cc", OP_FUNC_CC);
    add_op("func_ccc", OP_FUNC_CCC);

    add_op("real_dc", OP_REAL_DC);
    add_op("imag_dc", OP_IMAG_DC);
    add_op("complex_cdd", OP_COMPLEX_CDD);

    add_op("copy_ss", OP_COPY_SS);

    add_op("sum_iin", OP_SUM_IIN);
    add_op("sum_lln", OP_SUM_LLN);
    add_op("sum_ffn", OP_SUM_FFN);
    add_op("sum_ddn", OP_SUM_DDN);
    add_op("sum_ccn", OP_SUM_CCN);

    add_op("prod_iin", OP_PROD_IIN);
    add_op("prod_lln", OP_PROD_LLN);
    add_op("prod_ffn", OP_PROD_FFN);
    add_op("prod_ddn", OP_PROD_DDN);
    add_op("prod_ccn", OP_PROD_CCN);

#undef add_op

    if (PyModule_AddObject(m, "opcodes", d) < 0) return;

    d = PyDict_New();
    if (!d) return;

#define add_func(sname, name) o = PyInt_FromLong(name); \
    r = PyDict_SetItemString(d, sname, o);              \
    Py_XDECREF(o);                                      \
    if (r < 0) {PyErr_SetString(PyExc_RuntimeError, "add_func"); return;}

    add_func("sqrt_ff", FUNC_SQRT_FF);
    add_func("sin_ff", FUNC_SIN_FF);
    add_func("cos_ff", FUNC_COS_FF);
    add_func("tan_ff", FUNC_TAN_FF);
    add_func("arcsin_ff", FUNC_ARCSIN_FF);
    add_func("arccos_ff", FUNC_ARCCOS_FF);
    add_func("arctan_ff", FUNC_ARCTAN_FF);
    add_func("sinh_ff", FUNC_SINH_FF);
    add_func("cosh_ff", FUNC_COSH_FF);
    add_func("tanh_ff", FUNC_TANH_FF);
    add_func("arcsinh_ff", FUNC_ARCSINH_FF);
    add_func("arccosh_ff", FUNC_ARCCOSH_FF);
    add_func("arctanh_ff", FUNC_ARCTANH_FF);

    add_func("log_ff", FUNC_LOG_FF);
    add_func("log1p_ff", FUNC_LOG1P_FF);
    add_func("log10_ff", FUNC_LOG10_FF);
    add_func("exp_ff", FUNC_EXP_FF);
    add_func("expm1_ff", FUNC_EXPM1_FF);

    add_func("arctan2_fff", FUNC_ARCTAN2_FFF);
    add_func("fmod_fff", FUNC_FMOD_FFF);

    add_func("sqrt_dd", FUNC_SQRT_DD);
    add_func("sin_dd", FUNC_SIN_DD);
    add_func("cos_dd", FUNC_COS_DD);
    add_func("tan_dd", FUNC_TAN_DD);
    add_func("arcsin_dd", FUNC_ARCSIN_DD);
    add_func("arccos_dd", FUNC_ARCCOS_DD);
    add_func("arctan_dd", FUNC_ARCTAN_DD);
    add_func("sinh_dd", FUNC_SINH_DD);
    add_func("cosh_dd", FUNC_COSH_DD);
    add_func("tanh_dd", FUNC_TANH_DD);
    add_func("arcsinh_dd", FUNC_ARCSINH_DD);
    add_func("arccosh_dd", FUNC_ARCCOSH_DD);
    add_func("arctanh_dd", FUNC_ARCTANH_DD);

    add_func("log_dd", FUNC_LOG_DD);
    add_func("log1p_dd", FUNC_LOG1P_DD);
    add_func("log10_dd", FUNC_LOG10_DD);
    add_func("exp_dd", FUNC_EXP_DD);
    add_func("expm1_dd", FUNC_EXPM1_DD);

    add_func("arctan2_ddd", FUNC_ARCTAN2_DDD);
    add_func("fmod_ddd", FUNC_FMOD_DDD);

    add_func("sqrt_cc", FUNC_SQRT_CC);
    add_func("sin_cc", FUNC_SIN_CC);
    add_func("cos_cc", FUNC_COS_CC);
    add_func("tan_cc", FUNC_TAN_CC);
    add_func("arcsin_cc", FUNC_ARCSIN_CC);
    add_func("arccos_cc", FUNC_ARCCOS_CC);
    add_func("arctan_cc", FUNC_ARCTAN_CC);
    add_func("sinh_cc", FUNC_SINH_CC);
    add_func("cosh_cc", FUNC_COSH_CC);
    add_func("tanh_cc", FUNC_TANH_CC);
    add_func("arcsinh_cc", FUNC_ARCSINH_CC);
    add_func("arccosh_cc", FUNC_ARCCOSH_CC);
    add_func("arctanh_cc", FUNC_ARCTANH_CC);

    add_func("log_cc", FUNC_LOG_CC);
    add_func("log1p_cc", FUNC_LOG1P_CC);
    add_func("log10_cc", FUNC_LOG10_CC);
    add_func("exp_cc", FUNC_EXP_CC);
    add_func("expm1_cc", FUNC_EXPM1_CC);
    add_func("pow_ccc", FUNC_POW_CCC);

#undef add_func

    if (PyModule_AddObject(m, "funccodes", d) < 0) return;

    if (PyModule_AddObject(m, "allaxes", PyInt_FromLong(255)) < 0) return;
    if (PyModule_AddObject(m, "maxdims", PyInt_FromLong(MAX_DIMS)) < 0) return;

}
