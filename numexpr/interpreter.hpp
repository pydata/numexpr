#ifndef NUMEXPR_INTERPRETER_HPP
#define NUMEXPR_INTERPRETER_HPP

#include "numexpr_config.hpp"

// Forward declaration
struct NumExprObject;

enum OpCodes {
#define OPCODE(n, e, ...) e = n,
#include "opcodes.hpp"
#undef OPCODE
};

enum FuncF4F4Codes {
#define FUNC_F4F4(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_F4F4
};

enum FuncF4F4F4Codes {
#define FUNC_F4F4F4(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_F4F4F4
};

enum FuncF8F8Codes {
#define FUNC_F8F8(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_F8F8
};

enum FuncF8F8F8Codes {
#define FUNC_F8F8F8(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_F8F8F8
};

enum FuncC16C16Codes {
#define FUNC_C16C16(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_C16C16
};

enum FuncC16C16C16Codes {
#define FUNC_C16C16C16(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_C16C16C16
};

enum FuncC8C8Codes {
#define FUNC_C8C8(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_C8C8
};

enum FuncC8C8C8Codes {
#define FUNC_C8C8C8(fop, ...) fop,
#include "functions.hpp"
#undef FUNC_C8C8C8
};


struct vm_params {
    int prog_len;
    unsigned short *program;
    int n_inputs;
    int n_constants;
    int n_temps;
    unsigned int r_end;
    char *output;
    char **inputs;
    char **mem;
    npy_intp *memsteps;
    npy_intp *memsizes;
    struct index_data *index_data;
    // Memory for output buffering. If output buffering is unneeded,
    // it contains NULL.
    char *out_buffer;
};

// Structure for parameters in worker threads
struct thread_data {
    npy_intp start;
    npy_intp vlen;
    npy_intp block_size;
    vm_params params;
    int ret_code;
    int *pc_error;
    char **errmsg;
    // One memsteps array per thread
    npy_intp *memsteps[MAX_THREADS];
    // One iterator per thread */
    NpyIter *iter[MAX_THREADS];
    // When doing nested iteration for a reduction
    NpyIter *reduce_iter[MAX_THREADS];
    // Flag indicating reduction is the outer loop instead of the inner
    bool reduction_outer_loop;
    // Flag indicating whether output buffering is needed
    bool need_output_buffering;
};

// Global state which holds thread parameters
extern thread_data th_params;

PyObject *NumExpr_run(NumExprObject *self, PyObject *args, PyObject *kwds);

unsigned short get_return_sig(PyObject* program);
int check_program(NumExprObject *self);
int get_temps_space(const vm_params& params, char **mem, size_t block_size);
void free_temps_space(const vm_params& params, char **mem);
int vm_engine_iter_task(NpyIter *iter, npy_intp *memsteps,
                    const vm_params& params, int *pc_error, char **errmsg);

#endif // NUMEXPR_INTERPRETER_HPP