// WARNING: THIS FILE IS AUTO-GENERATED PRIOR TO COMPILATION.
// Editing should be done on the associated stub file instead.
#ifndef NUMEXPR_INTERPRETER_HPP
#define NUMEXPR_INTERPRETER_HPP

#include <numpy/npy_common.h>
#include "numexpr_config.hpp"

// Forward declaration
#include "numexpr_object.hpp"

// Here is where generator.py inserts the necessary header definitions, 
// anything between these lines will be replaced by generated code.
#define BLOCK_SIZE1 4096
#define BLOCK_SIZE2 32
#define BOUNDS_CHECK(arg) if ((arg) >= params->n_reg) { return -2; }
#define OP_END 430
// End of GENERATED CODE BLOCK



// The #USE_VML define is set with the code generator
#ifdef USE_VML
#include "mkl_vml.h"
#include "mkl_service.h"
#endif

// Global state which holds thread parameters
extern thread_data th_params;

PyObject *NumExpr_run(NumExprObject *self, PyObject *args, PyObject *kwds);

int NPYENUM_from_dchar(char c);
char get_return_sig(NumExprObject *self);
int get_temps_space(NumExprObject *self, size_t block_size);
void free_temps_space(const NumExprObject *self);
int vm_engine_iter_task(NpyIter *iter, const NumExprObject *params, 
                        int *pc_error, char **errorMessage);
NumExprObject* NumExprObject_copy_threadsafe( const NumExprObject *self );                


#endif // NUMEXPR_INTERPRETER_HPP