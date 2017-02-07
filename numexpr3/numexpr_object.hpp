#ifndef NUMEXPR_OBJECT_HPP
#define NUMEXPR_OBJECT_HPP
/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: BSD
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/


#define KIND_ARRAY  0
#define KIND_SCALAR 1
#define KIND_TEMP   2
#define KIND_RETURN 3

// self is a struct NumExprObject
#define GET_RETURN_REG(self) self->registers[ self->program[self->n_reg-1].ret ]

// self is a struct NumExprObject
#define LAST_OP(self) self->program[self->program_len].op

// RAM: What does vm_params do that's (significantly) different from NumExprObject?
//struct vm_params {
//    int prog_len;
//    NE_WORD *program;
//    int n_ndarrays;
//    int n_scalars;
//    int n_temp;
//    NE_WORD r_end;
//    char *output;
//    char **inputs;
//    char **mem;
//    npy_intp *memsteps;
//    npy_intp *memsizes;
//    //struct index_data *index_data;
//    // Memory for output buffering. If output buffering is unneeded,
//    // it contains NULL.
//    char *out_buffer;
//};

struct NumExprOperation
{
    NE_WORD      op;
    NE_REGISTER  ret;
    NE_REGISTER  arg1;
    NE_REGISTER  arg2;
    NE_REGISTER  arg3;
};

// We can pack the tuples in Python and pass them in directly?
// General idea here is to not access the Python API directly after initial 
// processing.
struct NumExprReg 
{
    char          *mem;        // Pointer to array data for scalars and temps (npy_iter used for arrays)
    char           dchar;      // numpy.dtype.char
    npy_uint8      kind;       // 0 = array, 1 = scalar, 2 = temp
    npy_intp       itemsize;   // element size in bytes   (was: memsizes)
    npy_intp       stride;     // How many bytes until next element  (was: memsteps)
    npy_intp       elements;   // number of array elements
};


// Perhaps the Python object should be Py_NumExprObject and this struct 
// doesn't need the PyObject_HEAD declaration?
struct NumExprObject
{
    PyObject_HEAD
    struct NumExprOperation  *program;
    struct NumExprReg        *registers;
    // Memory for output buffering. If output buffering is unneeded,
    // it contains NULL.
    char                     *outBuffer;
    // a chunk of raw memory for storing scalar BLOCKs, just a reference
    // for efficient garbage collection.
    char                    *scalar_mem;  
    int                      program_len;    
    NE_REGISTER              n_reg;
    NE_REGISTER              n_ndarray;
    NE_REGISTER              n_scalar;
    NE_REGISTER              n_temp;
};
        


// Structure for parameters in worker threads
struct thread_data {
    npy_intp start;
    npy_intp vlen;
    npy_intp block_size;
    // RAM: we could make params not-a-pointer so it's not being 
    // allocated and de-allocated.
    NumExprObject *params;
    int ret_code;
    int *pc_error;
    char **errorMessage;
    // One strides array per thread
    npy_intp *stridesArray[MAX_THREADS];
    // One iterator per thread */
    NpyIter *iter[MAX_THREADS];
    // When doing nested iteration for a reduction
    NpyIter *reduce_iter[MAX_THREADS];
    // Flag indicating reduction is the outer loop instead of the inner
    bool reduction_outer_loop;
    // Flag indicating whether output buffering is needed
    bool need_output_buffering;
};


extern PyTypeObject NumExprType;

#endif // NUMEXPR_OBJECT_HPP
