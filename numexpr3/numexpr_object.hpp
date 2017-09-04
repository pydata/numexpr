#ifndef NUMEXPR_OBJECT_HPP
#define NUMEXPR_OBJECT_HPP
/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: BSD
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#define KIND_ARRAY  1
#define KIND_SCALAR 2
#define KIND_TEMP   4
#define KIND_RETURN 8

// self is a struct NumExprObject
#define GET_RETURN_REG(self) self->registers[ self->program[self->n_reg-1].ret ]

// self is a struct NumExprObject
#define LAST_OP(self) self->program[self->program_len-1].op

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
    npy_uint8      kind;       // 1 = array, 2 = scalar, 4 = temp, 8 = return
    npy_intp       itemsize;   // element size in bytes   (was: memsizes)
    npy_intp       stride;     // How many bytes until next element  (was: memsteps)
};

// Presently the PyObject_HEAD macro expands into a single PyObject
// When copying data we generally don't want to overwrite the head.
// https://docs.python.org/3/c-api/structures.html
#define SIZEOF_PYOBJECT_HEAD sizeof(PyObject)
struct NumExprObject
{
    PyObject_HEAD
    struct NumExprOperation  *program;
    struct NumExprReg        *registers;
    
    // a chunk of raw memory for storing scalar BLOCKs, just a reference
    // for efficient garbage collection.
    char                    *scalar_mem;  
    npy_intp                 scalar_mem_size;
    // temporaries need a block of memory at least sum(BLOCKSIZE1*total_temp_itemsize)
    Py_ssize_t               total_temp_itemsize; 
    int                      program_len;    
    NE_REGISTER              n_reg;
    NE_REGISTER              n_array;          // The number of arrays AND return registers
    NE_REGISTER              n_scalar;
    NE_REGISTER              n_temp;  
    NE_REGISTER              returnReg;        // The index of the return in registers, defaults to -1
    NE_REGISTER              returnOperand;    // The index of the return in the iterator, defaults to -1
};


extern PyTypeObject NumExprType;

#endif // NUMEXPR_OBJECT_HPP
