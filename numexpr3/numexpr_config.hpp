#ifndef NUMEXPR_CONFIG_HPP
#define NUMEXPR_CONFIG_HPP

// x86 platform works with unaligned reads and writes
// MW: I have seen exceptions to this when the compiler chooses to use aligned SSE
#if (defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64))
#  define USE_UNALIGNED_ACCESS 1
#endif


#include <numpy/npy_common.h>

// Maximum number of arguments (including return) per operation
// Not related to NPY_MAXARGS.
#define NUMEXPR_MAX_ARGS 4
// Maximum number of arrays in a NumExprObject
#define NE_MAX_BUFFERS 255
// Size of a register in a program when encoded as bytes
#define NE_REGISTER      npy_uint8
// Size of an operation in a program when encoded as bytes
#define NE_WORD          npy_uint16
// Each program is an opword, followed by 4 registers (return,arg1,arg2,arg3)
#define NE_PROG_LEN (Py_ssize_t)(sizeof(NE_WORD) + NUMEXPR_MAX_ARGS*sizeof(NE_REGISTER))
//This is in vm_engine_iter_parallel() and is a potential source for future optimization
#define TASKS_PER_THREAD 16 
    // RAM: old comment: Try to make it so each thread gets 16 tasks.  This is 
    // a compromise between 1 task per thread and one block per task.
    
// RAM: This is also an arbitrary for reductions
#define INNER_LOOP_MAX_SIZE 64
// The default block size on module load.
#define DEFAULT_BLOCK 4096

#define OP_NOOP 0
// TODO: OP_REDUCTION will need to be inserted by the code generator.
#define OP_REDUCTION 9999999
#define OP_PROD 99999
#define OP_SUM 99999




// The maximum number of threads (for some static arrays).
// Choose this large enough for most monsters out there.
// Keep in sync this with the number in __init__.py. 
//#define MAX_THREADS 256
#define DEFAULT_THREADS 1

#if defined(_WIN32)
  #include "win32/pthread.h"
  #include <process.h>
  #define getpid _getpid
#else
  #include <pthread.h>
  #include "unistd.h"
#endif

// So apparently this _only_ works if SciPy was built with MKL?
// This should also go into the generator.
//#ifdef SCIPY_MKL_H
//#define USE_VML
//#endif
//
//#ifdef USE_VML
//#include "mkl_vml.h"
//#include "mkl_service.h"
//#endif

#ifdef _WIN32
  #ifndef __MINGW32__
    #include "missing_posix_functions.hpp"
  #endif
  #include "msvc_function_stubs.hpp"
#endif

#endif // NUMEXPR_CONFIG_HPP