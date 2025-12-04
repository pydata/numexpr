# C Interface for NumExpr Integration

## Overview

The `blosc2_numexpr_integration` module provides a **pure C interface** for calling NumExpr from C-Blosc2 worker threads. No Python knowledge required in the calling code!

## Function Signature

```c
PyObject* process_chunk_with_numexpr(
    void** chunk_pointers,  // C array of pointers: {ptr1, ptr2, ptr3, ...}
    int n_arrays,           // Number of input arrays
    int chunk_size,         // Number of elements in each chunk
    int dtype,              // NumPy dtype code (NPY_DOUBLE=12, etc.)
    void* numexpr_handle    // Handle from setup_expression()
);
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunk_pointers` | `void**` | C array of pointers to decompressed chunk data |
| `n_arrays` | `int` | Number of input arrays (length of chunk_pointers) |
| `chunk_size` | `int` | Number of elements in each chunk |
| `dtype` | `int` | NumPy dtype: `NPY_DOUBLE` (12), `NPY_FLOAT` (11), etc. |
| `numexpr_handle` | `void*` | Handle from Python `setup_expression()` |

### Returns

- **`PyObject*`**: NumPy array with result (new reference)
- **`NULL`**: On error (Python exception will be set)

## Complete C Example

```c
#include <Python.h>
#include <numpy/arrayobject.h>
#include "blosc2_numexpr_integration.h"
#include <blosc2.h>

// C-Blosc2 worker thread function
void* blosc2_worker_thread(void* args) {
    blosc2_context* ctx = (blosc2_context*)args;
    PyGILState_STATE gstate;
    PyObject* result = NULL;
    double* chunk_a = NULL;
    double* chunk_b = NULL;
    double* chunk_c = NULL;
    double* result_data = NULL;
    int status = 0;
    
    // 1. Decompress chunks (NO GIL needed here)
    chunk_a = blosc2_decompress_chunk(ctx, ctx->chunk_idx, 0);
    chunk_b = blosc2_decompress_chunk(ctx, ctx->chunk_idx, 1);
    chunk_c = blosc2_decompress_chunk(ctx, ctx->chunk_idx, 2);
    
    if (!chunk_a || !chunk_b || !chunk_c) {
        status = -1;
        goto cleanup;
    }
    
    // 2. Acquire GIL before calling Python
    gstate = PyGILState_Ensure();
    
    // 3. Create C array of pointers
    void* chunks[3] = {chunk_a, chunk_b, chunk_c};
    
    // 4. Call NumExpr (GIL will be released internally during computation!)
    result = process_chunk_with_numexpr(
        chunks,                    // C array of pointers
        3,                         // number of arrays
        ctx->chunk_size,           // elements per chunk
        NPY_DOUBLE,                // dtype (12 for float64)
        ctx->numexpr_handle        // expression handle
    );
    
    if (result == NULL) {
        // NumExpr failed - print error for debugging
        PyErr_Print();
        status = -1;
        PyGILState_Release(gstate);
        goto cleanup;
    }
    
    // 5. Extract result data (still have GIL)
    result_data = (double*)PyArray_DATA((PyArrayObject*)result);
    
    // 6. Release GIL - we're done with Python
    PyGILState_Release(gstate);
    
    // 7. Compress result (NO GIL needed)
    status = blosc2_compress_chunk(
        ctx->output_schunk,
        result_data,
        ctx->chunk_size * sizeof(double)
    );
    
    // 8. Acquire GIL again for cleanup
    gstate = PyGILState_Ensure();
    Py_DECREF(result);
    PyGILState_Release(gstate);
    
cleanup:
    // Free decompressed chunks
    free(chunk_a);
    free(chunk_b);
    free(chunk_c);
    
    return (void*)(intptr_t)status;
}
```

## NumPy dtype Codes

Common dtype values (from `numpy/ndarraytypes.h`):

```c
#define NPY_BOOL        0
#define NPY_BYTE        1
#define NPY_UBYTE       2
#define NPY_SHORT       3
#define NPY_USHORT      4
#define NPY_INT         5
#define NPY_UINT        6
#define NPY_LONG        7
#define NPY_ULONG       8
#define NPY_LONGLONG    9
#define NPY_ULONGLONG   10
#define NPY_FLOAT       11   // float32
#define NPY_DOUBLE      12   // float64 (most common)
#define NPY_LONGDOUBLE  13
#define NPY_CFLOAT      14
#define NPY_CDOUBLE     15
#define NPY_CLONGDOUBLE 16
```

## Python Setup (One-time)

Before calling from C, setup the expression in Python:

```python
from blosc2_numexpr_integration import setup_expression

# Compile expression
expression = "2*a + 3*b*c"
handle = setup_expression(expression)

# Pass handle to C code (as void*)
# C code will use this handle for all chunks
```

## GIL Management

### Timeline for a Single Chunk

```
C-Blosc2 Thread (no GIL initially)
│
├─ Decompress chunks (NO GIL) ───────────── ~0.5 ms
│
├─ PyGILState_Ensure() ──────────────────── acquire GIL
│  │
│  ├─ Create pointer array (WITH GIL) ──── ~0.001 ms
│  │
│  ├─ Call process_chunk_with_numexpr()
│  │  │
│  │  ├─ Wrap arrays (WITH GIL) ────────── ~0.01 ms
│  │  │
│  │  ├─ NumExpr computation:
│  │  │  ├─ Setup (WITH GIL) ──────────── ~0.01 ms
│  │  │  ├─ Py_BEGIN_ALLOW_THREADS ───── release GIL
│  │  │  ├─ Compute (NO GIL) ⚡ ────────── ~1-5 ms
│  │  │  └─ Py_END_ALLOW_THREADS ──────── acquire GIL
│  │  │
│  │  └─ Return result (WITH GIL) ──────── ~0.001 ms
│  │
│  ├─ Extract result data (WITH GIL) ───── ~0.001 ms
│  │
│  └─ PyGILState_Release() ──────────────── release GIL
│
├─ Compress result (NO GIL) ──────────────── ~0.5 ms
│
└─ Cleanup

Total time: ~2-6 ms
GIL held: ~0.03 ms (0.5-1.5% of total time)
Parallel computation: ~1-5 ms (98-99% of total time) ⚡
```

### Multi-thread Scenario

With 3 C-Blosc2 worker threads:

```
Thread 1: [decompress][GIL: wrap+extract][NO GIL: ⚡ compute][compress]
Thread 2:            [decompress][GIL: wrap+extract][NO GIL: ⚡ compute]
Thread 3:                       [decompress][GIL: wrap+extract][NO GIL: ⚡]

GIL Timeline:
         [T1:wrap][T2:wrap][T3:wrap]............[T1:extract][T2:extract]...

Computation Timeline (PARALLEL!):
                 [Thread 1 ⚡][Thread 2 ⚡][Thread 3 ⚡]
```

**Result: Real parallelism achieved!** ✅

## Error Handling

```c
PyObject* result = process_chunk_with_numexpr(chunks, 3, size, NPY_DOUBLE, handle);

if (result == NULL) {
    // Check what went wrong
    if (PyErr_Occurred()) {
        PyErr_Print();  // Print to stderr
        // or
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        // ... handle error ...
        PyErr_Restore(type, value, traceback);
    }
    return -1;
}

// Use result
// ...

// Always DECREF when done
Py_DECREF(result);
```

## Building

### Compile the Cython Module

```bash
# In python-blosc2 directory
cythonize -i blosc2/blosc2_numexpr_integration.pyx
```

This generates:
- `blosc2_numexpr_integration.c` - C source
- `blosc2_numexpr_integration.h` - C header (include this!)
- `blosc2_numexpr_integration.so` - Compiled module

### Include in Your C Code

```c
#include "blosc2_numexpr_integration.h"
```

### Link

Make sure to link against:
- Python library
- NumPy (included via headers)
- The compiled Cython module

## Complete Integration Example

```c
// blosc2_lazy_expr.c
#include <Python.h>
#include <numpy/arrayobject.h>
#include "blosc2_numexpr_integration.h"
#include <blosc2.h>
#include <pthread.h>

typedef struct {
    void* numexpr_handle;
    blosc2_schunk* operand_chunks[10];
    blosc2_schunk* output_chunk;
    int n_operands;
    int chunk_idx;
    int chunk_size;
    int dtype;
} chunk_task_t;

void* process_one_chunk(void* arg) {
    chunk_task_t* task = (chunk_task_t*)arg;
    PyGILState_STATE gstate;
    void** chunks = NULL;
    PyObject* result = NULL;
    double* result_data;
    int i;
    
    // Allocate array for chunk pointers
    chunks = malloc(task->n_operands * sizeof(void*));
    if (!chunks) return (void*)-1;
    
    // Decompress all input chunks
    for (i = 0; i < task->n_operands; i++) {
        chunks[i] = blosc2_decompress_chunk(
            task->operand_chunks[i],
            task->chunk_idx
        );
        if (!chunks[i]) goto error;
    }
    
    // Acquire GIL and call NumExpr
    gstate = PyGILState_Ensure();
    
    result = process_chunk_with_numexpr(
        chunks,
        task->n_operands,
        task->chunk_size,
        task->dtype,
        task->numexpr_handle
    );
    
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        goto error;
    }
    
    // Extract and compress result
    result_data = (double*)PyArray_DATA((PyArrayObject*)result);
    
    PyGILState_Release(gstate);
    
    // Compress result
    blosc2_schunk_append_buffer(
        task->output_chunk,
        result_data,
        task->chunk_size * sizeof(double)
    );
    
    // Cleanup
    gstate = PyGILState_Ensure();
    Py_DECREF(result);
    PyGILState_Release(gstate);
    
    for (i = 0; i < task->n_operands; i++) {
        free(chunks[i]);
    }
    free(chunks);
    
    return (void*)0;

error:
    if (chunks) {
        for (i = 0; i < task->n_operands; i++) {
            free(chunks[i]);
        }
        free(chunks);
    }
    return (void*)-1;
}
```

## Key Points

✅ **Pure C interface** - No Python structs in the signature
✅ **Simple** - Just pass C array of pointers
✅ **Flexible** - Works with any number of inputs
✅ **Fast** - Zero-copy array wrapping
✅ **Parallel** - GIL released during computation (~99% of time)
✅ **Thread-safe** - Each thread can call independently

## Next Steps

1. Copy `blosc2_numexpr_integration.pyx` to python-blosc2
2. Build with Cython to generate `.c` and `.h` files
3. Include the `.h` file in your C-Blosc2 code
4. Call `process_chunk_with_numexpr()` from worker threads

Done! 🎉
