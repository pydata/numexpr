# Cython Integration Guide for Python-Blosc2 + NumExpr C-API

## TL;DR: `nogil` vs `PyGILState_Ensure/Release`

**They are NOT equivalent!**

- **`nogil`**: Declares a function *can* run without GIL (but doesn't acquire/release it)
- **`with gil:`**: Temporarily *acquires* GIL inside a `nogil` function  
- **`PyGILState_Ensure/Release`**: C API to acquire/release GIL (Cython's `with gil:` uses this internally)

## The Pattern for C-Blosc2 Threads Calling Cython

```cython
# Declare function as nogil-compatible
cdef int my_function(...) noexcept nogil:
    # This function CAN be called without GIL
    # (e.g., from C-Blosc2 worker threads)
    
    # But we need GIL to call NumExpr C-API
    with gil:  # <-- Equivalent to PyGILState_Ensure()
        # Now we have GIL
        # Call NumExpr C-API
        result = numexpr_run_compiled_simple(...)
        # NumExpr will release GIL internally during computation!
    # <-- Equivalent to PyGILState_Release()
    
    # Back to nogil state
    return 0
```

## How It Works: Call Chain

```
Python-Blosc2                     Cython Wrapper              NumExpr C-API
═════════════                     ══════════════              ═════════════

┌─────────────────┐
│ Python code     │
│ expr.compute()  │
└────────┬────────┘
         │
         │ Pass function pointer & handle
         ▼
┌─────────────────────────────────────────┐
│ C-Blosc2 Worker Thread (NO GIL)         │
│                                         │
│ void worker() {                         │
│   // Decompress chunks                 │
│   void* chunk_a = decompress(...);     │
│   void* chunk_b = decompress(...);     │
│                                         │
│   // Call Cython function              │ ──┐
│   processor(chunk_a, chunk_b, ...);    │   │
│ }                                       │   │
└─────────────────────────────────────────┘   │
                                              │
                    ┌─────────────────────────┘
                    │
                    ▼
         ┌────────────────────────────────┐
         │ Cython: process_chunk() nogil  │
         │                                │
         │  with gil:  ◄──── Acquire GIL │
         │    wrap NumPy arrays           │
         │    call NumExpr C-API  ────────┼────┐
         │  # End with gil  ◄──── Release │    │
         │                                │    │
         │  return result                 │    │
         └────────────────────────────────┘    │
                                               │
                      ┌────────────────────────┘
                      │
                      ▼
           ┌──────────────────────────────────────┐
           │ NumExpr: numexpr_run_compiled_simple │
           │                                      │
           │  // Setup (with GIL)                 │
           │  Py_BEGIN_ALLOW_THREADS  ◄──── Drop  │
           │    // Computation (NO GIL!)          │
           │    vm_engine_iter_task(...)          │
           │  Py_END_ALLOW_THREADS   ◄──── Acquire│
           │  // Cleanup (with GIL)               │
           │                                      │
           │  return result                       │
           └──────────────────────────────────────┘
```

## Key Points

### 1. Function Signature for C-Blosc2 Callbacks

```cython
cdef int process_chunk_with_numexpr(
    void* chunk_a_data,
    void* chunk_b_data, 
    void* chunk_c_data,
    void* output_data,
    int64_t chunk_size,
    void* numexpr_handle
) noexcept nogil:  # <── KEY: noexcept nogil
    # Function can be called from C threads without GIL
    # ...
```

- **`noexcept`**: No Python exceptions will escape (required for nogil)
- **`nogil`**: Function can be called without holding GIL

### 2. Acquiring GIL When Needed

```cython
cdef int my_function() noexcept nogil:
    # We're in C-land, no GIL
    
    with gil:  # <── Acquire GIL (like PyGILState_Ensure)
        # Now we have GIL, can use Python/NumPy
        arr = <PyArrayObject*>PyArray_SimpleNewFromData(...)
        result = numexpr_run_compiled_simple(...)
        Py_DECREF(arr)
    # GIL released automatically (like PyGILState_Release)
    
    # Back to C-land
    return 0
```

### 3. NumExpr Releases GIL Internally

When you call `numexpr_run_compiled_simple()`:

```c
// Inside NumExpr (interpreter.cpp):
PyObject* result = ... // setup with GIL

Py_BEGIN_ALLOW_THREADS;  // ◄── Release GIL
// Heavy computation here - REAL PARALLELISM!
vm_engine_iter_task(...);
Py_END_ALLOW_THREADS;    // ◄── Re-acquire GIL

return result;  // cleanup with GIL
```

So the actual computation happens **without the GIL**, even though you called it with GIL!

## Complete Workflow Example

```python
# ============================================================
# Python side (blosc2/lazy.py)
# ============================================================
from blosc2_numexpr_integration import (
    setup_expression, 
    get_chunk_processor_ptr,
    get_cached_handle
)

# Setup
handle = setup_expression("2*a + 3*b*c")
processor_ptr = get_chunk_processor_ptr()

# Pass to C extension
blosc2_extension.evaluate_lazy_expr(processor_ptr, handle, operands)
```

```c
// ============================================================
// C extension side (blosc2_ext.c)
// ============================================================
typedef int (*chunk_processor_t)(
    void*, void*, void*, void*, int64_t, void*
);

PyObject* blosc2_evaluate_lazy_expr(PyObject* self, PyObject* args) {
    PyObject* processor_ptr_obj;
    PyObject* handle_obj;
    
    // Parse arguments
    PyArg_ParseTuple(args, "OO...", &processor_ptr_obj, &handle_obj, ...);
    
    // Extract pointers
    chunk_processor_t processor = (chunk_processor_t)PyLong_AsVoidPtr(processor_ptr_obj);
    void* handle = PyLong_AsVoidPtr(handle_obj);
    
    // Release GIL before launching threads
    Py_BEGIN_ALLOW_THREADS;
    
    // Launch C-Blosc2 threads (they don't have GIL)
    blosc2_set_chunk_processor(processor, handle);
    blosc2_evaluate_chunks(...);
    
    Py_END_ALLOW_THREADS;
    
    Py_RETURN_NONE;
}
```

```c
// ============================================================
// C-Blosc2 worker thread (c-blosc2 internals)
// ============================================================
void blosc2_worker_thread(blosc2_context* ctx) {
    // This thread was created by C-Blosc2, NO GIL
    
    chunk_processor_t processor = ctx->processor;
    void* handle = ctx->numexpr_handle;
    
    // Decompress chunks
    void* chunk_a = blosc2_decompress_chunk(ctx, 0);
    void* chunk_b = blosc2_decompress_chunk(ctx, 1);
    void* chunk_c = blosc2_decompress_chunk(ctx, 2);
    void* output = malloc(chunk_size * sizeof(double));
    
    // Call Cython function (NO GIL needed - it will acquire internally!)
    int status = processor(
        chunk_a, chunk_b, chunk_c,
        output, chunk_size, handle
    );
    
    if (status == 0) {
        blosc2_compress_chunk(output, ...);
    }
    
    free(chunk_a);
    free(chunk_b);
    free(chunk_c);
    free(output);
}
```

```cython
# ============================================================
# Cython wrapper (blosc2_numexpr_integration.pyx)
# ============================================================
cdef int process_chunk_with_numexpr(
    void* chunk_a_data,
    void* chunk_b_data, 
    void* chunk_c_data,
    void* output_data,
    int64_t chunk_size,
    void* numexpr_handle
) noexcept nogil:  # ◄── Called from C thread without GIL
    
    # Acquire GIL temporarily
    with gil:
        # Wrap data as NumPy arrays
        arr_a = PyArray_SimpleNewFromData(1, &chunk_size, NPY_DOUBLE, chunk_a_data)
        arr_b = PyArray_SimpleNewFromData(1, &chunk_size, NPY_DOUBLE, chunk_b_data)
        arr_c = PyArray_SimpleNewFromData(1, &chunk_size, NPY_DOUBLE, chunk_c_data)
        
        # Call NumExpr (it will release GIL during computation!)
        result = numexpr_run_compiled_simple(numexpr_handle, [arr_a, arr_b, arr_c], 3)
        
        # Copy result
        memcpy(output_data, PyArray_DATA(result), chunk_size * sizeof(double))
        
        # Cleanup
        Py_DECREF(arr_a)
        Py_DECREF(arr_b)
        Py_DECREF(arr_c)
        Py_DECREF(result)
    # GIL released automatically
    
    return 0
```

## Parallelism Analysis

When multiple C-Blosc2 threads call the Cython wrapper:

| Thread | State | GIL | NumExpr State |
|--------|-------|-----|---------------|
| Thread 1 | Decompressing | NO | - |
| Thread 2 | Decompressing | NO | - |
| Thread 3 | `with gil:` wrapping arrays | **YES** | - |
| Thread 4 | Waiting for GIL | blocked | - |

After wrapping, NumExpr releases GIL:

| Thread | State | GIL | NumExpr State |
|--------|-------|-----|---------------|
| Thread 1 | Computing | NO | VM running |
| Thread 2 | Computing | NO | VM running |
| Thread 3 | Computing | NO | VM running |
| Thread 4 | `with gil:` wrapping | **YES** | - |

**Result: Real parallelism!** Only brief serialization for array wrapping.

## Setup for Python-Blosc2

```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import numexpr
import os

extensions = [
    Extension(
        "blosc2.blosc2_numexpr_integration",
        sources=["blosc2/blosc2_numexpr_integration.pyx"],
        include_dirs=[
            np.get_include(),
            os.path.dirname(numexpr.__file__),  # For numexpr_capi.h
        ],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="python-blosc2",
    ext_modules=cythonize(extensions, language_level=3),
    # ...
)
```

## Benefits of Cython Approach

1. **Type Safety**: Cython catches type errors at compile time
2. **Easier to Maintain**: More readable than raw C
3. **Automatic GIL Management**: `with gil:` is cleaner than manual PyGILState
4. **NumPy Integration**: Built-in support for NumPy arrays
5. **Error Handling**: Python exceptions work naturally
6. **Performance**: Compiles to C, zero overhead

## Summary

- **`nogil`** function = "can be called without GIL"
- **`with gil:`** block = "acquire GIL temporarily" (uses `PyGILState_Ensure/Release` internally)
- **C-Blosc2 threads** call your `nogil` Cython function
- **Cython function** uses `with gil:` to call NumExpr
- **NumExpr** releases GIL during computation
- **Result**: Real parallelism! ✨

## Files Provided

1. **blosc2_numexpr_integration.pyx** - Cython wrapper (copy to python-blosc2)
2. **blosc2_integration_example.py** - Usage demonstration
3. **CYTHON_INTEGRATION_GUIDE.md** - This file

## Questions?

Check the integration example or see NumExpr's C-API documentation in `C_API.md`.
