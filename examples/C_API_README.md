# NumExpr C API Usage Guide

This directory contains examples of how to use NumExpr's C API from external C extensions, particularly designed for Python-Blosc2.

## Overview

The NumExpr C API allows external C libraries to:
1. Compile expressions once in Python
2. Efficiently re-evaluate them from C code on different data chunks
3. Avoid Python overhead in performance-critical loops

This is perfect for Python-Blosc2's lazy evaluation use case where you:
- Decompress chunks of data in C
- Apply NumExpr expressions to chunks
- Minimize Python/C transitions

## API Functions

### `void* numexpr_get_last_compiled(void)`

Retrieves the last compiled NumExpr expression from Python's cache.

**Returns:** Opaque handle to compiled expression, or NULL if none available

**Note:** Must be called after `ne.validate()` or `ne.evaluate()` in Python

### `PyObject* numexpr_run_compiled_simple(void *handle, PyArrayObject **arrays, int n_arrays)`

Re-evaluates a compiled expression with new input arrays.

**Parameters:**
- `handle` - Handle from `numexpr_get_last_compiled()`
- `arrays` - Array of PyArrayObject pointers (inputs)
- `n_arrays` - Number of input arrays

**Returns:** New reference to result PyArrayObject, or NULL on error

### `PyObject* numexpr_run_compiled(void *handle, PyArrayObject **arrays, int n_arrays, PyArrayObject *out, char order, const char *casting)`

Advanced version with full control over output and casting.

**Parameters:**
- `handle` - Handle from `numexpr_get_last_compiled()`
- `arrays` - Array of PyArrayObject pointers
- `n_arrays` - Number of input arrays
- `out` - Optional pre-allocated output array (NULL to allocate new)
- `order` - Memory order: 'K' (keep), 'C', 'F', 'A'
- `casting` - Casting mode: "safe", "same_kind", "unsafe"

**Returns:** New reference to result PyArrayObject

## Usage Pattern for Python-Blosc2

### Step 1: Python Side (One-time Setup)

```python
import numexpr as ne
import numpy as np

# Compile and cache the expression
a = np.array([1., 2., 3.])  # dummy arrays for compilation
b = np.array([4., 5., 6.])
c = np.array([7., 8., 9.])

# This compiles and caches the expression
ne.validate("2*a + 3*b*c")

# Now the C extension can use it...
```

### Step 2: C Extension Side (Fast Loop)

```c
#include <Python.h>
#include <numpy/arrayobject.h>
#include "numexpr_capi.h"

void blosc2_lazy_eval_chunks(void *blosc_array, /* ... */)
{
    void *numexpr_handle;
    PyGILState_STATE gstate;
    
    // Get compiled expression (once)
    gstate = PyGILState_Ensure();
    numexpr_handle = numexpr_get_last_compiled();
    PyGILState_Release(gstate);
    
    // Process each chunk
    for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++) {
        // Decompress chunk from Blosc2 (your existing code)
        double *chunk_a = blosc2_get_chunk(blosc_array, chunk_idx, 0);
        double *chunk_b = blosc2_get_chunk(blosc_array, chunk_idx, 1);
        double *chunk_c = blosc2_get_chunk(blosc_array, chunk_idx, 2);
        
        // Wrap as NumPy arrays (no copy!)
        gstate = PyGILState_Ensure();
        
        npy_intp dims[1] = {chunk_size};
        PyArrayObject *arr_a = (PyArrayObject*)PyArray_SimpleNewFromData(
            1, dims, NPY_DOUBLE, chunk_a);
        PyArrayObject *arr_b = (PyArrayObject*)PyArray_SimpleNewFromData(
            1, dims, NPY_DOUBLE, chunk_b);
        PyArrayObject *arr_c = (PyArrayObject*)PyArray_SimpleNewFromData(
            1, dims, NPY_DOUBLE, chunk_c);
        
        // Evaluate NumExpr on this chunk (FAST!)
        PyArrayObject *arrays[] = {arr_a, arr_b, arr_c};
        PyObject *result = numexpr_run_compiled_simple(
            numexpr_handle, arrays, 3);
        
        // Use result (e.g., compress back with Blosc2)
        if (result) {
            double *result_data = (double*)PyArray_DATA((PyArrayObject*)result);
            // ... store or process result_data ...
            Py_DECREF(result);
        }
        
        Py_DECREF(arr_a);
        Py_DECREF(arr_b);
        Py_DECREF(arr_c);
        
        PyGILState_Release(gstate);
    }
}
```

## Building External Extensions

To use NumExpr C API in your extension (e.g., Python-Blosc2):

### setup.py

```python
from setuptools import Extension, setup
import numpy as np

# Find numexpr include directory
import numexpr
numexpr_include = os.path.join(
    os.path.dirname(numexpr.__file__))

extension = Extension(
    'blosc2._blosc2',
    sources=['blosc2/blosc2_ext.c'],
    include_dirs=[
        np.get_include(),
        numexpr_include,  # Add this!
    ],
    # ... other settings ...
)
```

### Your C Extension

```c
#include <Python.h>
#include <numpy/arrayobject.h>
#include "numexpr_capi.h"  // NumExpr C API

// Your Blosc2 code can now call numexpr functions!
```

## Examples

- `c_api_example.py` - Python demonstration of the workflow
- `c_api_usage.c` - Complete C example showing chunk processing

## Performance Benefits

For Python-Blosc2 processing 1000 chunks:

**Without C API:**
```python
for chunk in chunks:
    result = ne.re_evaluate(local_dict={'a': chunk_a, 'b': chunk_b, 'c': chunk_c})
    # Python overhead: dict creation, argument parsing, etc. × 1000
```

**With C API:**
```c
for (chunk...) {
    result = numexpr_run_compiled_simple(handle, arrays, 3);
    // Pure C call, minimal overhead!
}
```

Expected speedup: 2-5x for small chunks, depending on expression complexity.

## Thread Safety

- `numexpr_get_last_compiled()` accesses thread-local storage - safe per thread
- `numexpr_run_compiled()` must be called with GIL held
- Multiple threads can use different expressions simultaneously
- Same expression can be used by multiple threads (NumExpr handles threading internally)

## Requirements

- NumExpr must be installed and initialized
- Python GIL must be held when calling API functions
- NumPy must be imported (`import_array()` called)

## Questions?

See the main NumExpr documentation or the example files in this directory.
