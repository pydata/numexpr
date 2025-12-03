# NumExpr C API Documentation

## Overview

NumExpr 2.14.2+ provides a C API that allows external C extensions to efficiently re-evaluate compiled expressions. This is particularly useful for libraries like Python-Blosc2 that need to apply expressions to multiple data chunks with minimal Python overhead.

## Key Concept

The C API separates expression compilation (done in Python) from execution (can be done in C):

1. **Python side**: Compile complex expression once using `ne.validate()` or `ne.evaluate()`
2. **C side**: Re-evaluate efficiently on different data chunks using C API functions

This avoids repeated Python overhead (dictionary lookups, argument parsing, etc.) when processing many chunks.

## API Reference

### Header File

Include in your C extension:

```c
#include "numexpr_capi.h"
```

### Functions

#### `void* numexpr_get_last_compiled(void)`

Retrieves the last compiled NumExpr expression from the thread-local cache.

**Returns:**
- Opaque handle to compiled expression
- `NULL` if no expression has been compiled yet

**Thread Safety:** Safe (uses thread-local storage)

**Requirements:**
- Must hold Python GIL
- Must call after Python has executed `ne.validate()` or `ne.evaluate()`

**Example:**
```c
PyGILState_STATE gstate = PyGILState_Ensure();
void *handle = numexpr_get_last_compiled();
if (handle == NULL) {
    fprintf(stderr, "No compiled expression available\n");
}
PyGILState_Release(gstate);
```

---

#### `PyObject* numexpr_run_compiled_simple(void *handle, PyArrayObject **arrays, int n_arrays)`

Re-evaluates a compiled expression with new input arrays using default settings.

**Parameters:**
- `handle`: Handle from `numexpr_get_last_compiled()`
- `arrays`: Array of `PyArrayObject*` pointers (input data)
- `n_arrays`: Number of input arrays (must match expression signature)

**Returns:**
- New reference to result `PyArrayObject`
- `NULL` on error (check with `PyErr_Occurred()`)

**Defaults:**
- Memory order: 'K' (keep)
- Casting: "safe"
- Output: Newly allocated array

**Thread Safety:** Must hold GIL

**Example:**
```c
PyGILState_STATE gstate = PyGILState_Ensure();

// Create array wrappers (no copy)
npy_intp dims[1] = {chunk_size};
PyArrayObject *arr_a = (PyArrayObject*)PyArray_SimpleNewFromData(
    1, dims, NPY_DOUBLE, chunk_data_a);
PyArrayObject *arr_b = (PyArrayObject*)PyArray_SimpleNewFromData(
    1, dims, NPY_DOUBLE, chunk_data_b);
PyArrayObject *arr_c = (PyArrayObject*)PyArray_SimpleNewFromData(
    1, dims, NPY_DOUBLE, chunk_data_c);

// Evaluate
PyArrayObject *arrays[] = {arr_a, arr_b, arr_c};
PyObject *result = numexpr_run_compiled_simple(handle, arrays, 3);

if (result) {
    double *data = (double*)PyArray_DATA((PyArrayObject*)result);
    // ... use data ...
    Py_DECREF(result);
}

Py_DECREF(arr_a);
Py_DECREF(arr_b);
Py_DECREF(arr_c);

PyGILState_Release(gstate);
```

---

#### `PyObject* numexpr_run_compiled(void *handle, PyArrayObject **arrays, int n_arrays, PyArrayObject *out, char order, const char *casting)`

Advanced version with full control over execution parameters.

**Parameters:**
- `handle`: Handle from `numexpr_get_last_compiled()`
- `arrays`: Array of `PyArrayObject*` pointers
- `n_arrays`: Number of input arrays
- `out`: Pre-allocated output array, or `NULL` to allocate new
- `order`: Memory order - 'K' (keep), 'C' (C-order), 'F' (Fortran), 'A' (auto)
- `casting`: Casting mode - "safe", "same_kind", "unsafe"

**Returns:**
- New reference to result `PyArrayObject` (may be `out` if provided)
- `NULL` on error

**Example:**
```c
// Re-use output buffer to avoid allocations
npy_intp dims[1] = {chunk_size};
PyArrayObject *output = (PyArrayObject*)PyArray_SimpleNew(
    1, dims, NPY_DOUBLE);

for (int i = 0; i < n_chunks; i++) {
    // ... create arrays for chunk i ...
    
    PyObject *result = numexpr_run_compiled(
        handle, arrays, 3, 
        output,        // reuse buffer
        'K',           // keep memory order
        "same_kind"    // allow same-kind casts
    );
    
    // result == output (same object)
    // ... process output ...
}

Py_DECREF(output);
```

## Integration Guide for Python-Blosc2

### Python Side Setup

```python
import numexpr as ne
import numpy as np

def setup_expression(expr_string):
    """Compile expression for later C-level evaluation"""
    # Use dummy arrays just for type inference
    a = np.array([0.0], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    c = np.array([0.0], dtype=np.float64)
    
    # Compile and cache
    ne.validate(expr_string, local_dict={'a': a, 'b': b, 'c': c})
    
    print(f"Expression '{expr_string}' compiled and ready for C evaluation")
```

### C Extension Integration

**In your setup.py:**

```python
import numexpr
import os

numexpr_include = os.path.dirname(numexpr.__file__)

extension = Extension(
    'blosc2._blosc2',
    sources=['blosc2/blosc2_ext.c'],
    include_dirs=[
        np.get_include(),
        numexpr_include,  # Add NumExpr headers
    ],
    # ...
)
```

**In your C extension:**

```c
#include <Python.h>
#include <numpy/arrayobject.h>
#include "numexpr_capi.h"

static void* cached_numexpr_handle = NULL;

PyObject* blosc2_evaluate_chunks(PyObject *self, PyObject *args)
{
    // ... parse arguments to get chunk data ...
    
    PyGILState_STATE gstate;
    
    // Get expression handle (once per expression)
    if (cached_numexpr_handle == NULL) {
        gstate = PyGILState_Ensure();
        cached_numexpr_handle = numexpr_get_last_compiled();
        PyGILState_Release(gstate);
        
        if (cached_numexpr_handle == NULL) {
            PyErr_SetString(PyExc_RuntimeError, 
                "No NumExpr expression compiled. Call ne.validate() first.");
            return NULL;
        }
    }
    
    // Process chunks
    for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++) {
        // 1. Decompress chunk from Blosc2
        void *chunk_a = blosc2_decompress_chunk(array, chunk_idx, 0);
        void *chunk_b = blosc2_decompress_chunk(array, chunk_idx, 1);
        void *chunk_c = blosc2_decompress_chunk(array, chunk_idx, 2);
        
        // 2. Wrap as NumPy arrays (zero-copy)
        gstate = PyGILState_Ensure();
        
        npy_intp dims[1] = {chunk_size};
        PyArrayObject *arr_a = (PyArrayObject*)
            PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, chunk_a);
        PyArrayObject *arr_b = (PyArrayObject*)
            PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, chunk_b);
        PyArrayObject *arr_c = (PyArrayObject*)
            PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, chunk_c);
        
        // 3. Evaluate expression (FAST!)
        PyArrayObject *input_arrays[] = {arr_a, arr_b, arr_c};
        PyObject *result = numexpr_run_compiled_simple(
            cached_numexpr_handle, input_arrays, 3);
        
        if (result) {
            // 4. Store result back to Blosc2
            double *result_data = (double*)PyArray_DATA((PyArrayObject*)result);
            blosc2_compress_chunk(output_array, chunk_idx, result_data, chunk_size);
            Py_DECREF(result);
        } else {
            // Handle error
            Py_DECREF(arr_a);
            Py_DECREF(arr_b);
            Py_DECREF(arr_c);
            PyGILState_Release(gstate);
            return NULL;
        }
        
        Py_DECREF(arr_a);
        Py_DECREF(arr_b);
        Py_DECREF(arr_c);
        
        PyGILState_Release(gstate);
    }
    
    Py_RETURN_NONE;
}
```

## Performance Considerations

### Expected Speedup

For small to medium chunks (1K-100K elements):
- **Without C API**: Python overhead dominates (dict creation, arg parsing)
- **With C API**: Pure C function call overhead
- **Speedup**: 2-5x depending on expression complexity

For large chunks (>1M elements):
- **Speedup**: 1.1-1.5x (computation dominates, not overhead)

### Best Practices

1. **Compile once, reuse many times**
   ```python
   ne.validate(expr)  # Once in Python
   # ... then many C-level re_evaluate calls
   ```

2. **Zero-copy array wrapping**
   ```c
   // Good: wrap existing data
   PyArray_SimpleNewFromData(1, dims, dtype, existing_data);
   
   // Bad: would copy data
   PyArray_FROM_OTF(py_obj, dtype, flags);
   ```

3. **Reuse output buffers** for multiple chunks
   ```c
   PyArrayObject *output = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
   for (each chunk) {
       numexpr_run_compiled(..., output, ...);  // reuse
   }
   ```

4. **Release GIL between chunks** if doing CPU-heavy work
   ```c
   for (chunk in chunks) {
       // Acquire GIL
       PyGILState_STATE gstate = PyGILState_Ensure();
       result = numexpr_run_compiled_simple(...);
       PyGILState_Release(gstate);
       // Release GIL
       
       // Do non-Python work here
       compress_result(result_data);
   }
   ```

## Error Handling

Always check return values:

```c
void *handle = numexpr_get_last_compiled();
if (handle == NULL) {
    // No expression compiled
    PyErr_SetString(PyExc_RuntimeError, 
        "Call ne.validate() before using C API");
    return NULL;
}

PyObject *result = numexpr_run_compiled_simple(handle, arrays, n);
if (result == NULL) {
    // NumExpr error occurred
    PyErr_Print();  // or handle appropriately
    return NULL;
}
```

## Thread Safety

- **Thread-local storage**: Each thread has its own last compiled expression
- **GIL required**: All C API functions must be called with GIL held
- **Multiple threads**: Can use different expressions simultaneously
- **Same expression**: Multiple threads can use same expression (NumExpr handles threading)

## Limitations

1. **Expression must be pre-compiled** in Python
2. **Array types must match** signature from compilation
3. **Number of inputs must match** expression
4. **GIL required** for all calls
5. **Handle lifetime**: Valid as long as expression stays cached

## Examples

See `examples/` directory:
- `c_api_example.py` - Python demonstration
- `c_api_usage.c` - Complete C example
- `C_API_README.md` - Quick start guide

## Changelog

### Version 2.14.2
- Initial C API release
- Added `numexpr_get_last_compiled()`
- Added `numexpr_run_compiled()`
- Added `numexpr_run_compiled_simple()`

## Support

For issues or questions about the C API:
1. Check examples in `examples/` directory
2. See main NumExpr documentation
3. Open an issue on GitHub: https://github.com/pydata/numexpr
