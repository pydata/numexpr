# cython: language_level=3
"""
Cython wrapper for NumExpr C-API integration with Python-Blosc2

This module demonstrates how to call NumExpr C-API from Cython code
that will be invoked by C-Blosc2 threads.
"""

cimport numpy as cnp
import numpy as np
from libc.stdint cimport int64_t, uint8_t
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.exc cimport PyErr_SetString

# Import NumExpr C-API declarations
cdef extern from "numexpr_capi.h":
    void* numexpr_get_last_compiled() nogil
    PyObject* numexpr_run_compiled_simple(void* handle, 
                                         cnp.PyArrayObject** arrays, 
                                         int n_arrays) nogil

# Import Python C-API exception types
cdef extern from "Python.h":
    PyObject* PyExc_MemoryError


# Cython wrapper function that receives C array of C pointers
# This is the function that C-Blosc2 threads will call
cdef public api PyObject* process_chunk_with_numexpr(
    void** chunk_pointers,
    int n_arrays,
    int chunk_size,
    int dtype,
    void* numexpr_handle
) noexcept nogil:
    """
    Process a chunk using NumExpr from pure C code.
    
    This function receives a C array of pointers (from decompressed chunks),
    wraps them as NumPy arrays, evaluates the NumExpr expression, and returns
    the result.
    
    This function is exposed to C via the public API and can be called directly
    from C code without any Python knowledge. The function handles GIL
    acquisition/release internally, so the caller does NOT need to hold the GIL.
    
    Parameters
    ----------
    chunk_pointers : void**
        C array of pointers to decompressed chunk data
        Example: void* arr[3] = {chunk_a, chunk_b, chunk_c}
    n_arrays : int
        Number of input arrays (length of chunk_pointers array)
    chunk_size : int
        Number of elements in each chunk
    dtype : int
        NumPy dtype number (e.g., NPY_DOUBLE = 12, NPY_FLOAT = 11)
    numexpr_handle : void*
        Handle to compiled NumExpr expression
        
    Returns
    -------
    PyObject* : Result array from NumExpr evaluation (new reference)
                Returns NULL on error (with Python exception set)
                
    Notes
    -----
    - This function can be called from C threads that do NOT hold the GIL
    - GIL is acquired internally and released before returning
    - NumExpr releases the GIL during computation for parallelism
    """
    cdef:
        cnp.PyArrayObject** arrays_ptr = NULL
        cnp.PyArrayObject* arr
        PyObject* result = NULL
        cnp.npy_intp dims[1]
        int i
    
    # Acquire GIL - we need it for Python/NumPy operations
    with gil:
        dims[0] = chunk_size
        
        # Allocate array of PyArrayObject pointers
        arrays_ptr = <cnp.PyArrayObject**>PyMem_Malloc(n_arrays * sizeof(cnp.PyArrayObject*))
        if arrays_ptr == NULL:
            PyErr_SetString(PyExc_MemoryError, "Could not allocate array pointers")
            return NULL
        
        # Wrap each C pointer as a NumPy array (zero-copy!)
        for i in range(n_arrays):
            arr = <cnp.PyArrayObject*>cnp.PyArray_SimpleNewFromData(
                1, dims, dtype, chunk_pointers[i]
            )
            if arr == NULL:
                # Cleanup any arrays created so far
                for j in range(i):
                    cnp.Py_DECREF(<object>arrays_ptr[j])
                PyMem_Free(arrays_ptr)
                return NULL
            
            arrays_ptr[i] = arr
        
        # Call NumExpr C-API (this will release GIL internally!)
        result = numexpr_run_compiled_simple(numexpr_handle, arrays_ptr, n_arrays)
        
        # Cleanup wrapped arrays (they were just temporary wrappers)
        for i in range(n_arrays):
            cnp.Py_DECREF(<object>arrays_ptr[i])
        
        PyMem_Free(arrays_ptr)
    
    # GIL is automatically released here when exiting 'with gil:' block
    # Return result (or NULL if NumExpr failed)
    return result


# Python-accessible wrapper for setup
cdef void* cached_numexpr_handle = NULL

def setup_expression(str expression, dict dummy_vars=None):
    """
    Compile a NumExpr expression and cache it for C-level evaluation.
    
    This should be called from Python before launching C-Blosc2 operations.
    
    Parameters
    ----------
    expression : str
        NumExpr expression string (e.g., "2*a + 3*b*c")
    dummy_vars : dict, optional
        Dictionary of dummy arrays for type inference
        If not provided, defaults to float64 scalars
        
    Returns
    -------
    int : Handle pointer (as Python int) for passing to C code
    """
    global cached_numexpr_handle
    
    import numexpr as ne
    
    # Create dummy arrays if not provided
    if dummy_vars is None:
        dummy_vars = {
            'a': np.array([0.0], dtype=np.float64),
            'b': np.array([0.0], dtype=np.float64),
            'c': np.array([0.0], dtype=np.float64),
        }
    
    # Compile expression
    ne.validate(expression, local_dict=dummy_vars)
    
    # Get handle
    with nogil:
        cached_numexpr_handle = numexpr_get_last_compiled()
    
    if cached_numexpr_handle == NULL:
        raise RuntimeError("Failed to get compiled NumExpr expression")
    
    return <size_t>cached_numexpr_handle


def get_cached_handle():
    """Get the cached NumExpr handle as a Python int."""
    return <size_t>cached_numexpr_handle


# Example: High-level Python interface for testing
def evaluate_chunks_parallel(
    cnp.ndarray[double, ndim=1] array_a,
    cnp.ndarray[double, ndim=1] array_b,
    cnp.ndarray[double, ndim=1] array_c,
    int64_t chunk_size,
    void* numexpr_handle = NULL
):
    """
    Evaluate expression on arrays split into chunks (demonstration).
    
    This demonstrates how C-Blosc2 would call process_chunk_with_numexpr.
    In reality, C-Blosc2 would have pointers to decompressed data.
    
    Parameters
    ----------
    array_a, array_b, array_c : ndarray
        Input arrays
    chunk_size : int
        Size of each chunk
    numexpr_handle : void*, optional
        Handle to compiled expression. If NULL, uses cached handle.
        
    Returns
    -------
    ndarray : Result array
    """
    global cached_numexpr_handle
    
    if numexpr_handle == NULL:
        numexpr_handle = cached_numexpr_handle
        
    if numexpr_handle == NULL:
        raise RuntimeError("No NumExpr expression compiled. Call setup_expression() first.")
    
    cdef:
        int64_t total_size = len(array_a)
        int64_t n_chunks = (total_size + chunk_size - 1) // chunk_size
        cnp.ndarray[double, ndim=1] result = np.empty(total_size, dtype=np.float64)
        int64_t chunk_idx, start, end, actual_chunk_size
        void* chunk_ptrs[3]
        PyObject* chunk_result
    
    # Process each chunk
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_size)
        actual_chunk_size = end - start
        
        # Create C array of pointers (simulating what C-Blosc2 would provide)
        chunk_ptrs[0] = <void*>&array_a[start]
        chunk_ptrs[1] = <void*>&array_b[start]
        chunk_ptrs[2] = <void*>&array_c[start]
        
        # Call the C-level function (NO GIL needed - function handles it!)
        chunk_result = process_chunk_with_numexpr(
            chunk_ptrs,
            3,
            actual_chunk_size,
            cnp.NPY_DOUBLE,
            numexpr_handle
        )
        
        if chunk_result == NULL:
            raise RuntimeError(f"Error processing chunk {chunk_idx}")
        
        # Copy result
        result[start:end] = <object>chunk_result
        cnp.Py_DECREF(chunk_result)
    
    return result
