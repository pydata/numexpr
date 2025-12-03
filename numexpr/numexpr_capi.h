#ifndef NUMEXPR_CAPI_H
#define NUMEXPR_CAPI_H

/*********************************************************************
  NumExpr C API - For calling NumExpr from external C libraries
  
  This header provides a minimal C API to use NumExpr from C code.
  
  Basic workflow:
  1. Python side: Compile expression with ne.validate("2*a + 3*b*c")
  2. C side: Get compiled expression with numexpr_get_last_compiled()
  3. C side: Re-evaluate with numexpr_run_compiled() on new data chunks
  
  Requirements:
  - Must link against Python and NumPy
  - Must initialize Python before calling (Py_Initialize)
  - Must hold GIL when calling these functions
  
  Example usage from another C extension (e.g., Python-Blosc2):
  
    // After Python has called ne.validate("expression")
    void *handle = numexpr_get_last_compiled();
    
    // In a loop processing chunks:
    PyArrayObject *arrays[] = {arr_a, arr_b, arr_c};
    PyObject *result = numexpr_run_compiled(handle, arrays, 3, NULL);
    
**********************************************************************/

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Get the last compiled NumExpr expression.
 * 
 * This retrieves the expression that was last compiled via Python's
 * ne.evaluate() or ne.validate() call from the thread-local cache.
 * 
 * Returns: Opaque handle to compiled expression, or NULL if none available
 * 
 * Note: Caller must hold the GIL. The returned handle is a borrowed
 * reference and remains valid as long as the expression stays cached.
 */
void* numexpr_get_last_compiled(void);

/*
 * Execute a compiled NumExpr expression with new input arrays.
 * 
 * This is the C equivalent of Python's re_evaluate() function.
 * 
 * Parameters:
 *   handle - Opaque handle from numexpr_get_last_compiled()
 *   input_arrays - Array of PyArrayObject pointers (inputs to expression)
 *   n_arrays - Number of input arrays
 *   out - Optional output array (NULL to allocate new)
 *   order - Memory order: 'K' (keep), 'C', 'F', 'A' (default: 'K')
 *   casting - Casting mode: "safe", "same_kind", "unsafe" (default: "safe")
 * 
 * Returns: New reference to result PyArrayObject, or NULL on error
 * 
 * Note: Caller must hold the GIL and manage returned reference with Py_DECREF
 */
PyObject* numexpr_run_compiled(void *handle,
                               PyArrayObject **input_arrays,
                               int n_arrays,
                               PyArrayObject *out,
                               char order,
                               const char *casting);

/*
 * Simplified version of numexpr_run_compiled with default parameters.
 * Uses order='K' and casting='safe'.
 */
PyObject* numexpr_run_compiled_simple(void *handle,
                                      PyArrayObject **input_arrays,
                                      int n_arrays);

#ifdef __cplusplus
}
#endif

#endif /* NUMEXPR_CAPI_H */
