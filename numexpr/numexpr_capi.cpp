/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
  
  C API Implementation - Functions to call NumExpr from external C code
**********************************************************************/

#include "numexpr_capi.h"
#include "numexpr_object.hpp"
#include "interpreter.hpp"

/*
 * Get the last compiled NumExpr expression from Python's thread-local storage.
 * This accesses the _numexpr_last.l['ex'] variable from the necompiler module.
 */
void* numexpr_get_last_compiled(void)
{
    PyObject *necompiler_module = NULL;
    PyObject *numexpr_last = NULL;
    PyObject *local_dict = NULL;
    PyObject *compiled_ex = NULL;
    void *handle = NULL;
    
    // Import numexpr.necompiler module
    necompiler_module = PyImport_ImportModule("numexpr.necompiler");
    if (necompiler_module == NULL) {
        return NULL;
    }
    
    // Get _numexpr_last thread-local object
    numexpr_last = PyObject_GetAttrString(necompiler_module, "_numexpr_last");
    Py_DECREF(necompiler_module);
    if (numexpr_last == NULL) {
        return NULL;
    }
    
    // Get the 'l' attribute (ContextDict)
    local_dict = PyObject_GetAttrString(numexpr_last, "l");
    Py_DECREF(numexpr_last);
    if (local_dict == NULL) {
        PyErr_Clear();  // No compiled expression yet
        return NULL;
    }
    
    // Get the 'ex' key (compiled expression)
    compiled_ex = PyObject_GetItem(local_dict, PyUnicode_FromString("ex"));
    Py_DECREF(local_dict);
    if (compiled_ex == NULL) {
        PyErr_Clear();  // No compiled expression
        return NULL;
    }
    
    // Return as opaque handle (borrowed reference kept alive by cache)
    handle = (void*)compiled_ex;
    Py_DECREF(compiled_ex);  // Safe because it's cached in _numexpr_last.l
    
    return handle;
}

/*
 * Execute a compiled NumExpr expression with new input arrays.
 */
PyObject* numexpr_run_compiled(void *handle,
                               PyArrayObject **input_arrays,
                               int n_arrays,
                               PyArrayObject *out,
                               char order,
                               const char *casting)
{
    NumExprObject *compiled = (NumExprObject *)handle;
    PyObject *args = NULL;
    PyObject *kwargs = NULL;
    PyObject *result = NULL;
    int i;
    
    if (compiled == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid compiled expression handle");
        return NULL;
    }
    
    // Check that we have a NumExpr object
    if (!PyObject_TypeCheck(compiled, &NumExprType)) {
        PyErr_SetString(PyExc_TypeError, "Handle is not a NumExpr object");
        return NULL;
    }
    
    // Verify number of inputs matches signature
    if (PyBytes_Size(compiled->signature) != n_arrays) {
        PyErr_Format(PyExc_ValueError,
                    "Number of input arrays (%d) doesn't match expression signature (%zd)",
                    n_arrays, PyBytes_Size(compiled->signature));
        return NULL;
    }
    
    // Build args tuple
    args = PyTuple_New(n_arrays);
    if (args == NULL) {
        return NULL;
    }
    
    for (i = 0; i < n_arrays; i++) {
        Py_INCREF(input_arrays[i]);
        PyTuple_SET_ITEM(args, i, (PyObject*)input_arrays[i]);
    }
    
    // Build kwargs dict
    kwargs = PyDict_New();
    if (kwargs == NULL) {
        Py_DECREF(args);
        return NULL;
    }
    
    // Add 'out' parameter if provided
    if (out != NULL) {
        PyDict_SetItemString(kwargs, "out", (PyObject*)out);
    } else {
        PyDict_SetItemString(kwargs, "out", Py_None);
    }
    
    // Add 'order' parameter
    PyObject *order_obj = PyUnicode_FromFormat("%c", order);
    PyDict_SetItemString(kwargs, "order", order_obj);
    Py_DECREF(order_obj);
    
    // Add 'casting' parameter
    if (casting != NULL) {
        PyObject *casting_obj = PyUnicode_FromString(casting);
        PyDict_SetItemString(kwargs, "casting", casting_obj);
        Py_DECREF(casting_obj);
    } else {
        PyObject *casting_obj = PyUnicode_FromString("safe");
        PyDict_SetItemString(kwargs, "casting", casting_obj);
        Py_DECREF(casting_obj);
    }
    
    // Add ex_uses_vml parameter (required by NumExpr_run)
    // We need to get this from the cached kwargs
    PyObject *necompiler_module = PyImport_ImportModule("numexpr.necompiler");
    if (necompiler_module != NULL) {
        PyObject *numexpr_last = PyObject_GetAttrString(necompiler_module, "_numexpr_last");
        if (numexpr_last != NULL) {
            PyObject *local_dict = PyObject_GetAttrString(numexpr_last, "l");
            if (local_dict != NULL) {
                PyObject *cached_kwargs = PyObject_GetItem(local_dict, PyUnicode_FromString("kwargs"));
                if (cached_kwargs != NULL) {
                    PyObject *ex_uses_vml = PyDict_GetItemString(cached_kwargs, "ex_uses_vml");
                    if (ex_uses_vml != NULL) {
                        PyDict_SetItemString(kwargs, "ex_uses_vml", ex_uses_vml);
                    } else {
                        // Default to False if not found
                        PyDict_SetItemString(kwargs, "ex_uses_vml", Py_False);
                    }
                    Py_DECREF(cached_kwargs);
                } else {
                    PyErr_Clear();
                    PyDict_SetItemString(kwargs, "ex_uses_vml", Py_False);
                }
                Py_DECREF(local_dict);
            } else {
                PyErr_Clear();
                PyDict_SetItemString(kwargs, "ex_uses_vml", Py_False);
            }
            Py_DECREF(numexpr_last);
        } else {
            PyErr_Clear();
            PyDict_SetItemString(kwargs, "ex_uses_vml", Py_False);
        }
        Py_DECREF(necompiler_module);
    } else {
        PyErr_Clear();
        PyDict_SetItemString(kwargs, "ex_uses_vml", Py_False);
    }
    
    // Call NumExpr_run
    result = NumExpr_run(compiled, args, kwargs);
    
    Py_DECREF(args);
    Py_DECREF(kwargs);
    
    return result;
}

/*
 * Simplified version with default parameters.
 */
PyObject* numexpr_run_compiled_simple(void *handle,
                                      PyArrayObject **input_arrays,
                                      int n_arrays)
{
    return numexpr_run_compiled(handle, input_arrays, n_arrays, 
                               NULL, 'K', "safe");
}
