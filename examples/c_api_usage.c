/*
 * Example: How Python-Blosc2 (or other C extensions) can use numexpr C API
 * 
 * Compile with:
 *   gcc -I/path/to/python/include -I/path/to/numpy/include \
 *       -I../numexpr c_api_usage.c -lpython3.x -o c_api_usage
 * 
 * Usage:
 *   1. First run Python code to compile expression: ne.validate("2*a + 3*b*c")
 *   2. Then this C code can re-evaluate it efficiently
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include "numexpr_capi.h"
#include <stdio.h>

/*
 * Example 1: Simple re-evaluation
 * This shows the basic pattern for re-evaluating a pre-compiled expression
 */
void example_simple_reevaluate(void)
{
    void *compiled_handle = NULL;
    PyArrayObject *arr_a = NULL, *arr_b = NULL, *arr_c = NULL;
    PyArrayObject *result = NULL;
    npy_intp dims[1] = {5};
    double data_a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double data_b[] = {2.0, 3.0, 4.0, 5.0, 6.0};
    double data_c[] = {3.0, 4.0, 5.0, 6.0, 7.0};
    int i;
    
    printf("Example 1: Simple re-evaluation\n");
    printf("==================================\n\n");
    
    // Get the compiled expression (assumes Python already called ne.validate)
    compiled_handle = numexpr_get_last_compiled();
    if (compiled_handle == NULL) {
        fprintf(stderr, "Error: No compiled expression found.\n");
        fprintf(stderr, "Please run: ne.validate('2*a + 3*b*c') first\n");
        return;
    }
    printf("✓ Got compiled expression handle\n");
    
    // Create NumPy arrays from C data (no copy)
    arr_a = (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data_a);
    arr_b = (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data_b);
    arr_c = (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data_c);
    
    if (arr_a == NULL || arr_b == NULL || arr_c == NULL) {
        fprintf(stderr, "Error: Failed to create NumPy arrays\n");
        goto cleanup;
    }
    printf("✓ Created NumPy array wrappers\n");
    
    // Re-evaluate the expression
    PyArrayObject *arrays[] = {arr_a, arr_b, arr_c};
    result = (PyArrayObject*)numexpr_run_compiled_simple(compiled_handle, arrays, 3);
    
    if (result == NULL) {
        fprintf(stderr, "Error: numexpr_run_compiled_simple failed\n");
        PyErr_Print();
        goto cleanup;
    }
    printf("✓ Re-evaluated expression\n\n");
    
    // Print results
    printf("Results:\n");
    double *result_data = (double*)PyArray_DATA(result);
    for (i = 0; i < 5; i++) {
        printf("  [%d] 2*%.1f + 3*%.1f*%.1f = %.1f\n", 
               i, data_a[i], data_b[i], data_c[i], result_data[i]);
    }
    printf("\n");
    
cleanup:
    Py_XDECREF(arr_a);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_c);
    Py_XDECREF(result);
}

/*
 * Example 2: Processing chunks (like Python-Blosc2 would)
 * This demonstrates the typical use case: evaluating expression on multiple chunks
 */
void example_chunk_processing(void)
{
    void *compiled_handle = NULL;
    int chunk_size = 1000;
    int n_chunks = 10;
    int total_size = chunk_size * n_chunks;
    double *data_a = NULL, *data_b = NULL, *data_c = NULL;
    double *output = NULL;
    npy_intp dims[1];
    int i, chunk;
    
    printf("Example 2: Processing chunks\n");
    printf("==============================\n\n");
    
    // Allocate large data buffers (simulating Blosc2 decompressed chunks)
    data_a = (double*)malloc(total_size * sizeof(double));
    data_b = (double*)malloc(total_size * sizeof(double));
    data_c = (double*)malloc(total_size * sizeof(double));
    output = (double*)malloc(total_size * sizeof(double));
    
    if (!data_a || !data_b || !data_c || !output) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        goto cleanup_mem;
    }
    
    // Fill with test data
    for (i = 0; i < total_size; i++) {
        data_a[i] = (double)i;
        data_b[i] = (double)(i % 100);
        data_c[i] = (double)(i / 100);
    }
    
    // Get compiled expression
    compiled_handle = numexpr_get_last_compiled();
    if (compiled_handle == NULL) {
        fprintf(stderr, "Error: No compiled expression found\n");
        goto cleanup_mem;
    }
    printf("✓ Got compiled expression\n");
    printf("Processing %d chunks of %d elements each...\n", n_chunks, chunk_size);
    
    dims[0] = chunk_size;
    
    // Process each chunk
    for (chunk = 0; chunk < n_chunks; chunk++) {
        PyArrayObject *arr_a, *arr_b, *arr_c;
        PyArrayObject *result;
        double *result_data;
        int offset = chunk * chunk_size;
        
        // Wrap chunk data as NumPy arrays (no copy!)
        arr_a = (PyArrayObject*)PyArray_SimpleNewFromData(
            1, dims, NPY_DOUBLE, &data_a[offset]);
        arr_b = (PyArrayObject*)PyArray_SimpleNewFromData(
            1, dims, NPY_DOUBLE, &data_b[offset]);
        arr_c = (PyArrayObject*)PyArray_SimpleNewFromData(
            1, dims, NPY_DOUBLE, &data_c[offset]);
        
        // Evaluate expression on this chunk
        PyArrayObject *arrays[] = {arr_a, arr_b, arr_c};
        result = (PyArrayObject*)numexpr_run_compiled_simple(
            compiled_handle, arrays, 3);
        
        if (result) {
            // Copy result to output buffer
            result_data = (double*)PyArray_DATA(result);
            memcpy(&output[offset], result_data, chunk_size * sizeof(double));
            Py_DECREF(result);
        }
        
        Py_DECREF(arr_a);
        Py_DECREF(arr_b);
        Py_DECREF(arr_c);
    }
    
    printf("✓ Processed all chunks\n");
    printf("Sample results: [0]=%.1f, [500]=%.1f, [9999]=%.1f\n\n",
           output[0], output[500], output[9999]);
    
cleanup_mem:
    free(data_a);
    free(data_b);
    free(data_c);
    free(output);
}

/*
 * Main function - demonstrates Python-Blosc2 usage pattern
 */
int main(int argc, char *argv[])
{
    printf("NumExpr C API Example\n");
    printf("=====================\n\n");
    printf("This demonstrates how Python-Blosc2 can use numexpr from C.\n\n");
    
    // Initialize Python and NumPy
    Py_Initialize();
    import_array();
    
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Error: Failed to initialize Python\n");
        return 1;
    }
    
    // The expression should be compiled by Python first
    printf("NOTE: Before running this, you should:\n");
    printf("  1. Import numexpr: import numexpr as ne\n");
    printf("  2. Compile expression: ne.validate('2*a + 3*b*c')\n");
    printf("  3. Then this C code can efficiently re-evaluate it\n\n");
    
    // Example 1: Simple case
    example_simple_reevaluate();
    
    // Example 2: Chunk processing (like Blosc2)
    example_chunk_processing();
    
    printf("Summary\n");
    printf("=======\n");
    printf("The C API allows external libraries to:\n");
    printf("  1. Let Python compile complex expressions once\n");
    printf("  2. Re-evaluate efficiently from C on different data chunks\n");
    printf("  3. Avoid Python overhead in tight loops\n");
    printf("  4. Perfect for Blosc2's lazy evaluation use case!\n\n");
    
    // Cleanup
    Py_Finalize();
    
    return 0;
}
