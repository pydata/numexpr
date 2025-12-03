"""
Example demonstrating how Python-Blosc2 (or other packages) can use numexpr C API.

This shows the workflow:
1. Python side: Compile expression with ne.validate()
2. C side: Get compiled expression and re-evaluate with new data

This file simulates what Python-Blosc2's C extension would do.
"""

import numpy as np
import numexpr as ne
import timeit

def example_usage():
    """Example showing how to use the C API from Python (simulating what Blosc2 would do)"""
    
    # Step 1: Compile expression in Python
    print("Step 1: Compiling expression in Python...")
    a = np.array([1., 2., 3., 4., 5.])
    b = np.array([2., 3., 4., 5., 6.])
    c = np.array([3., 4., 5., 6., 7.])
    
    # This compiles and caches the expression
    result1 = ne.evaluate("2*a + 3*b*c")
    print(f"Initial evaluation result: {result1}")
    
    # Step 2: Re-evaluate with Python's re_evaluate (this is what we're replacing in C)
    print("\nStep 2: Re-evaluating with Python's re_evaluate()...")
    a2 = np.array([10., 20., 30., 40., 50.])
    b2 = np.array([1., 2., 3., 4., 5.])
    c2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    result2 = ne.re_evaluate(local_dict={'a': a2, 'b': b2, 'c': c2})
    print(f"Re-evaluation result: {result2}")
    
    # Step 3: What the C API would do (from Python-Blosc2's C extension)
    print("\nStep 3: This is what Python-Blosc2's C extension would do:")
    print("  - Call numexpr_get_last_compiled() to get the compiled expression")
    print("  - In a loop over chunks:")
    print("    - Create PyArrayObject* for each chunk")
    print("    - Call numexpr_run_compiled(handle, arrays, n_arrays, NULL, 'K', 'safe')")
    print("    - Process the result")
    
    # Verify it still works
    result3 = ne.re_evaluate(local_dict={'a': a, 'b': b, 'c': c})
    print(f"\nVerification: {result3}")
    assert np.array_equal(result1, result3), "Results should match!"
    
    print("\n✓ Example completed successfully!")
    print("\nNote: The actual C API calls would be made from a C extension like:")
    print("""
    // In Python-Blosc2's C extension:
    void* handle = numexpr_get_last_compiled();
    
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        // Wrap chunk data as PyArrayObject
        PyArrayObject* arrays[3] = {arr_a, arr_b, arr_c};
        
        // Call numexpr - this is the fast re_evaluate!
        PyObject* result = numexpr_run_compiled_simple(handle, arrays, 3);
        
        // Use result...
        Py_DECREF(result);
    }
    """)

def benchmark_comparison():
    """Compare re_evaluate performance - useful for testing"""
    
    # Large arrays
    size = 10**6
    a = np.random.rand(size)
    b = np.random.rand(size)
    c = np.random.rand(size)
    
    # Initial compile
    ne.evaluate("2*a + 3*b*c")
    
    # Time re_evaluate with proper local_dict
    n_iterations = 100
    local_dict = {'a': a, 'b': b, 'c': c}
    time_taken = timeit.timeit(
        lambda: ne.re_evaluate(local_dict=local_dict),
        number=n_iterations
    )
    
    print(f"\nBenchmark: {n_iterations} re_evaluate calls on {size} elements")
    print(f"Time: {time_taken:.4f} seconds")
    print(f"Per call: {time_taken/n_iterations*1000:.4f} ms")
    print(f"Throughput: {size*n_iterations/time_taken/1e6:.2f} M elements/sec")

if __name__ == "__main__":
    example_usage()
    benchmark_comparison()
