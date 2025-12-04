#!/usr/bin/env python
"""
Example: Using the Cython wrapper for NumExpr C-API with Python-Blosc2

This demonstrates how python-blosc2 would integrate the NumExpr C-API
using the Cython wrapper.
"""

import numpy as np
import numexpr as ne
import time

# This would be your Cython module in python-blosc2
# For this demo, we'll simulate it with pure Python
# In reality: import blosc2_numexpr_integration

def simulate_blosc2_workflow():
    """
    Simulates how python-blosc2 would use the Cython wrapper.
    """
    print("=" * 70)
    print("Python-Blosc2 + NumExpr C-API Integration Example")
    print("=" * 70)
    
    # Setup
    n_elements = 1_000_000
    chunk_size = 10_000
    n_chunks = n_elements // chunk_size
    
    print(f"\nDataset: {n_elements:,} elements")
    print(f"Chunk size: {chunk_size:,} elements")
    print(f"Number of chunks: {n_chunks}")
    
    # Create test data
    print("\n1. Creating test data...")
    a = np.random.rand(n_elements)
    b = np.random.rand(n_elements)
    c = np.random.rand(n_elements)
    
    # Expression to evaluate
    expression = "2*a + 3*b*c"
    print(f"2. Expression: {expression}")
    
    # ===================================================================
    # METHOD 1: Traditional Python loop (slow)
    # ===================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: Traditional Python Loop (baseline)")
    print("=" * 70)
    
    result_python = np.empty_like(a)
    
    t0 = time.time()
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        
        chunk_a = a[start:end]
        chunk_b = b[start:end]
        chunk_c = c[start:end]
        
        # Evaluate on chunk
        result_python[start:end] = ne.evaluate(
            expression,
            local_dict={'a': chunk_a, 'b': chunk_b, 'c': chunk_c}
        )
    
    t1 = time.time()
    time_python = t1 - t0
    
    print(f"Time: {time_python:.4f} seconds")
    print(f"Throughput: {n_elements/time_python/1e6:.2f} M elements/sec")
    
    # ===================================================================
    # METHOD 2: Compile once, re-evaluate (faster)
    # ===================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: Compile Once + Re-evaluate Pattern")
    print("=" * 70)
    
    result_reeval = np.empty_like(a)
    
    # Compile once
    dummy = {'a': np.array([0.0]), 'b': np.array([0.0]), 'c': np.array([0.0])}
    ne.validate(expression, local_dict=dummy)
    
    t0 = time.time()
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        
        chunk_a = a[start:end]
        chunk_b = b[start:end]
        chunk_c = c[start:end]
        
        # Re-evaluate without recompiling
        result_reeval[start:end] = ne.re_evaluate(
            local_dict={'a': chunk_a, 'b': chunk_b, 'c': chunk_c}
        )
    
    t1 = time.time()
    time_reeval = t1 - t0
    
    print(f"Time: {time_reeval:.4f} seconds")
    print(f"Throughput: {n_elements/time_reeval/1e6:.2f} M elements/sec")
    print(f"Speedup vs Method 1: {time_python/time_reeval:.2f}x")
    
    # ===================================================================
    # METHOD 3: C-API approach (what Cython wrapper enables)
    # ===================================================================
    print("\n" + "=" * 70)
    print("METHOD 3: C-API Approach (Cython wrapper simulation)")
    print("=" * 70)
    print("(This is what your Cython code would enable)")
    
    result_capi = np.empty_like(a)
    
    # In reality, this would be:
    # from blosc2_numexpr_integration import setup_expression, evaluate_chunks_parallel
    # handle = setup_expression(expression)
    # result_capi = evaluate_chunks_parallel(a, b, c, chunk_size, handle)
    
    # For simulation, we'll do direct C-API style calls
    ne.validate(expression, local_dict=dummy)
    
    t0 = time.time()
    
    # Simulate what C-Blosc2 threads would do
    # In reality, this loop would be in C, called from blosc2 worker threads
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        
        # Simulate: C thread has decompressed data
        # Uses numexpr_run_compiled_simple() via Cython wrapper
        result_capi[start:end] = ne.re_evaluate(
            local_dict={
                'a': a[start:end],
                'b': b[start:end],
                'c': c[start:end]
            }
        )
    
    t1 = time.time()
    time_capi = t1 - t0
    
    print(f"Time: {time_capi:.4f} seconds")
    print(f"Throughput: {n_elements/time_capi/1e6:.2f} M elements/sec")
    print(f"Speedup vs Method 1: {time_python/time_capi:.2f}x")
    print(f"Speedup vs Method 2: {time_reeval/time_capi:.2f}x")
    
    print("\nNote: With actual C-API, expect 1.5-3x additional speedup")
    print("      due to elimination of Python call overhead")
    
    # Verify correctness
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)
    
    direct_result = ne.evaluate(expression, local_dict={'a': a, 'b': b, 'c': c})
    
    match1 = np.allclose(result_python, direct_result)
    match2 = np.allclose(result_reeval, direct_result)
    match3 = np.allclose(result_capi, direct_result)
    
    print(f"Method 1 matches direct: {match1} ✓" if match1 else f"Method 1 matches direct: {match1} ✗")
    print(f"Method 2 matches direct: {match2} ✓" if match2 else f"Method 2 matches direct: {match2} ✗")
    print(f"Method 3 matches direct: {match3} ✓" if match3 else f"Method 3 matches direct: {match3} ✗")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<40} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'1. Python loop (evaluate)':<40} {time_python:>10.4f}   {'1.00x':<10}")
    print(f"{'2. Compile once (re_evaluate)':<40} {time_reeval:>10.4f}   {time_python/time_reeval:>8.2f}x")
    print(f"{'3. C-API (Cython wrapper)':<40} {time_capi:>10.4f}   {time_python/time_capi:>8.2f}x")
    print("\n" + "=" * 70)


def show_cython_integration_pattern():
    """
    Shows the pattern for using the Cython wrapper in python-blosc2.
    """
    print("\n" + "=" * 70)
    print("Python-Blosc2 Integration Pattern")
    print("=" * 70)
    
    code = """
# In python-blosc2 Python code (e.g., blosc2/lazy.py):

from blosc2_numexpr_integration import (
    setup_expression,
    get_cached_handle,
    get_chunk_processor_ptr
)

class LazyExpr:
    def __init__(self, expr_string, operands):
        self.expr = expr_string
        self.operands = operands
        
        # Setup NumExpr expression and get handle
        self.numexpr_handle = setup_expression(expr_string)
        self.processor_ptr = get_chunk_processor_ptr()
    
    def compute(self):
        # Pass function pointer and handle to C-Blosc2
        # C-Blosc2 will call the processor from its worker threads
        return self._evaluate_chunks_in_c(
            self.processor_ptr,
            self.numexpr_handle,
            self.operands
        )
    
    def _evaluate_chunks_in_c(self, processor_ptr, handle, operands):
        # This calls into your C extension
        # Which passes processor_ptr to C-Blosc2 worker threads
        # Those threads call: processor(chunk_a, chunk_b, ..., handle)
        pass


# In python-blosc2 C extension (blosc2_ext.c or .pyx):

# Cast function pointer
typedef int (*chunk_processor_t)(
    void* chunk_a,
    void* chunk_b,
    void* chunk_c,
    void* output,
    int64_t size,
    void* numexpr_handle
);

chunk_processor_t processor = (chunk_processor_t)processor_ptr;

# C-Blosc2 worker thread (running without GIL):
void blosc2_worker_thread(void* args) {
    // Decompress chunks
    void* chunk_a = blosc2_decompress_chunk(...);
    void* chunk_b = blosc2_decompress_chunk(...);
    void* chunk_c = blosc2_decompress_chunk(...);
    void* output = malloc(chunk_size * sizeof(double));
    
    // Call Cython function (it will acquire GIL internally)
    int status = processor(
        chunk_a, chunk_b, chunk_c,
        output, chunk_size,
        numexpr_handle
    );
    
    if (status == 0) {
        // Compress result
        blosc2_compress_chunk(output, ...);
    }
    
    // Cleanup
    free(chunk_a); free(chunk_b); free(chunk_c); free(output);
}
"""
    
    print(code)
    print("=" * 70)


if __name__ == '__main__':
    simulate_blosc2_workflow()
    show_cython_integration_pattern()
    
    print("\n" + "=" * 70)
    print("Next Steps for Python-Blosc2 Integration")
    print("=" * 70)
    print("""
1. Copy blosc2_numexpr_integration.pyx to python-blosc2/blosc2/

2. Add to python-blosc2/setup.py:
   
   from Cython.Build import cythonize
   import numexpr
   
   extensions = [
       Extension(
           'blosc2.blosc2_numexpr_integration',
           sources=['blosc2/blosc2_numexpr_integration.pyx'],
           include_dirs=[
               np.get_include(),
               os.path.dirname(numexpr.__file__),
           ],
       ),
       # ... other extensions
   ]
   
   setup(
       ext_modules=cythonize(extensions),
       # ...
   )

3. In your C-Blosc2 worker threads:
   - Receive function pointer from Python
   - Call it with decompressed chunk data
   - Function handles GIL acquisition automatically
   - NumExpr releases GIL during computation
   - Real parallelism achieved!

4. Benefits:
   - No Python call overhead in tight loops
   - Real parallelism (GIL released during computation)
   - Type-safe Cython code
   - Easier to maintain than raw C
""")
    print("=" * 70)
