#################################################################################
# To mimic the scenario that computation is i/o bound and constrained by memory
#
# It's a much simplified version that the chunk is computed in a loop,
# and expression is evaluated in a sequence, which is not true in reality.
# Neverthless, numexpr outperforms numpy.
#################################################################################
"""
Benchmarking Expression 1:
NumPy time (threaded over 32 chunks with 2 threads): 4.612313 seconds
numexpr time (threaded with re_evaluate over 32 chunks with 2 threads): 0.951172 seconds
numexpr speedup: 4.85x
----------------------------------------
Benchmarking Expression 2:
NumPy time (threaded over 32 chunks with 2 threads): 23.862752 seconds
numexpr time (threaded with re_evaluate over 32 chunks with 2 threads): 2.182058 seconds
numexpr speedup: 10.94x
----------------------------------------
Benchmarking Expression 3:
NumPy time (threaded over 32 chunks with 2 threads): 20.594895 seconds
numexpr time (threaded with re_evaluate over 32 chunks with 2 threads): 2.927881 seconds
numexpr speedup: 7.03x
----------------------------------------
Benchmarking Expression 4:
NumPy time (threaded over 32 chunks with 2 threads): 12.834101 seconds
numexpr time (threaded with re_evaluate over 32 chunks with 2 threads): 5.392480 seconds
numexpr speedup: 2.38x
----------------------------------------
"""

import os

os.environ["NUMEXPR_NUM_THREADS"] = "16"
import numpy as np
import numexpr as ne
import timeit
import threading

array_size = 10**8
num_runs = 10
num_chunks = 32  # Number of chunks
num_threads = 2  # Number of threads constrained by how many chunks memory can hold

a = np.random.rand(array_size).reshape(10**4, -1)
b = np.random.rand(array_size).reshape(10**4, -1)
c = np.random.rand(array_size).reshape(10**4, -1)

chunk_size = array_size // num_chunks

expressions_numpy = [
    lambda a, b, c: a + b * c,
    lambda a, b, c: a**2 + b**2 - 2 * a * b * np.cos(c),
    lambda a, b, c: np.sin(a) + np.log(b) * np.sqrt(c),
    lambda a, b, c: np.exp(a) + np.tan(b) - np.sinh(c),
]

expressions_numexpr = [
    "a + b * c",
    "a**2 + b**2 - 2 * a * b * cos(c)",
    "sin(a) + log(b) * sqrt(c)",
    "exp(a) + tan(b) - sinh(c)",
]


def benchmark_numpy_chunk(func, a, b, c, results, indices):
    for index in indices:
        start = index * chunk_size
        end = (index + 1) * chunk_size
        time_taken = timeit.timeit(
            lambda: func(a[start:end], b[start:end], c[start:end]), number=num_runs
        )
        results.append(time_taken)


def benchmark_numexpr_re_evaluate(expr, a, b, c, results, indices):
    for index in indices:
        start = index * chunk_size
        end = (index + 1) * chunk_size
        if index == 0:
            # Evaluate the first chunk with evaluate
            time_taken = timeit.timeit(
                lambda: ne.evaluate(
                    expr,
                    local_dict={
                        "a": a[start:end],
                        "b": b[start:end],
                        "c": c[start:end],
                    },
                ),
                number=num_runs,
            )
        else:
            # Re-evaluate subsequent chunks with re_evaluate
            time_taken = timeit.timeit(
                lambda: ne.re_evaluate(
                    local_dict={"a": a[start:end], "b": b[start:end], "c": c[start:end]}
                ),
                number=num_runs,
            )
        results.append(time_taken)


def run_benchmark_threaded():
    chunk_indices = list(range(num_chunks))

    for i in range(len(expressions_numpy)):
        print(f"Benchmarking Expression {i+1}:")

        results_numpy = []
        results_numexpr = []

        threads_numpy = []
        for j in range(num_threads):
            indices = chunk_indices[j::num_threads]  # Distribute chunks across threads
            thread = threading.Thread(
                target=benchmark_numpy_chunk,
                args=(expressions_numpy[i], a, b, c, results_numpy, indices),
            )
            threads_numpy.append(thread)
            thread.start()

        for thread in threads_numpy:
            thread.join()

        numpy_time = sum(results_numpy)
        print(
            f"NumPy time (threaded over {num_chunks} chunks with {num_threads} threads): {numpy_time:.6f} seconds"
        )

        threads_numexpr = []
        for j in range(num_threads):
            indices = chunk_indices[j::num_threads]  # Distribute chunks across threads
            thread = threading.Thread(
                target=benchmark_numexpr_re_evaluate,
                args=(expressions_numexpr[i], a, b, c, results_numexpr, indices),
            )
            threads_numexpr.append(thread)
            thread.start()

        for thread in threads_numexpr:
            thread.join()

        numexpr_time = sum(results_numexpr)
        print(
            f"numexpr time (threaded with re_evaluate over {num_chunks} chunks with {num_threads} threads): {numexpr_time:.6f} seconds"
        )
        print(f"numexpr speedup: {numpy_time / numexpr_time:.2f}x")
        print("-" * 40)


if __name__ == "__main__":
    run_benchmark_threaded()
