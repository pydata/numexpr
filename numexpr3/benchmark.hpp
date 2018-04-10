#ifndef NUMEXPR_BENCH_FUNC
#define NUMEXPR_BENCH_FUNC

#define MILLION 1000000L
#define BILLION 1000000000L
#define BCOUNT 512

// Use the --bench flag to enable benchmark results
// (Some uncommenting in ne3compiler.py may also be necessary to get Python 
//  benchmarks, I know of no zero-overhead way to avoid that.)
// e.g.
//    python setup.py install --bench

#if defined(_WIN32) && defined(BENCHMARKING)

    extern LARGE_INTEGER TIMES[BCOUNT];
    extern LARGE_INTEGER T_NOW;
    extern double FREQ;

    // Get a single time mark
    #define BENCH_TIME(index) QueryPerformanceCounter(&TIMES[index])
    #define BENCH_RANGE(index, count) for(int T=0;T<count;T++){BENCH_TIME(index+T);}
    #define DIFF_TIME(index) QueryPerformanceCounter(&T_NOW); TIMES[index].QuadPart=T_NOW.QuadPart-TIMES[index].QuadPart

    // Find the accumulated time between index start and now, and accumulate that value in end
    //#define ACCUM_TIME(start,end) QueryPerformanceCounter(&T_NOW); TIMES[end].QuadPart+=T_NOW.QuadPart-TIMES[start].QuadPart

#elif defined(BENCHMARKING) // Linux
#include <time.h>
    extern timespec TIMES[BCOUNT];
    extern timespec T_NOW;

    // Get a single time mark
    #define BENCH_TIME(index) clock_gettime(CLOCK_REALTIME, &TIMES[index] )
    #define BENCH_RANGE(index, count) for(int T=0;T<count;T++){BENCH_TIME(index+T);}

    #define DIFF_TIME(index) clock_gettime(CLOCK_REALTIME, &T_NOW); TIMES[index].tv_nsec=T_NOW.tv_nsec-TIMES[index].tv_nsec; TIMES[index].tv_sec=T_NOW.tv_sec-TIMES[index].tv_sec

    // Find the accumulated time between index start and now, and accumulate that value in end
    //#define ACCUM_TIME(start, end) clock_gettime(CLOCK_REALTIME, &T_NOW); TIMES[end].tv_nsec+=T_NOW.tv_nsec-TIMES[start].tv_nsec; TIMES[end].tv_sec+=T_NOW.tv_sec-TIMES[start].tv_sec

#else  // No benchmarking
    #define BENCH_TIME(index)  // Do nothing
    #define DIFF_TIME(index)
    #define BENCH_RANGE(index, count)

#endif 

#endif // NUMEXPR_BENCH_FUNC