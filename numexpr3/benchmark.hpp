#ifndef NUMEXPR_BENCH_FUNC
#define NUMEXPR_BENCH_FUNC

#define MILLION 1000000L
#define BILLION 1000000000L

// BENCHMARKING, Comment out to disable benchmarking code
// #define BENCHMARKING 1

#if defined(_WIN32) && defined(BENCHMARKING)
    extern LARGE_INTEGER TIMES[512];
    extern double FREQ;

    #define BENCH_TIME(index) QueryPerformanceCounter(&TIMES[index])

    double inline
    GetDiff( int startIndex, int endIndex ) {
        // Returns time elapsed between two benchmark points in microseconds
        LARGE_INTEGER elapsed;
        elapsed.QuadPart = TIMES[endIndex].QuadPart - TIMES[startIndex].QuadPart;
        return (MILLION*(double)elapsed.QuadPart) / FREQ;
    }

    void inline
    printBenchmarks() {
        LARGE_INTEGER cpu_freq;
        QueryPerformanceFrequency( &cpu_freq );
        FREQ = (double)cpu_freq.QuadPart;

        printf( "---===BENCHMARK RESULTS===---\n" );
        printf( "  Clock frequency:  %f GHz\n", (FREQ/1e6) );
        printf( "  N_threads:        %d\n", gs.n_thread );
        printf( "  Temporary arena:  %d KB\n", (double)gs.tempSize/1024.0 );
        printf( "__________________________________________________________\n" );
        printf( "  Prepare threads/serial tasks:  %.3f us\n", GetDiff(0,1)  );
        for( int T=0; T < gs.n_thread; T++ ) {
            printf( "  Thread #%d task loop:  %.3f us\n", T, GetDiff(100+T,200+T) );
        }
        printf( "\n" );
    }
# elif defined(BENCHMARKING) // Linux
#include <time.h>
    extern timespec TIMES[512];

    #define BENCH_TIME(index) clock_gettime(CLOCK_REALTIME, &TIMES[index] )

    double inline
    GetDiff( int startIndex, int endIndex ) {
        // Returns time elapsed between two benchmark points in nanoseconds
        return 0.001*(double)( BILLION*(TIMES[endIndex].tv_sec - TIMES[startIndex].tv_sec) + TIMES[endIndex].tv_nsec - TIMES[startIndex].tv_nsec );
    }

    void inline
    printBenchmarks() {
        printf( "---===BENCHMARK RESULTS===---\n" );
        // printf( "  Clock frequency:  %f GHz\n", (FREQ/1e6) );
        printf( "  N_threads:        %d\n", gs.n_thread );
        printf( "  Temporary arena:  %.0f KB\n", (double)(gs.tempSize)/1024.0 );
        printf( "__________________________________________________________\n" );
        printf( "  Prepare threads/serial tasks:  %.3f us\n", GetDiff(0,1)  );
        for( int T=0; T < gs.n_thread; T++ ) {
            printf( "  Thread #%d task loop:  %.3f us\n", T, GetDiff(100+T,200+T) );
        }
        printf( "\n" );
    }
#else  // No benchmarking
    #define BENCH_TIME(index)  // Do nothing
#endif 

#endif // NUMEXPR_BENCH_FUNC