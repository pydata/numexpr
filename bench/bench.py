# Setup number of cores and associated environment variables
n_cores = 8
import os
os.environ['NUMBA_NUM_THREADS'] = str(n_cores)

# Imports
import numpy as np
import numexpr3 as ne3
from time import perf_counter as pc
import os
import timeit
from collections import defaultdict
import colorama
from colorama import Fore, Back, Style
colorama.init()

import numexpr as ne2
import numba

try:
    from matplotlib import use
    use('Qt5Agg')
    import matplotlib.pyplot as plt
    PLT = True
except ImportError:
    PLT = False

RED = 0
PURPLE = -8
_COLOR_MAG = { 0: Fore.LIGHTRED_EX, -1: Fore.RED, -2: Fore.LIGHTYELLOW_EX, 
                -3: Fore.YELLOW, -4: Fore.LIGHTGREEN_EX, -5: Fore.GREEN, 
                -6: Fore.LIGHTBLUE_EX, -7: Fore.BLUE, -8: Fore.MAGENTA }

class Case(object):
    _MAX_ARGS = 4
    _C_BENCH = {0:'prepare_threads', 1:'run_preamble', 2:'magic_broadcast', 
                            3:'npyiter_con', 4:'barrier_init', 5:'task_barrier_final',
                            6:'unlock_global', 7:'run_clean', 8:'release_gil', 
                            9:'reacquire_gil',
                            50: 'cobj_init' }

    _C_THREADS = {150:'thread_init', 200:'thread_tasks', 250:'thread_final'}

    def initZeros(self):
        # return [0.0]*self.tries
        return np.zeros(self.tries, dtype='float64')

    def __init__(self, expr, size, dtype, tries=1, 
                 compares={}, nthreads=ne3.get_nthreads() ):
        self.expr       = expr
        self.size       = size
        self.dtype      = dtype
        self.tries      = tries
        self.compares   = compares    # Should be a dict of {name: function_handle}
        self.nthreads   = nthreads
        self.timings    = defaultdict(self.initZeros, {})
        self.stats      = defaultdict(self.initZeros, {})


    def run(self):
        # Setup variables x1, x2, x3, x4
        # print( "Running benchmark for '{}' x {} times".format(self.expr, self.tries))
        args = []
        varsDict = vars()
        r = np.empty(self.size, dtype=self.dtype)
        for I in range(Case._MAX_ARGS):
            argName = 'x%d'%I

            if argName in self.expr:
                argArray = np.arange(self.size, dtype=self.dtype)
                varsDict[argName] = argArray
                args.append(argArray)

        x1 = args[0]
        x2 = args[1]
        # Run benchmark `tries` times
        ne3.set_nthreads(self.nthreads)
        ne2.set_num_threads(self.nthreads)
        # Numba does not have dynamic thread setting
        for I in range(self.tries):
            # Grab local references to our attributes
            timings = self.timings; stats = self.stats

            t0 = pc()
            nefunc = ne3.NumExpr( self.expr )
            t1 = pc()
            nefunc()
            t2 = pc()
            timings['total_run'][I] = t2 - t1
            timings['total_assemble'][I] = t1 - t0
            timings['total'][I] = t2 - t0

            if bool(self.compares):
                for name, func in self.compares.items():
                    t3 = pc()
                    # func(*args)
                    func(x1, x2)
                    t4 = pc()
                    timings['compare', name][I] = t4 - t3
            
            # Get Python times
            for timingKey in nefunc.timings:
                timings[timingKey][I] = nefunc.timings[timingKey]
            # Get C times
            if os.name == 'nt':
                bench_times = ne3.interpreter.bench_times / ne3.interpreter.cpu_freq
            else:
                bench_times = ne3.interpreter.bench_times[0::2] + 1.0E-9*ne3.interpreter.bench_times[1::2]

            timings['missing_assemble'][I] = timings['total_assemble'][I] - \
                (timings['__init__'][I]+timings['frame_call'][I]+timings['ast_parse'][I]+timings['ast_walk'][I]+timings['to_tuple'][I]+timings['obj_construct'][I] )
            timings['missing_run'][I] = timings['total_run'][I] - \
                (np.sum(bench_times[0:10])+timings['run_pre'][I]+timings['run_post'][I])

            for key,name in Case._C_BENCH.items():
                timings[name][I] = bench_times[key]

            for key,name in Case._C_THREADS.items():
                if not name in timings:
                    timings[name] = np.zeros( [self.tries, self.nthreads] )
                    
                timings[name][I] = bench_times[key:key+self.nthreads]

        
        stats['thread_task_run_ratio'] = timings['thread_tasks'].mean() / timings['total_run'].mean()
        # print( 'Init: ' + str(timings['thread_init'].mean(axis=1) / timings['total_run'] ) )
        # print( 'Tasks: ' + str(timings['thread_tasks'].mean(axis=1) / timings['total_run'] ) )
        # print( 'Final: ' + str(timings['thread_final'].mean(axis=1) / timings['total_run'] ) )
        stats['thread_barrier_run_ratio'] = ((timings['thread_init'] +timings['thread_final']).mean(axis=1) / timings['total_run'] ).mean()
        stats['overhead_run_ratio'] = 1.0 - stats['thread_task_run_ratio'] - stats['thread_barrier_run_ratio']
        stats['thread_load_balance'] = timings['thread_tasks'].std()/timings['thread_tasks'].mean()
        stats['n_fails'] = np.sum(timings['thread_tasks'] < 1E-8 )
        stats['n_fails_prob'] = stats['n_fails']/self.nthreads/self.tries
        for name in self.compares:
            stats['compare_scaling', name] = timings['compare', name].mean()/timings['total'].mean()
            stats['compare_scaling_run', name] = timings['compare', name].mean()/timings['total_run'].mean()


    def print_results(self, scale=1E6):
        # Grab local references to our attributes
        timings = self.timings; stats = self.stats

        print( "---===BENCHMARK RESULTS===---\n" )
        print( "  Expression:  {}".format(self.expr) )
        print( "  N_threads:        {}".format(self.nthreads) )
        print( "  Array size: {},  dtype: {}".format(self.size, self.dtype) )
        if os.name == 'nt':
            print( "  Reported CPU frequency: {:.3f} GHz".format(ne3.interpreter.cpu_freq/1E6) )
        print( "________________________________________________________________" )
        for name in self.compares:
            print( "{:8} function:                  {:.2f} ± {:.2f} μs".format( 
                name, scale*timings['compare', name].mean(), scale*timings['compare', name].std()) )
        print( "NumExpr3 function:               {:10.2f} ± {:.2f} μs".format( 
            scale*timings['total'].mean(), scale*timings['total'].std() ) )
        print( "________________________________________________________________" )
        self.cfp( "Assembly time:                   {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'total_assemble' )
        self.cfp( "  __init__:                      {:10.2f} ± {:6.2f} μs ({:2.2f} %)", '__init__' )
        self.cfp( "  sys.getframe:                  {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'frame_call' )
        self.cfp( "  ast.parse:                     {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'ast_parse' )
        self.cfp( "  ASTAssembler tree walk:        {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'ast_walk' )
        self.cfp( "  Register mutation              {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'to_tuple' )
        self.cfp( "  NumExpr Object Construction    {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'obj_construct' )
        self.cfp( "  Unaccounted time:              {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'missing_assemble' )
        self.cfp( "Run time:                        {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'total_run' )
        self.cfp( "  Python pre-VM entry run():     {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'run_pre' )
        self.cfp( "  Run preamble:                  {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'run_preamble' )
        self.cfp( "  Magic output broadcasting:     {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'magic_broadcast' )
        self.cfp( "  NpyIter construction:          {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'npyiter_con' )
        self.cfp( "  Prepare threads/serial tasks:  {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'prepare_threads' )
        self.cfp( "  VM Initialize Barrier:         {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'barrier_init' )
        self.cfp( "  Release GIL                    {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'release_gil' )
        self.cfp( "  VM Task + Finalize Barrier:    {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'task_barrier_final' )

        # # Somehow thread_tasks is just not being written?
        # print( 'Thread init: \n{}'.format(timings['thread_init']))
        # print( 'Thread tasks: \n{}'.format(timings['thread_tasks']))
        # print( 'Thread finalization: \n{}'.format(timings['thread_final']))
        
        self.cfp( "    Thread initialization:       {:10.2f} ± {:6.2f} μs ({:2.1f} %)", 'thread_init' )
        self.cfp( "    Thread task loop:            {:10.2f} ± {:6.2f} μs ({:2.1f} %)", 'thread_tasks' )
        self.cfp( "    Thread finalization:         {:10.2f} ± {:6.2f} μs ({:2.1f} %)", 'thread_final' )
        self.cfp( "  Reacquire GIL                  {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'reacquire_gil' )
        self.cfp( "  VM clean-up and exit:          {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'run_clean' )
        self.cfp( "  Python post-VM exit run():     {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'run_post' )
        self.cfp( "  Unaccounted time:              {:10.2f} ± {:6.2f} μs ({:2.2f} %)", 'missing_run' )
        
        print( "________________________________________________________________" )
        print( "Statistics:" )
        print( "  VM program execution:                      {:4.1f} %".format(100.0*stats['thread_task_run_ratio']) )
        print( "  Threading barrier overhead:                {:4.1f} %".format(100.0*stats['thread_barrier_run_ratio']) )
        print( "  Run setup overhead:                        {:4.1f} %".format(100.0*stats['overhead_run_ratio']) )
        print( "  Thread load-balancing standard deviation:  {:4.1f} %".format(100.0*stats['thread_load_balance']) )
        if stats['n_fails'] > 0:
            print( ''.join([Fore.RED, "  # threads that failed to start:       {:2} ({:4.1f} %)".format(stats['n_fails'], 100.0*stats['n_fails_prob']), Fore.RESET]) )
        for name in self.compares:
            print( "  Scaling compared to {:8}:             {:4.1f} %".format(name, 100.0*stats['compare_scaling', name] ))


    def cfp(self, text, key, scale=1e6 ):
        '''
        Colored-formatted print
        '''
        time = self.timings[key]
        meanTime = time.mean()
        stdTime = time.std()
        
        mag = 0
        try:
            percentTime = meanTime / self.timings['total'].mean()
            mag = np.clip( int(np.log(percentTime)), PURPLE, RED) if percentTime > 0.0 else PURPLE
        except OverflowError:
            mag = PURPLE
        
        print( ''.join([ 
            _COLOR_MAG[mag], text.format(scale*meanTime, scale*stdTime, 100.0*percentTime), Style.RESET_ALL
        ]))


testSize = 2**20
dtype1 = 'float64'
expr1 = 'r=x1*sin(x2)'
def case1_numpy(x1, x2): 
    return x1*np.sin(x2)

def case1_ne2(x1, x2):
    return ne2.evaluate('x1*sin(x2)')

# You _MUST_ set NUMBA_NUM_THREADS before importing Numba
@numba.vectorize([numba.float64(numba.float64, numba.float64)], target='parallel', cache=False)
def case1_numba(x1, x2):
    return x1 * np.sin(x2)

compares = {'NumExpr2': case1_ne2, 'NumPy':case1_numpy, 'Numba':case1_numba}

threadRange = np.arange(1, n_cores+1)
scalingNE3 = np.zeros(n_cores)
scalingNE2 = np.zeros(n_cores)
scalingNumba = np.zeros(n_cores)
threadFails = np.zeros(n_cores)
for I, T in enumerate(threadRange):
    # case1 = Case(expr='r=x1*sin(np.pi*x2)', size=2**20, dtype='float64', tries=5,
    #     compare=case1_numpy, nthreads=T
    # )
    
    case1 = Case(expr=expr1, size=testSize, dtype=dtype1, tries=5,
        compares=compares, nthreads=T
    )
    case1.run()
    case1.print_results()
    # 'compare_scaling' includes assembly time, 'compare_scaling_run' does not
    scalingNE3[I] = case1.stats['compare_scaling_run', 'NumPy']
    scalingNE2[I] = case1.stats['compare_scaling_run', 'NumPy'] / case1.stats['compare_scaling_run', 'NumExpr2']
    scalingNumba[I] = case1.stats['compare_scaling_run', 'NumPy'] / case1.stats['compare_scaling_run', 'Numba']
    threadFails[I] = case1.stats['n_fails_prob']

if PLT:
    plt.figure()
    plt.plot( threadRange, scalingNE3*100.0, '.-', markeredgecolor='k', label='NumExpr3' )
    plt.plot( threadRange, scalingNE2*100.0, '.-', markeredgecolor='k', label='NumExpr2' )
    plt.plot( threadRange, scalingNumba*100.0, '.-', markeredgecolor='k', label='Numba (fixed@{} threads)'.format(os.environ['NUMBA_NUM_THREADS']) )
    plt.xlabel( 'Number of threads' )
    plt.ylabel( 'Scaling relative to NumPy(%)' )
    plt.legend(loc='best')
    plt.title( "Scaling for '{}' on {}k,{} arrays".format(case1.expr, case1.size/1024, case1.dtype))
    plt.ylim([0.0, 100.0 * np.max([scalingNE2.max(), scalingNE3.max(), scalingNumba.max()])])
    plt.show(block=True)

print("NOTE: Numba cannot be dynamically thread scaled; Numba threads = {}".format(os.environ['NUMBA_NUM_THREADS']))


