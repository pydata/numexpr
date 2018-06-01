###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: BSD
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import os, sys, platform
import subprocess
import numpy as np
import numexpr3.interpreter
import numexpr3
import time

_nthreads = 1     
_ncores =   1     
_cpu_info = None

def _detect_ncores() -> None:
    """Check for number of cores."""
    # TODO: we would prefer physical cores.
    # Linux, Unix and MacOS:
    global _ncores
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                _ncores = ncpus
        else:  # OSX:
            _ncores = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]))
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            _ncores = ncpus

def _detect_nthreads() -> None:
    """Detect number of threads, if set in an environment variable."""
    # Otherwise use number of detected cores.
    global _nthreads, _ncores
    if 'sparc' in platform.machine():
        import warnings as __warnings
        __warnings.warn('''The number of threads have been set to 1 because problems 
related to threading have been reported on some sparc machines.
The number of threads can be changed using the "set_num_threads"
function.''')
        _nthreads = 1
    elif 'NUMEXPR_NUM_THREADS' in os.environ:
        _nthreads = int( os.environ['NUMEXPR_NUM_THREADS'] )
    elif 'OMP_NUM_THREADS' in os.environ:
        _nthreads = int( os.environ['OMP_NUM_THREADS'] )
    else:
        _nthreads = _ncores

# More module initialization
_detect_ncores()
_detect_nthreads()
_nthreads = numexpr3.interpreter.set_num_threads( _nthreads )

def get_nthreads() -> int:
    """Get the number of threads currently being used by the virtual machine."""
    global _nthreads
    _nthreads = numexpr3.interpreter.get_num_threads()
    return _nthreads

def set_nthreads(new_nthreads: int) -> None:
    """
    Sets a number of threads to be used in operations. Generally speaking one should
    use >= the number of physical cores.  NumExpr does not benefit from 
    Hyperthreading.

    During initialization time Numexpr sets the thread count by 
    the ``_detect_number_of_threads()`` function.
    """
    global _nthreads
    _nthreads = int(new_nthreads)
    numexpr3.interpreter.set_num_threads(_nthreads)

def get_ncores() -> int:
    global _ncores
    return _ncores
# There's no setter for ncores

def str_info() -> str:
    """String representation of software versions that NumExpr imports and CPU information."""
    global _nthreads, _ncores
    
    repr = []
    repr.append('-=' * 38)
    repr.append("Numexpr version:   %s" % numexpr3.__version__)
    repr.append("NumPy version:     %s" % np.__version__)
    repr.append('Python version:    %s' % sys.version)
    try:
        (sysname, nodename, release, version, machine) = os.uname()
        repr.append('Platform:          %s-%s' % (sys.platform, machine))
    except AttributeError:
        repr.append('Platform:          %s-%s' % (sys.platform, os.name))
    # repr.append("VML available?     %s" % use_vml)
    # try: repr.append("VML/MKL version:   %s" % numexpr3.get_vml_version())
    # except NameError: pass
    repr.append("Number of threads used by default: %d "
        "(out of %d detected cores)" % (_nthreads, _ncores))

    ## Unfortuantely cpuinfo is too slow, this doubles the test time (!!!) on Windows.
    # repr.append( "CPU information: " )
    # repr.append( "_"*78 )
    # _cpu_info = _cpu_info
    # if _cpu_info:
    #     repr.append('Vendor ID: {0}'.format(_cpu_info.get('vendor_id', '')))
    #     repr.append('Brand: {0}'.format(_cpu_info.get('brand', '')))
    #     repr.append('Hz Actual: {0}'.format(_cpu_info.get('hz_actual', '')))
    #     repr.append('Arch: {0}'.format(_cpu_info.get('arch', '')))
    #     repr.append('Bits: {0}'.format(_cpu_info.get('bits', '')))
    #     repr.append('Core Count: {0}'.format(_cpu_info.get('count', '')))
    #     repr.append('L2 Cache Size: {0}'.format(_cpu_info.get('l2_cache_size', '')))
    #     repr.append('L2 Cache Line Size: {0}'.format(_cpu_info.get('l2_cache_line_size', '')))
    #     repr.append('L2 Cache Associativity: {0}'.format(_cpu_info.get('l2_cache_associativity', '')))
    #     repr.append('Stepping: {0}'.format(_cpu_info.get('stepping', '')))
    #     repr.append('Model: {0}'.format(_cpu_info.get('model', '')))
    #     repr.append('Family: {0}'.format(_cpu_info.get('family', '')))
    #     repr.append('Processor Type: {0}'.format(_cpu_info.get('processor_type', '')))
    #     repr.append('Extended Model: {0}'.format(_cpu_info.get('extended_model', '')))
    #     repr.append('Extended Family: {0}'.format(_cpu_info.get('extended_family', '')))
    #     repr.append('Flags: {0}'.format(', '.join(_cpu_info.get('flags', ''))))
    repr.append('-=' * 38)
    return ''.join(repr)

def print_info() -> None:
    """Print software versions that NumExpr imports and CPU information."""
    print(str_info())

