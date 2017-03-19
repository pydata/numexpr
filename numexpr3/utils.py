###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: BSD
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import os, sys
import subprocess
import numpy as np
from numexpr3.interpreter import _set_num_threads
import numexpr3

try:
    from numexpr3.interpreter import (
        _get_vml_version, _set_vml_accuracy_mode, _set_vml_num_threads)
    
    def get_vml_version():
        """Get the VML/MKL library version."""
        return _get_vml_version()

    def set_vml_accuracy_mode(mode):
        """
        Set the accuracy mode for VML operations.
    
        The `mode` parameter can take the values:
        - 'high': high accuracy mode (HA), <1 least significant bit
        - 'low': low accuracy mode (LA), typically 1-2 least significant bits
        - 'fast': enhanced performance mode (EP)
        - None: mode settings are ignored
    
        This call is equivalent to the `vmlSetMode()` in the VML library.
        See:
    
        http://www.intel.com/software/products/mkl/docs/webhelp/vml/vml_DataTypesAccuracyModes.html
    
        for more info on the accuracy modes.
    
        Returns old accuracy settings.
        """
        acc_dict = {None: 0, 'low': 1, 'high': 2, 'fast': 3}
        acc_reverse_dict = {1: 'low', 2: 'high', 3: 'fast'}
        if mode not in acc_dict.keys():
            raise ValueError(
                "mode argument must be one of: None, 'high', 'low', 'fast'")
        retval = _set_vml_accuracy_mode(acc_dict.get(mode, 0))
        return acc_reverse_dict.get(retval)



    def set_vml_num_threads(new_nthreads):
        """
        Suggests a maximum number of threads to be used in VML operations.
    
        This function is equivalent to the call
        `mkl_domain_set_num_threads(nthreads, MKL_DOMAIN_VML)` in the MKL
        library.  See:
    
        http://www.intel.com/software/products/mkl/docs/webhelp/support/functn_mkl_domain_set_num_threads.html
    
        for more info about it.
        """
        _set_vml_num_threads(new_nthreads)
        
except ImportError: 
    pass # End of VML utility function import block

def print_info():
    """Print the versions of software that numexpr relies on."""
    print('-=' * 38)
    print("Numexpr version:   %s" % numexpr3.__version__)
    print("NumPy version:     %s" % np.__version__)
    print('Python version:    %s' % sys.version)
    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print('Platform:          %s-%s' % (sys.platform, machine))
    # print("VML available?     %s" % use_vml)
    try: print("VML/MKL version:   %s" % numexpr3.get_vml_version())
    except NameError: pass
    print("Number of threads used by default: %d "
          "(out of %d detected cores)" % (numexpr3.nthreads, numexpr3.ncores))
    print('-=' * 38)
    


def set_num_threads(new_nthreads):
    """
    Sets a number of threads to be used in operations.

    Returns the previous setting for the number of threads.

    During initialization time Numexpr sets this number to the number
    of detected cores in the system (see `detect_number_of_cores()`).

    If you are using Intel's VML, you may want to use
    `set_vml_num_threads(nthreads)` to perform the parallel job with
    VML instead.  However, you should get very similar performance
    with VML-optimized functions, and VML's parallelizer cannot deal
    with common expresions like `(x+1)*(x-2)`, while Numexpr's one
    can.
    """
    old_nthreads = _set_num_threads(new_nthreads)
    numexpr3.nthreads = new_nthreads
    return old_nthreads


def detect_number_of_cores():
    """
    Detects the number of cores on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]))
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
        if ncpus > 0:
            return ncpus
    return 1  # Default

def detect_number_of_threads():		
    """		
    If this is modified, please update the note in: https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide		
    """		
    try:		
        nthreads = int(os.environ['NUMEXPR_NUM_THREADS'])		
    except KeyError:		
        nthreads = int(os.environ.get('OMP_NUM_THREADS', detect_number_of_cores()))		
        # Check that we don't activate too many threads at the same time.		
        # 8 seems a sensible value.
        max_sensible_threads = 8
        if nthreads > max_sensible_threads:	# RAM: add a notification
            print( "NumExpr3 notification: n_threads set to " + str(max_sensible_threads) 
				  + ", increase with util.set_num_threads(n)" )
            nthreads = max_sensible_threads		
    # Check that we don't surpass the MAX_THREADS in interpreter.cpp		
    if nthreads > 256:		
        nthreads = 256		
    return nthreads		


