#!/usr/bin/env python
###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: BSD 3-clause
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import os, os.path, sys, glob, platform
import time
import numpy as np
from setuptools import setup, Extension
from setuptools.config import read_configuration

if sys.version_info.major == 3 and sys.version_info.minor < 8:
    raise RuntimeError( 'NumExpr requires Python 3.8 or greater.' )
    
# Increment version for each PyPi release.
major_ver = 3
minor_ver = 0
nano_ver = 1
branch = 'a8'
version = '%d.%d.%d%s' % (major_ver, minor_ver, nano_ver, branch)

# Write __version__.py
with open( 'numexpr3/__version__.py', 'w' ) as fh:
    fh.write( "__version__ = '{}'\n".format(version) )
with open( 'doc/__version__.py', 'w' ) as fh:
    fh.write( "__version__ = '{}'\n".format(version) )   

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


lib_dirs = []
inc_dirs = [np.get_include()]
libs = []  # Pre-built libraries ONLY, like python36.so
clibs = []
def_macros = []
sources = ['numexpr3/interpreter.cpp',
           'numexpr3/module.cpp',
           'numexpr3/numexpr_object.cpp']
depends = [ 'numexpr3/functions_GENERATED.cpp',
            'numexpr3/interp_body_GENERATED.cpp',
            'numexpr3/interp_header_GENERATED.hpp',
            'numexpr3/module.hpp',
            'numexpr3/msvc_function_stubs.hpp',
            'numexpr3/numexpr_config.hpp',
            'numexpr3/numexpr_object.hpp',
            'numexpr3/complex_functions.hpp',
            'numexpr3/real_functions.hpp',
            'numexpr3/string_functions.hpp',
            'numexpr3/benchmark.hpp']
extra_cflags = []
extra_link_args = []

if platform.uname().system == 'Windows':
    # For MSVC only
    if "MSC" in platform.python_compiler():
        extra_cflags = ['/O2']
    extra_link_args = []
    sources.append('numexpr3/win32/pthread.c')
else:
    extra_cflags = []
    extra_link_args = []


# Process additional command-line arguments before distutils gets them
NOGEN = False
BENCHMARKING = False
args = sys.argv[1:]
if len(args) == 1 and args[0] == '--help' or args[0] == '-h':
    print('''
Setup options for Numexpr3:
    --bench         : enable benchmarking suite in virtual machine
    --nogen         : forbid the generator from overwritting `*_GENERATED.py` files; useful for making manual modifications to these files for debugging.
    --help-commands : display help on available commands
    cmd --help      : display help on a specific command.
''')
    sys.exit()

for arg in args:
    if arg == '--nogen':
        NOGEN = True
        sys.argv.remove(arg)
    elif arg == '--bench':
        BENCHMARKING = True
        sys.argv.remove(arg)
        

# Unique MSVC / GCC / LLVM flags
# TODO: compile detections of AVX2 and SSE2 using cpuinfo
# extra_compile_args = {}
# extra_libraries = {}
# extra_link_args = {}
# define_macros = {}

#### GCC (default) ####
# Turned off funroll, doesn't seem to affect speed.
# https://gcc.gnu.org/onlinedocs/gcc-6.4.0/gcc/AArch64-Options.html#AArch64-Options
# For GCC -fopt-info-vec-missed adds a ton of information that's a little too much
# to easily process.
# extra_compile_args['default'] = [  '-fopt-info-vec', '-fopt-info-vec-missed', '-fdiagnostics-color=always' ]
# Try turning off march=native for these complex function errors.
# extra_compile_args['default'] = [ '-march=native', '-fopt-info-vec' ]
# extra_libraries['default'] = ['m']
# extra_link_args['default'] = []
# define_macros['default'] = []

#### MSVC ####
# Due to distutils using os.spawnv(), we don't get back the auto-vectorization messages
# from MSVC. Manually running the compile command by hand can show the 
# results.
'''"C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\BIN\\amd64\\cl.exe" /c
/nologo /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /W3 /GS- /DNDEBUG /MD
-IC:\Anaconda3\lib\site-packages\numpy\core\include -IC:\\Anaconda3\\include
-IC:\Anaconda3\include -I"C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\INCLUDE"
-I"C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\ATLMFC\\INCLUDE"
-I"C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.10240.0\\ucrt"
-I"C:\\Program Files (x86)\\Windows Kits\\NETFXSDK\\4.6.1\\include\\um"
-I"C:\\Program Files (x86)\\Windows Kits\\8.1\\include\\shared"
-I"C:\\Program Files (x86)\\Windows Kits\\8.1\\include\\um"
-I"C:\\Program Files (x86)\\Windows Kits\\8.1\\include\\winrt"
/EHsc /Tpnumexpr3\interpreter.cpp /Fobuild\\temp.win-amd64-3.6\\Release\\numexpr3\\interpreter.obj /Zm1000'''

# Windows auto-vectorization error messages:
# https://blogs.msdn.microsoft.com/nativeconcurrency/2012/05/22/auto-vectorizer-in-visual-studio-2012-did-it-work/
# Overall the the degree of auto-vectorization is fairly poor in MSVC compared to GCC.
# We also have a problem whenever the stride is zero (i.e. scalar constants): 
# these cases could use different paths with explicit array[0] indexing.
# Other options for arch are '/arch:AVX512', '/arch:AVX2'
# '/fp:fast' was tried and caused accuracy problems.
# extra_compile_args['msvc'] =  [ '/Qvec-report:2'  ]
# extra_compile_args['msvc'] =  []
# extra_libraries['msvc'] = []
# extra_link_args['msvc'] = []
# define_macros['msvc'] = []

def run_generator(blocksize=(4096,32), mkl=False, C11=True):
    # Do not generate new files if the GENERATED files are all newer than
    # interp_generator.py.  This saves recompiling if its not needed.
    GENERATED_files = glob.glob('numexpr3/*GENERATED*') + glob.glob('numexpr3/tests/*GENERATED*')
    generator_time  = os.path.getmtime('code_generators/interp_generator.py')
    if all([generator_time < os.path.getmtime(GEN_file) for GEN_file in GENERATED_files]) \
        and os.path.isfile('numexpr3/lookup.pkl'):

        # Open the 
        import pickle
        with open('numexpr3/lookup.pkl', 'rb') as lookup:
            OPTABLE = pickle.load(lookup)
            if 'os.name' in OPTABLE and OPTABLE['os.name'] == os.name:
                print('---===Generation not required===---')
                return

 
    # This no-generated could cause problems if people clone from GitHub and 
    # we insert configuration tricks into the code_generator directory.
    # TODO: It also needs to see if the stubs have been incremented.
    import code_generators.interp_generator as generator
    print( '---===GENERATING INTERPRETER CODE===---' )
    # Try to auto-detect MKL and C++/11 here.
    # mkl=True 
    
    # TODO: pass a configuration dict instead of a list of parameters.  For 
    # example ICC might be another one...
    generator.generate(blocksize=blocksize, mkl=mkl, C11=C11)
    

def setup_package():

    numexpr_extension = Extension(
        'numexpr3.interpreter',
        include_dirs=inc_dirs,
        define_macros=def_macros,
        sources=sources,
        depends=depends,
        library_dirs=lib_dirs,
        libraries=libs,
        extra_compile_args=extra_cflags,
        extra_link_args=extra_link_args,
    )
    
    metadata = dict(
        install_requires=requirements,
        libraries=clibs,
        ext_modules=[
            numexpr_extension
        ],
        data_files=[
            'numexpr3/lookup.pkl'
        ]
    )
    setup(**metadata)


if __name__ == '__main__':
    # On cloning from GitHub the lookup dict doesn't exist so we always must run the generator.
    # Otherwise the numpy.distutils check for the file occurs before build_ext is called.
    if not os.path.isfile('numexpr3/lookup.pkl'):
        run_generator()

    t0 = time.time()
    sp = setup_package()
    t1 = time.time()
    print( 'Build success: ' + sys.argv[1] +  ' in time (s): ' + str(t1-t0) )
