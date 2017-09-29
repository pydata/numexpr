#!/usr/bin/env python
###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: BSD
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import shutil
import os, os.path, sys, glob, inspect
from distutils.command.clean import clean
import time
import setuptools


if sys.version_info.major == 3 and sys.version_info.minor < 3:
    raise RuntimeError( 'NumExpr requires Python 3.3 or greater.' )
    
# Increment version for each PyPi release.
major_ver = 3
minor_ver = 0
nano_ver = 1
branch = 'a0'
version = '%d.%d.%d%s' % (major_ver, minor_ver, nano_ver, branch)

# Write __version__.py
with open( 'numexpr3/__version__.py', 'w' ) as fh:
    fh.write( "__version__ = '{}'\n".format(version) )

with open('requirements.txt') as f:
    requirements = f.read().splitlines()    

# Process additional command-line arguments before distutils gets them
NOGEN = False
BENCHMARKING = False
args = sys.argv[1:]
for arg in args:
    if arg == '--nogen':
        NOGEN = True
        sys.argv.remove(arg)
    elif arg == '--bench':
        BENCHMARKING = True
        sys.argv.remove(arg)

# Unique MSVC / GCC / LLVM flags
# TODO: compile detections of AVX2 and SSE2 using cpuinfo
extra_compile_args = {}
extra_libraries = {}
extra_link_args = {}
define_macros = {}

#### GCC (default) ####
# Turned off funroll, doesn't seem to affect speed.
# https://gcc.gnu.org/onlinedocs/gcc-6.4.0/gcc/AArch64-Options.html#AArch64-Options
# For GCC -fopt-info-vec-missed adds a ton of information that's a little too much
# to easily process.
# extra_compile_args['default'] = [  '-fopt-info-vec', '-fopt-info-vec-missed', '-fdiagnostics-color=always' ]
# Try turning off march=native for these complex function errors.
extra_compile_args['default'] = [ '-march=native', '-fopt-info-vec' ]
extra_libraries['default'] = ['m']
extra_link_args['default'] = []
define_macros['default'] = []

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
extra_compile_args['msvc'] =  []
extra_libraries['msvc'] = []
extra_link_args['msvc'] = []
define_macros['msvc'] = []


def run_generator( blocksize=(4096,32), mkl=False, C11=True ):
    # Do not generate new files if the GENERATED files are all newer than
    # interp_generator.py.  This saves recompiling if its not needed.
    GENERATED_files = glob.glob( 'numexpr3/*GENERATED*' ) + glob.glob( 'numexpr3/tests/*GENERATED*' )
    generator_time  = os.path.getmtime( 'code_generators/interp_generator.py' )
    if all( [generator_time < os.path.getmtime(GEN_file) for GEN_file in GENERATED_files] ) \
        and os.path.isfile('numexpr3/lookup.pkl'):

        # Open the 
        import pickle
        with open( 'numexpr3/lookup.pkl', 'rb' ) as lookup:
            OPTABLE = pickle.load( lookup )
            if 'os.name' in OPTABLE and OPTABLE['os.name'] == os.name:
                print( "---===Generation not required===---" )
                return
    # Python 2.7 cannot do relative imports here so we need to use inspect to 
    # modify the PYTHONPATH.
    setup_fileName = inspect.getfile(inspect.currentframe())
    GEN_dir = os.path.join( os.path.abspath( os.path.dirname(setup_fileName)), 'code_generators' )
    sys.path.insert(0,GEN_dir) 
    
    # This no-generated could cause problems if people clone from GitHub and 
    # we insert configuration tricks into the code_generator directory.
    # TODO: It also needs to see if the stubs have been incremented.
    import interp_generator
    print( '---===GENERATING INTERPRETER CODE===---' )
    # Try to auto-detect MKL and C++/11 here.
    # mkl=True 
    
    # TODO: pass a configuration dict instead of a list of parameters.  For 
    # example ICC might be another one...
    interp_generator.generate( blocksize=blocksize, mkl=mkl, C11=C11 )
    

def setup_package():
    metadata = dict(
                      description='Fast numerical expression evaluator for NumPy',
                      author='Robert A. McLeod, David M. Cooke, Francesc Alted, and others',
                      author_email='robbmcleod@gmail.com, faltet@gmail.com',
                      url='https://github.com/pydata/numexpr',
                      license='BSD',
                      packages=['numexpr3'],
                      install_requires=requirements,
                      setup_requires=requirements,
                      classifiers=['Programming Language :: Python :: 3'],
    )
    if (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1]
    in ('--help-commands', 'egg_info', '--version', 'clean'))):

        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Numexpr when Numpy is not yet present in
        # the system.
        # (via https://github.com/abhirk/scikit-learn/blob/master/setup.py)
        from setuptools import setup


        metadata['name']    = 'numexpr3'
        metadata['version'] = version
    else:
        from numpy.distutils.core import setup
        from numpy.distutils.command.build_ext import build_ext as numpy_build_ext


        # Code taken form numpy/distutils/command/build_py.py
        from distutils.command.build_py import build_py as du_build_py
        from numpy.distutils.misc_util import is_string

        class build_py(du_build_py):

            def run(self):
                build_src = self.get_finalized_command('build_src')
                if build_src.py_modules_dict and self.packages is None:
                    self.packages = list(build_src.py_modules_dict.keys())
                du_build_py.run(self)

            def find_package_modules(self, package, package_dir):
                modules = du_build_py.find_package_modules(self, package, package_dir)

                # Find build_src generated *.py files.
                build_src = self.get_finalized_command('build_src')
                modules += build_src.py_modules_dict.get(package, [])

                return modules

            def find_modules(self):
                old_py_modules = self.py_modules[:]
                new_py_modules = list(filter(is_string, self.py_modules))
                self.py_modules[:] = new_py_modules
                modules = du_build_py.find_modules(self)
                self.py_modules[:] = old_py_modules

                return modules

        _DEBUG = False
        def localpath(*args):
            return os.path.abspath(os.pathjoin(*((os.path.dirname(__file__),) + args)))

        def debug(instring):
            if _DEBUG:
                print(' DEBUG: ' + instring)


        def configuration():
            from numpy.distutils.misc_util import Configuration, dict_append
            from numpy.distutils.system_info import system_info, mkl_info

            config = Configuration('numexpr3')

            #try to find configuration for MKL, either from environment or site.cfg
            # if os.path.exists('site.cfg'):
            #     # Probably here we should build custom build_mkl or build_icc 
            #     # commands instead?
            #     mkl_config = mkl_info() 
            #
            #     print( 'Found Intel MKL at: {}'.format( mkl_config.get_mkl_rootdir() ) )
            #     # Check if the user mis-typed a directory
            #     # mkl_include_dir = mkl_config.get_include_dirs()
            #     # mkl_lib_dir = mkl_config.get_lib_dirs()
            #     mkl_config_data = config.get_info('mkl') 
            #
            #     # some version of MKL need to be linked with libgfortran, for this, use
            #     # entries of DEFAULT section in site.cfg
            #     # default_config = system_info() # You don't work
            #     # dict_append(mkl_config_data,
            #     #            libraries=default_config.get_libraries(),
            #     #            library_dirs=default_config.get_lib_dirs())
            #     # RAM: mkl_info() doesn't see to populate get_libraries()...
            #     dict_append(mkl_config_data,
            #                 include_dirs=mkl_config.get_include_dirs(), 
            #                 libraries=mkl_config.get_libraries(),
            #                 library_dirs=mkl_config.get_lib_dirs() )
            # else:
            #     mkl_config_data = {}
                

            #setup information for C extension
            pthread_win = []
            if os.name == 'nt':
                pthread_win = ['numexpr3/win32/pthread.c']
                

            # Maybe we need a newer cpuinfo.py
            extension_config_data = {
                'sources': ['numexpr3/interpreter.cpp',
                            'numexpr3/module.cpp',
                            'numexpr3/numexpr_object.cpp',
                            'numexpr3/benchmarking.hpp'] + pthread_win,
                'depends': ['numexpr3/functions_GENERATED.cpp',
                            'numexpr3/interp_body_GENERATED.cpp',
                            'numexpr3/interp_header_GENERATED.hpp',
                            'numexpr3/module.hpp',
                            'numexpr3/msvc_function_stubs.hpp',
                            'numexpr3/numexpr_config.hpp',
                            'numexpr3/numexpr_object.hpp',
                            'numexpr3/complex_functions.hpp',
                            'numexpr3/string_functions.hpp',
                            ],
                'libraries': [],
                # Compile args come in after the initial args, so I'm not sure 
                # if we can over-ride the optimization flags.  For example
                # Windows is using /Ox instead of /O2.
                'extra_compile_args': [],
            }
            # dict_append(extension_config_data, **mkl_config_data)
            # if 'library_dirs' in mkl_config_data:
            #   library_dirs = ':'.join(mkl_config_data['library_dirs'])
            config.add_extension('interpreter', **extension_config_data)
            
            config.add_data_files( ('','numexpr3/lookup.pkl') )

            config.make_config_py()
            config.add_subpackage('tests', 'numexpr3/tests')

            # Version handling
            config.get_version('numexpr3/__version__.py')
            return config


        class cleaner(clean):

            def run(self):
                # Recursive deletion of build/ directory
                path = localpath('build')
                try:
                    shutil.rmtree(path)
                except Exception:
                    debug('Failed to remove directory %s' % path)
                else:
                    debug('Cleaned up %s' % path)

                # Now, the extension and other files
                try:
                    import imp
                except ImportError:
                    if os.name == 'posix':
                        # RAM: with Python 3 the lib is now versioned.
                        paths = [localpath('numexpr/interpreter.so')]
                    else:
                        paths = [localpath('numexpr/interpreter.pyd')]
                else:
                    paths = []
                    for suffix, _, _ in imp.get_suffixes():
                        if suffix == '.py':
                            continue
                        paths.append(localpath('numexpr3', 'interpreter' + suffix))
                paths.append(localpath('numexpr3/__config__.py'))
                paths.append(localpath('numexpr3/__config__.pyc'))
                for path in paths:
                    try:
                        os.remove(path)
                    except Exception:
                        debug('Failed to clean up file %s' % path)
                    else:
                        debug('Cleaning up %s' % path)

                clean.run(self)

        class build_ext(numpy_build_ext):
            ''' Builds the interpreter shared/dynamic library.'''
                
            def build_extension(self, ext):

                if not NOGEN:
                    # Run Numpy to C generator 
                    run_generator()
                
                compiler = self.compiler.compiler_type
                # print( 'compiler_type is {}'.format(compiler ) )

                if not compiler in extra_compile_args:
                    compiler = 'default'

                if BENCHMARKING:
                    print( "--BENCHMARKING IS ON--" )
                    define_macros[compiler] += [('BENCHMARKING', 1)]

                ext.extra_compile_args = extra_compile_args[compiler]
                ext.extra_libraries = extra_libraries[compiler]
                ext.extra_link_args = extra_link_args[compiler]
                ext.define_macros = define_macros[compiler]

                numpy_build_ext.build_extension(self, ext)

        if setuptools:
            metadata['zip_safe'] = False

        metadata['cmdclass'] = {
            'build_ext': build_ext,
            'clean': cleaner,
            'build_py': build_py,
        }
        metadata['configuration'] = configuration

    setup(**metadata)
    return metadata

if __name__ == '__main__':
    # On cloning from GitHub the lookup dict doesn't exist so we always must run the generator.
    # Otherwise the numpy.distutils check for the file occurs before build_ext is called.
    if not os.path.isfile('numexpr3/lookup.pkl'):
        run_generator()

    t0 = time.time()
    sp = setup_package()
    t1 = time.time()
    print( 'Build success: ' + sys.argv[1] +  ' in time (s): ' + str(t1-t0) )
