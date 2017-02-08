#!/usr/bin/env python
###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import shutil
import os
import sys
import os.path as op
from distutils.command.clean import clean
import time

'''
 NOTES FOR WINDOWS:
 For Python 2.7, you must have the x64 environment variables set correctly...

from:
    http://scikit-learn.org/stable/developers/advanced_installation.html#windows
for the 64-bit architecture, you either need the full visual studio or the free windows sdks that 
can be downloaded from the links below.
the windows sdks include the msvc compilers both for 32 and 64-bit architectures. they come as a 
grmsdkx_en_dvd.iso file that can be mounted as a new drive with a setup.exe installer in it.

for python 2 you need sdk v7.0: ms windows sdk for windows 7 and .net framework 3.5 sp1

for python 3 you need sdk v7.1: ms windows sdk for windows 7 and .net framework 4

both sdks can be installed in parallel on the same host. to use the windows sdks, you need to setup 
the environment of a cmd console launched with the following flags (at least for sdk v7.0):
cmd /e:on /v:on /k
then configure the build environment with (FOR PYTHON 2.7):

set distutils_use_sdk=1
set mssdk=1
"c:\program files\microsoft sdks\windows\v7.0\setup\windowssdkver.exe" -q -version:v7.0
"c:\program files\microsoft sdks\windows\v7.0\bin\setenv.cmd" /x64 /release

finally you can build scikit-learn in the same cmd console:

python setup.py install

replace v7.0 by the v7.1 in the above commands to do the same for python 3 instead of python 2.
replace /x64 by /x86 to build for 32-bit python instead of 64-bit python.

Ignore the "Missing compiler_cxx fix for MSVCCompiler" error message.

You also need the .NET Framework 3.5 SP1 installed for Python 2.7
'''

if sys.version_info < (2, 7):
    raise RuntimeError( 'NumExpr3 requires Python 2.7 or greater.' )
    
if sys.version_info.major == 3 and sys.version_info.minor < 3:
    raise RuntimeError( 'NumExpr requires Python 3.3 or greater.' )
    
import setuptools


# Increment version for each PyPi release.
major_ver = 3
minor_ver = 0
nano_ver = 0
branch = 'a0'
version = '%d.%d.%d%s' % (major_ver, minor_ver, nano_ver, branch)

# Write __version__.py
with open( 'numexpr3/__version__.py', 'w' ) as fh:
    fh.write( "__version__ = '{}'\n".format(version) )

with open('requirements.txt') as f:
    requirements = f.read().splitlines()    
    
# TODO: unique Windows / Linux / OSX flags
# TODO: compile detections of AVX2 and SSE2 using numpy.distutils
# Turned off funroll, doesn't seem to affect speed.
# https://gcc.gnu.org/onlinedocs/gcc-5.4.0/gcc/Optimize-Options.html
#compile_args = {
#    'gcc': [ # '-funroll-loops',
#            '-fdiagnostics-color=always', 
#             '-fopt-info-vec',
#            ],
#    'msvc':[  '/Qvec-report:2', 
#            ],
#}


def run_generator( blocksize=(4096,32), mkl=False, C11=True ):
    from code_generators import interp_generator
    print( '=====GENERATING INTERPRETER CODE=====' )
    # Try to auto-detect MKL and C++/11 here.
    # mkl=True 
    
    # TODO: pass a configuration dict instead of a list of parameters.  For 
    # example ICC might be another one...
    interp_generator.generate( blocksize=blocksize, mkl=mkl, C11=C11 )
    
#def generate( body_stub='interp_body_stub.cpp', header_stub='interp_header_stub.hpp', 
#             blocksize=(4096,32), bounds_check=True, mkl=False ):
    
def setup_package():
    metadata = dict(
                      description='Fast numerical expression evaluator for NumPy',
                      author='Robert A. McLeod, David M. Cooke, Francesc Alted, and others',
                      author_email='robbmcleod@gmail.com, faltet@gmail.com',
                      url='https://github.com/pydata/numexpr',
                      license='BSD',
                      packages=['numexpr3'],
                      install_requires=requirements,
                      setup_requires=requirements
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

        try:  # Python 3
            # Code taken form numpy/distutils/command/build_py.py
            from distutils.command.build_py import build_by as du_build_py
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

        except ImportError:  # Python 2
            from numpy.distutils.command.build_py import build_py

        DEBUG = False

        def localpath(*args):
            return op.abspath(op.join(*((op.dirname(__file__),) + args)))

        def debug(instring):
            if DEBUG:
                print(' DEBUG: ' + instring)


        def configuration():
            from numpy.distutils.misc_util import Configuration, dict_append
            from numpy.distutils.system_info import system_info, mkl_info

            config = Configuration('numexpr3')

            #try to find configuration for MKL, either from environment or site.cfg
            if op.exists('site.cfg'):
                # RAM: argh, distutils...
                # Probably here we should build custom build_mkl or build_icc 
                # commands instead?
                mkl_config = mkl_info() 

                
                print( 'Found Intel MKL at: {}'.format( mkl_config.get_mkl_rootdir() ) )
                # Check if the user mis-typed a directory
#                mkl_include_dir = mkl_config.get_include_dirs()
#                mkl_lib_dir = mkl_config.get_lib_dirs()
                    
                mkl_config_data = config.get_info('mkl') 
                
                # some version of MKL need to be linked with libgfortran, for this, use
                # entries of DEFAULT section in site.cfg
                # default_config = system_info() # You don't work
#                dict_append(mkl_config_data,
#                            libraries=default_config.get_libraries(),
#                            library_dirs=default_config.get_lib_dirs())
                # RAM: mkl_info() doesn't see to populate get_libraries()...
                dict_append(mkl_config_data,
                            include_dirs=mkl_config.get_include_dirs(), 
                            libraries=mkl_config.get_libraries(),
                            library_dirs=mkl_config.get_lib_dirs() )
            else:
                mkl_config_data = {}
                
            print( 'DEBUG: mkl_config_data: {}'.format(mkl_config_data) )

            #setup information for C extension
            if os.name == 'nt':
                pthread_win = ['numexpr3/win32/pthread.c']
            else:
                pthread_win = []
            # TODO: add support for -msse2 and -mavx2 flags via auto-detection?
            # Maybe we need a newer cpuinfo.py
            extension_config_data = {
                'sources': ['numexpr3/interpreter.cpp',
                            'numexpr3/module.cpp',
                            'numexpr3/numexpr_object.cpp'] + pthread_win,
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
                'libraries': ['m'],
                # Now how to get setuptools to actually pass arguments to MSVC?
                'extra_compile_args': [ # '-funroll-loops',
                    '-fdiagnostics-color=always', 
                     '-fopt-info-vec',
                    ],
            }
            dict_append(extension_config_data, **mkl_config_data)
            if 'library_dirs' in mkl_config_data:
                library_dirs = ':'.join(mkl_config_data['library_dirs'])
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

        class no_gen(numpy_build_ext):
            ''' Identical to build_ext but without generation, for debugging.'''

            def build_extension(self, ext):
            
                # at this point we know what the C compiler is.
                if self.compiler.compiler_type == 'msvc':
                    ext.extra_compile_args = []
                    # also remove extra linker arguments msvc doesn't understand
                    ext.extra_link_args = []
                    # also remove gcc math library
                    ext.libraries.remove('m')
                numpy_build_ext.build_extension(self, ext)
                
        class build_ext(numpy_build_ext):
            ''' Builds the interpreter shared/dynamic library.'''
                
            
            def build_extension(self, ext):
                # Run Numpy to C generator 
                run_generator()
                    
                # at this point we know what the C compiler is.
                if self.compiler.compiler_type == 'msvc':
                    ext.extra_compile_args = []
                    # also remove extra linker arguments msvc doesn't understand
                    ext.extra_link_args = []
                    # also remove gcc math library
                    ext.libraries.remove('m')
                numpy_build_ext.build_extension(self, ext)

        if setuptools:
            metadata['zip_safe'] = False

        metadata['cmdclass'] = {
            'build_ext': build_ext,
            'clean': cleaner,
            'build_py': build_py,
            'no_gen': no_gen,
        }
        metadata['configuration'] = configuration

    setup(**metadata)
    return metadata

if __name__ == '__main__':
    t0 = time.time()
    sp = setup_package()
    t1 = time.time()
    print( 'Build success: ' + sys.argv[1] +  ' in time (s): ' + str(t1-t0) )
