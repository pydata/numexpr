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

"""
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
"""

if sys.version_info < (2, 6):
    raise RuntimeError("must use python 2.6 or greater")

try:
    import setuptools
except ImportError:
    setuptools = None

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Fetch the version for numexpr (will be put in variable `version`)
with open(os.path.join('numexpr3', 'version.py')) as f:
    exec(f.read())
    
    
# RAM: NEW compile-time definition of #define BLOCK_SIZE variables
blocks_text = """/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifdef USE_VML
// The values below have been tuned for a nowadays Core2 processor 
// Note: with VML functions a larger block size (e.g. 4096) allows to make use
// of the automatic multithreading capabilities of the VML library 
#define BLOCK_SIZE1 4096
#define BLOCK_SIZE2 32
#else
// The values below have been tuned for a nowadays Core2 processor
// Note: without VML available a smaller block size is best, specially
// for the strided and unaligned cases.  Recent implementation of
// multithreading make it clear that larger block sizes benefit
// performance (although it seems like we don't need very large sizes
// like VML yet). 
#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 16
#endif
"""
with open( "numexpr3/blocks.hpp", 'wb' ) as fh:
     fh.writelines( blocks_text )

def setup_package():
    metadata = dict(
                      description='Fast numerical expression evaluator for NumPy',
                      author='David M. Cooke, Francesc Alted and others',
                      author_email='david.m.cooke@gmail.com, faltet@gmail.com',
                      url='https://github.com/pydata/numexpr',
                      license='MIT',
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
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['name']    = 'numexpr3'
        metadata['version'] = version
    else:
        from numpy.distutils.core import setup
        from numpy.distutils.command.build_ext import build_ext as numpy_build_ext

        try:  # Python 3
            # Code taken form numpy/distutils/command/build_py.py
            # XXX: update LICENSES
            from distutils.command.build_py import build_py_2to3 as old_build_py
            from numpy.distutils.misc_util import is_string

            class build_py(old_build_py):

                def run(self):
                    build_src = self.get_finalized_command('build_src')
                    if build_src.py_modules_dict and self.packages is None:
                        self.packages = list(build_src.py_modules_dict.keys())
                    old_build_py.run(self)

                def find_package_modules(self, package, package_dir):
                    modules = old_build_py.find_package_modules(self, package, package_dir)

                    # Find build_src generated *.py files.
                    build_src = self.get_finalized_command('build_src')
                    modules += build_src.py_modules_dict.get(package, [])

                    return modules

                def find_modules(self):
                    old_py_modules = self.py_modules[:]
                    new_py_modules = list(filter(is_string, self.py_modules))
                    self.py_modules[:] = new_py_modules
                    modules = old_build_py.find_modules(self)
                    self.py_modules[:] = old_py_modules

                    return modules

        except ImportError:  # Python 2
            from numpy.distutils.command.build_py import build_py

        DEBUG = False

        def localpath(*args):
            return op.abspath(op.join(*((op.dirname(__file__),) + args)))

        def debug(instring):
            if DEBUG:
                print(" DEBUG: " + instring)


        def configuration():
            from numpy.distutils.misc_util import Configuration, dict_append
            from numpy.distutils.system_info import system_info

            config = Configuration('numexpr3')

            #try to find configuration for MKL, either from environment or site.cfg
            if op.exists('site.cfg'):
                mkl_config_data = config.get_info('mkl')
                # some version of MKL need to be linked with libgfortran, for this, use
                # entries of DEFAULT section in site.cfg
                default_config = system_info()
                dict_append(mkl_config_data,
                            libraries=default_config.get_libraries(),
                            library_dirs=default_config.get_lib_dirs())
            else:
                mkl_config_data = {}

            #setup information for C extension
            if os.name == 'nt':
                pthread_win = ['numexpr3/win32/pthread.c']
            else:
                pthread_win = []
            extension_config_data = {
                'sources': ['numexpr3/interpreter.cpp',
                            'numexpr3/module.cpp',
                            'numexpr3/numexpr_object.cpp'] + pthread_win,
                'depends': ['numexpr3/interp_body.cpp',
                            'numexpr3/complex_functions.hpp',
                            'numexpr3/interpreter.hpp',
                            'numexpr3/module.hpp',
                            'numexpr3/msvc_function_stubs.hpp',
                            'numexpr3/numexpr_config.hpp',
                            'numexpr3/numexpr_object.hpp',
                            'numexpr3/opcodes.hpp'],
                'libraries': ['m'],
                'extra_compile_args': ['-funroll-all-loops', ],
            }
            dict_append(extension_config_data, **mkl_config_data)
            if 'library_dirs' in mkl_config_data:
                library_dirs = ':'.join(mkl_config_data['library_dirs'])
            config.add_extension('interpreter', **extension_config_data)

            config.make_config_py()
            config.add_subpackage('tests', 'numexpr3/tests')

            #version handling
            config.get_version('numexpr3/version.py')
            return config


        class cleaner(clean):

            def run(self):
                # Recursive deletion of build/ directory
                path = localpath("build")
                try:
                    shutil.rmtree(path)
                except Exception:
                    debug("Failed to remove directory %s" % path)
                else:
                    debug("Cleaned up %s" % path)

                # Now, the extension and other files
                try:
                    import imp
                except ImportError:
                    if os.name == 'posix':
                        paths = [localpath("numexpr/interpreter.so")]
                    else:
                        paths = [localpath("numexpr/interpreter.pyd")]
                else:
                    paths = []
                    for suffix, _, _ in imp.get_suffixes():
                        if suffix == '.py':
                            continue
                        paths.append(localpath("numexpr3", "interpreter" + suffix))
                paths.append(localpath("numexpr3/__config__.py"))
                paths.append(localpath("numexpr3/__config__.pyc"))
                for path in paths:
                    try:
                        os.remove(path)
                    except Exception:
                        debug("Failed to clean up file %s" % path)
                    else:
                        debug("Cleaning up %s" % path)

                clean.run(self)

        class build_ext(numpy_build_ext):
            def build_extension(self, ext):
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
        }
        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    t0 = time.time()
    setup_package()
    t1 = time.time()
    print( "No error on " + sys.argv[1] +  " in time (s): " + str(t1-t0) )
