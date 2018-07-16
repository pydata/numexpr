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
import sys, os, os.path as op, io
from distutils.command.clean import clean

if sys.version_info < (2, 6):
    raise RuntimeError("must use python 2.6 or greater")

try:
    import setuptools
except ImportError:
    setuptools = None

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with io.open('README.rst', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Fetch the version for numexpr (will be put in variable `version`)
with open(os.path.join('numexpr', 'version.py')) as f:
    exec(f.read())

def setup_package():
    metadata = dict(
                      description='Fast numerical expression evaluator for NumPy',
                      author='David M. Cooke, Francesc Alted and others',
                      author_email='david.m.cooke@gmail.com, faltet@gmail.com',
                      url='https://github.com/pydata/numexpr',
                      long_description=LONG_DESCRIPTION,
                      license='MIT',
                      packages=['numexpr'],
                      install_requires=requirements,
                      setup_requires=requirements
    )
    if (len(sys.argv) >= 2 and
        ('--help' in sys.argv[1:] or
         (sys.argv[1] in (
             '--help-commands', 'egg_info', '--version', 'clean', '--name')))):

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

        metadata['name']    = 'numexpr'
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
                    modules = old_build_py.find_package_modules(
                        self, package, package_dir)

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

            config = Configuration('numexpr')

            #try to find configuration for MKL, either from environment or site.cfg
            if op.exists('site.cfg'):
                mkl_config_data = config.get_info('mkl')
                # Some version of MKL needs to be linked with libgfortran.
                # For this, use entries of DEFAULT section in site.cfg.
                default_config = system_info()
                dict_append(mkl_config_data,
                            libraries=default_config.get_libraries(),
                            library_dirs=default_config.get_lib_dirs())
            else:
                mkl_config_data = {}

            # setup information for C extension
            if os.name == 'nt':
                pthread_win = ['numexpr/win32/pthread.c']
            else:
                pthread_win = []
            extension_config_data = {
                'sources': ['numexpr/interpreter.cpp',
                            'numexpr/module.cpp',
                            'numexpr/numexpr_object.cpp'] + pthread_win,
                'depends': ['numexpr/interp_body.cpp',
                            'numexpr/complex_functions.hpp',
                            'numexpr/interpreter.hpp',
                            'numexpr/module.hpp',
                            'numexpr/msvc_function_stubs.hpp',
                            'numexpr/numexpr_config.hpp',
                            'numexpr/numexpr_object.hpp'],
                'libraries': ['m'],
                'extra_compile_args': ['-funroll-all-loops', ],
            }
            dict_append(extension_config_data, **mkl_config_data)
            if 'library_dirs' in mkl_config_data:
                library_dirs = ':'.join(mkl_config_data['library_dirs'])
            config.add_extension('interpreter', **extension_config_data)
            config.set_options(quiet=True)

            config.make_config_py()
            config.add_subpackage('tests', 'numexpr/tests')

            #version handling
            config.get_version('numexpr/version.py')
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
                        paths.append(localpath("numexpr", "interpreter" + suffix))
                paths.append(localpath("numexpr/__config__.py"))
                paths.append(localpath("numexpr/__config__.pyc"))
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
                if self.compiler.compiler_type == 'msvc' or self.compiler.compiler_type == 'intelemw':
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
    setup_package()
