#!/usr/bin/env python
import shutil
import os
import os.path as op
from distutils.command.clean import clean
from numpy.distutils.command.build_ext import build_ext as numpy_build_ext

try:
    import setuptools
except ImportError:
    setuptools = None
extra_setup_opts = {}
if setuptools:
    extra_setup_opts['zip_safe'] = False

DEBUG = False

def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))

def debug(instring):
    if DEBUG:
        print " DEBUG: "+instring


def configuration():
    from numpy.distutils.misc_util import Configuration, dict_append
    from numpy.distutils.system_info import system_info

    config = Configuration('numexpr')

    #try to find configuration for MKL, either from environment or site.cfg
    if op.exists('site.cfg'):
        mkl_config_data = config.get_info('mkl')
        # some version of MKL need to be linked with libgfortran, for this, use
        # entries of DEFAULT section in site.cfg
        default_config = system_info()
        dict_append(mkl_config_data,
                    libraries = default_config.get_libraries(),
                    library_dirs = default_config.get_lib_dirs() )
    else:
        mkl_config_data = {}

    #setup information for C extension
    if os.name == 'nt':
        pthread_win = ['numexpr/win32/pthread.c']
    else:
        pthread_win = []
    extension_config_data = {
        'sources': ['numexpr/interpreter.c'] + pthread_win,
        'depends': ['numexpr/interp_body.c',
                    'numexpr/complex_functions.inc',
                    'numexpr/msvc_function_stubs.inc'],
        'extra_compile_args': ['-funroll-all-loops',],
        }
    dict_append(extension_config_data, **mkl_config_data)
    if 'library_dirs' in mkl_config_data:
        library_dirs = ':'.join(mkl_config_data['library_dirs'])
        rpath_link = '-Xlinker --rpath -Xlinker %s' % library_dirs
        extension_config_data['extra_link_args'] = [rpath_link]
    config.add_extension('interpreter', **extension_config_data)

    config.make_config_py()
    config.add_subpackage('tests', 'numexpr/tests')

    #version handling
    config.make_svn_version_py()
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
        if os.name == 'posix':
            paths = [localpath("numexpr/interpreter.so")]
        else:
            paths = [localpath("numexpr/interpreter.pyd")]
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


def setup_package():
    import os
    from numpy.distutils.core import setup

    extra_setup_opts['cmdclass'] = {'build_ext': build_ext,
                                    'clean': cleaner,
                                    }
    setup(#name='numexpr',  # name already set in numpy.distutils
          description='Fast numerical expression evaluator for NumPy',
          author='David M. Cooke, Francesc Alted and others',
          author_email='david.m.cooke@gmail.com, faltet@pytables.org',
          url='http://code.google.com/p/numexpr/',
          license = 'MIT',
          packages = ['numexpr'],
          configuration = configuration,
          **extra_setup_opts
          )

class build_ext(numpy_build_ext):
    def build_extension(self, ext):
        # at this point we know what the C compiler is.
        c = self.compiler
        old_compile_options = None
        # For MS Visual C, we use /O1 instead of the default /Ox,
        # as /Ox takes a long time (~5 mins) to compile.
        # The speed of the code isn't noticeably different.
        if c.compiler_type == 'msvc':
            if not c.initialized:
                c.initialize()
            old_compile_options = c.compile_options[:]
            if '/Ox' in c.compile_options:
                c.compile_options.remove('/Ox')
            c.compile_options.append('/O1')
            ext.extra_compile_args = []
        numpy_build_ext.build_extension(self, ext)
        if old_compile_options is not None:
            self.compiler.compile_options = old_compile_options

if __name__ == '__main__':
    setup_package()

