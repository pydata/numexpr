try:
    import setuptools
except ImportError:
    setuptools = None
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext as old_build_ext

import numpy

extra_setup_opts = {}
if setuptools:
    extra_setup_opts['zip_safe'] = False
    extra_setup_opts['test_suite'] = 'nose.collector'

interpreter_ext = Extension('numexpr.interpreter',
                            sources=['numexpr/interpreter.c'],
                            depends = ['numexpr/interp_body.c',
                                       'numexpr/complex_functions.inc'],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=['-funroll-all-loops']
                            )

class build_ext(old_build_ext):
    def build_extension(self, ext):
        # at this point we know what the C compiler is.
        c = self.compiler
        old_compile_options = None
        if ext is interpreter_ext:
            # For MS Visual C, we use /O1 instead of the default /Ox,
            # as /Ox takes a long time (~20 mins) to compile.
            # The speed of the code isn't noticeably different.
            if c.compiler_type == 'msvc':
                if not c.initialized:
                    c.initialize()
                old_compile_options = c.compile_options[:]
                if '/Ox' in c.compile_options:
                    c.compile_options.remove('/Ox')
                c.compile_options.append('/O1')
                ext.extra_compile_args = []
        old_build_ext.build_extension(self, ext)
        if old_compile_options is not None:
            self.compiler.compile_options = old_compile_options

extra_setup_opts['cmdclass'] = {'build_ext': build_ext}

setup(name='numexpr',
      version='1.1',
      description='Fast numerical expression evaluator for NumPy',
      author='David M. Cooke, Tim Hochberg, Francesc Alted, Ivan Vilata',
      author_email='david.m.cooke@gmail.com, faltet@pytables.org',
      url='http://code.google.com/p/numexpr/',
      packages=['numexpr', 'numexpr.tests'],
      ext_modules=[interpreter_ext],
      **extra_setup_opts
    )
