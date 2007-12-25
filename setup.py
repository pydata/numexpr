try:
    import setuptools
    have_setuptools = True
except ImportError:
    have_setuptools = False
from numpy.distutils.core import setup, Extension

extra_setup_opts = {}
if have_setuptools:
    extra_setup_opts['zip_safe'] = False
    extra_setup_opts['install_requires'] = ['numpy >= 1.0']
    extra_setup_opts['test_suite'] = 'nose.collector'

setup(name='numexpr',
      version='0.8',
      description='Fast numerical expression evaluator for NumPy',
      author='David M. Cooke',
      author_email='david.m.cooke@gmail.com',
      url='http://code.google.com/p/numexpr/',
      packages=['numexpr', 'numexpr.tests'],
      ext_modules=[Extension('numexpr.interpreter',
                             sources=['numexpr/interpreter.c'],
                             depends = ['numexpr/interp_body.c',
                                        'numexpr/complex_functions.inc'],
                             extra_compile_args=['-O2', '-funroll-all-loops'],
                            )],
      **extra_setup_opts
    )
