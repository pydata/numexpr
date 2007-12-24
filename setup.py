from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('numexpr', parent_package, top_path)
    config.add_extension('numexpr.interpreter',
                         sources = ['numexpr/interpreter.c'],
                         depends = ['numexpr/interp_body.c',
                                    'numexpr/complex_functions.inc'],
                         extra_compile_args=['-O2', '-funroll-all-loops'],
                         )
    config.add_data_dir('numexpr/tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(author='David M. Cooke',
          author_email='cookedm@physics.mcmaster.ca',
          version='0.8',
          zip_safe=False,
          **configuration(top_path='').todict())
