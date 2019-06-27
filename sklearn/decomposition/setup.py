import os
import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("decomposition", parent_package, top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension("_online_lda",
                         sources=["_online_lda.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)

    config.add_extension('cdnmf_fast',
                         sources=['cdnmf_fast.pyx'],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=['-fopenmp', '-O3'],
                         extra_link_args=['-fopenmp', '-O3'],
                         libraries=libraries)

    config.add_subpackage("tests")

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
