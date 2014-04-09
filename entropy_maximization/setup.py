import os
from os.path import join

from sklearn._build_utils import get_blas_info


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('entropy_maximization', parent_package, top_path)

    cblas_libs, blas_info = get_blas_info()
    cblas_includes = [join('..', 'src', 'cblas'),
                      numpy.get_include(),
                      blas_info.pop('include_dirs', [])]

    libraries = []
    if os.name == 'posix':
        libraries.append('m')
        cblas_libs.append('m')

    print 'blas_info' , blas_info
    print 'cblas_libs' , cblas_libs
    print 'cblas includes ', cblas_includes

    config.add_extension('weight_vector',
                         sources=['weight_vector.c'],
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         **blas_info)

    config.add_extension('entropy_maximization_sgd_fast',
                         sources=['entropy_maximization_sgd_fast.c'],
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         **blas_info)


    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())