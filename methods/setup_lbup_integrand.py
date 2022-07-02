from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension("lbup_integrand",
                                    sources=["lbup_integrand.pyx"],
                                    include_dirs=['./']),
                          annotate=True)
)
