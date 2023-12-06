# from setuptools import setup
from numpy.distutils.core import Extension, setup

# List of Fortran source files
fortran_sources = [
    'src/c2ray/photorates.f90',
    'src/c2ray/raytracing.f90',
    'src/c2ray/chemistry.f90',
]

# Extension module definition
extension_module = Extension(
    name='c2ray.libc2ray',
    sources=fortran_sources,
    extra_compile_args=['-E'],  # Use -E for preprocessing
)

setup(
    name='pyc2ray',
    version='0.1',
    ext_modules=[extension_module],
    install_requires=[
        'numpy',
        # Add other dependencies here
    ],
    packages=['pyc2ray'],  # List your Python packages here
    package_dir={'pyc2ray': 'pyc2ray'},  # Specify the package directory
)
