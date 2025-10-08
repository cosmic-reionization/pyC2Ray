from numpy.distutils.core import Extension, setup

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# List of Fortran source files
fortran_sources = [
    'src/c2ray/photorates.f90',
    'src/c2ray/raytracing.f90',
    'src/c2ray/chemistry.f90',
]

# List of CUDA source files
cuda_sources = [
    'src/asora/memory.cu',
    'src/asora/rates.cu',
    'src/asora/raytracing.cu',
    'src/asora/python_module.cu',
]

# Fortran extension module definition
fortran_extension_module = Extension(
    name='c2ray.libc2ray',
    sources=fortran_sources,
    extra_compile_args=['-E'],  # Use -E for preprocessing
)

# CUDA extension module definition
cuda_extension_module = Extension(
    name='asora.libasora',
    sources=cuda_sources,
    extra_compile_args=['-std=c++14', '-O2', '-Xcompiler', '-fPIC', '-DPERIODIC', '-DLOCALRATES', '--gpu-architecture=sm_60'],
)

setup(
    name='pyc2ray',
    version='0.1',
    ext_modules=[fortran_extension_module, cuda_extension_module],
    install_requires=install_requires,
    packages=['pyc2ray'],  # Include 'pyc2ray.c2ray' in packages
    package_dir={'pyc2ray': 'pyc2ray'},  # Specify the package directory
)
