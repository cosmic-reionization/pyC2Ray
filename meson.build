project('pyc2ray','c','cpp',
    version : '0.1',
    default_options : [
        'warning_level=2',
        'cpp_std=c++14',
        'cuda_std=c++14'])

# Define global options
add_languages('fortran')
add_global_arguments('-cpp', language : 'fortran') # For lowercase-uppercase stuff

# CUDA compiler is optional, required to build ASORA
has_cuda = add_languages('cuda',required:false)
if not has_cuda
    warning('CUDA compiler not found, will not build ASORA library')
else
    add_global_arguments(['-O2','-Xcompiler','-fPIC','-rdc','true'], language : 'cuda')
    add_global_arguments(['-D PERIODIC','-D LOCALRATES'], language : 'cuda')
endif

# Fortran source files for C2Ray
c2ray_sources = [
    'src/c2ray/photorates.f90',
    'src/c2ray/raytracing.f90',
    'src/c2ray/chemistry.f90',
]

# Python sources
pyc2ray_sources = [
    'pyc2ray/__init__.py',
    'pyc2ray/asora_core.py',
    'pyc2ray/c2ray_base.py',
    'pyc2ray/c2ray_cubep3m.py',
    'pyc2ray/c2ray_test.py',
    'pyc2ray/chemistry.py',
    'pyc2ray/evolve.py',
    'pyc2ray/load_extensions.py',
    'pyc2ray/radiation.py',
    'pyc2ray/raytracing.py'
]

pyc2ray_utils_sources = [
    'pyc2ray/utils/__init__.py',
    'pyc2ray/utils/logutils.py',
    'pyc2ray/utils/other_utils.py',
    'pyc2ray/utils/paramutils.py',
    'pyc2ray/utils/source_converter.py',
    'pyc2ray/utils/sourceutils.py'
]

# CUDA sources for ASORA
asora_sources = [
    'src/asora/memory.cu',
    'src/asora/rates.cu',
    'src/asora/raytracing.cu',
    'src/asora/python_module.cu',
]

# ==================================
# PART 1: BUILD WRAPPED C2RAY MODULE
# ==================================

# https://numpy.org/doc/stable/f2py/buildtools/meson.html
py_mod = import('python')
py3 = py_mod.find_installation('python3')
py3_dep = py3.dependency()
message(py3.path())
message(py3.get_install_dir())

# Get Numpy and F2PY include dirs
incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# F2PY type mapping file for Fortran -> C
f2cmapfile = files('src/c2ray/f2c.f2py_f2cmap')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif


# Construct C-wrapper for Fortran extension module using F2PY
c2ray_wrapper_source = custom_target('libc2raymodule.c',
    input : c2ray_sources,
    output : ['libc2raymodule.c','libc2ray-f2pywrappers2.f90'],
    command : [py3,'-m','numpy.f2py','@INPUT@','-m','libc2ray','--f2cmap',f2cmapfile])

# Include directories for extension module
inc_np = include_directories(incdir_numpy,incdir_f2py)


# Build C2Ray extension module
py3.extension_module('libc2ray',
    c2ray_wrapper_source,
    c2ray_sources,
    incdir_f2py+'/fortranobject.c',
    include_directories: inc_np,
    dependencies : py3_dep,
    install : true,
    subdir : 'pyc2ray/lib',
    fortran_args : '-DUSE_SUBBOX')


# ===================================
# PART 2: BUILD OPTIONAL ASORA MODULE
# ===================================
if has_cuda
    # Get CUDA architecture to be used for compilation
    gpu_arch = get_option('gpu-architecture')

    # Build ASORA module with nvcc
    py3.extension_module('libasora',
        asora_sources,
        include_directories: inc_np,
        dependencies: py3_dep,
        install : true,
        subdir : 'pyc2ray/lib',
        cuda_args : '--gpu-architecture='+gpu_arch)
endif


# ==============================
# PART 3: INSTALL PYTHON SOURCES
# ==============================
py3.install_sources(
  pyc2ray_sources,
  subdir: 'pyc2ray',
  pure: false
)

py3.install_sources(
  pyc2ray_utils_sources,
  subdir: 'pyc2ray/utils',
  pure: false
)