project(
    'pyc2ray',
    'c',
    'cpp',
    'fortran',
    version: '0.1',
    meson_version: '>= 1.1',
    default_options: [
        'buildtype=release',
        'warning_level=2',
        'cpp_std=c++14',
        'cuda_std=c++14',
    ],
)

# For lowercase-uppercase stuff
add_project_arguments('-cpp', language: 'fortran')

has_cuda = add_languages('cuda', required: false, native: false)
if has_cuda
    add_project_arguments(
        [
            '-Xcompiler',
            '-fPIC',
            '-rdc',
            'true',
            '-DPERIODIC',
            '-DLOCALRATES',
        ],
        language: 'cuda',
    )
else
    warning('CUDA compiler not found, will not build ASORA library')
endif

# https://numpy.org/doc/stable/f2py/buildtools/meson.html
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Get Numpy and F2PY include dirs
incdir_numpy = run_command(
    py,
    [
        '-c',
        'import os; os.chdir(".."); import numpy; print(numpy.get_include())',
    ],
    check: true,
).stdout().strip()

incdir_f2py = run_command(
    py,
    [
        '-c',
        'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())',
    ],
    check: true,
).stdout().strip()

# Include directories for extension module
inc_np = include_directories(incdir_numpy, incdir_f2py)

subdir('src/c2ray')
subdir('src/asora')
subdir('pyc2ray')
