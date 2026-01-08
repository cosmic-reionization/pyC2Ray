project(
    'pyc2ray',
    'c',
    'cpp',
    'fortran',
    version: '0.1',
    meson_version: '>= 1.4.0',
    default_options: [
        'buildtype=release',
        'warning_level=3',
        'cpp_std=c++20',
        'cuda_std=c++20',
    ],
)

# For lowercase-uppercase stuff
add_project_arguments('-cpp', language: 'fortran')

has_cuda = add_languages('cuda', required: false, native: false)
if not has_cuda
    warning('CUDA compiler not found, will not build ASORA library')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
np_dep = dependency('numpy', version: '>=2', method: 'config-tool')

subdir('src/c2ray')
subdir('src/asora')
subdir('pyc2ray')
