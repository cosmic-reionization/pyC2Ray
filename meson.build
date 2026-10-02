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

gpu_backend = get_option('gpu-backend')
if gpu_backend == 'cuda'
    has_gpu = add_languages('cuda', required: false, native: false)
    if not has_gpu
        warning('CUDA compiler not found, will not build ASORA library')
    endif
else
    # Meson has no HIP language: ASORA is compiled as C++ by hipcc (see hipnative.ini)
    has_gpu = meson.get_compiler('cpp').has_header(
        'hip/hip_runtime.h',
        args: ['-x', 'hip'],
    )
    if not has_gpu
        warning('C++ compiler is not hipcc, will not build ASORA library')
    endif
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
np_dep = dependency('numpy')

subdir('src/c2ray')
subdir('src/asora')
subdir('pyc2ray')
