# Fortran source files for C2Ray
c2ray_sources = ['photorates.f90', 'raytracing.f90', 'chemistry.f90']

# F2PY type mapping file for Fortran -> C
f2cmapfile = files('f2c.f2py_f2cmap')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)
if m_dep.found()
    add_project_link_arguments('-lm', language: 'c')
endif

# Construct C-wrapper for Fortran extension module using F2PY
c2ray_wrapper_sources = custom_target(
    'c2ray_wrapper',
    input: c2ray_sources,
    output: ['libc2raymodule.c', 'libc2ray-f2pywrappers2.f90'],
    command: [
        py,
        '-m',
        'numpy.f2py',
        '@INPUT@',
        '-m',
        'libc2ray',
        '--f2cmap',
        f2cmapfile,
        '--build-dir',
        meson.current_build_dir(),
    ],
)

# Find f2py headers
incdir_f2py = run_command(
    py,
    [
        '-c',
        'import os; import numpy.f2py; print(os.path.relpath(numpy.f2py.get_include()))',
    ],
    check: true,
).stdout().strip()
inc_f2py = include_directories(incdir_f2py)

# Build C2Ray extension module
py.extension_module(
    'libc2ray',
    c2ray_wrapper_sources,
    c2ray_sources,
    incdir_f2py / 'fortranobject.c',
    include_directories: inc_f2py,
    dependencies: [py_dep, np_dep],
    install: true,
    subdir: 'pyc2ray/lib',
    fortran_args: '-DUSE_SUBBOX',
    override_options: {'warning_level': '0'}
)
