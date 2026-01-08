# CUDA compiler is optional, required to build ASORA
if not has_cuda
    subdir_done()
endif

# Get CUDA architecture to be used for compilation
compute_cap = get_option('gpu-architecture')
if compute_cap == ''
    message('CUDA compute capability was not specified by the user')
    r_compute_cap = run_command(
        'nvidia-smi',
        '--query-gpu=compute_cap',
        '--format=csv,noheader',
        check: false,
    )
    if r_compute_cap.returncode() != 0
        warning('nvidia-smi not detected; defaulting to compute capability 6.0')
        compute_cap = '60'
    else
        cc = r_compute_cap.stdout().strip()
        message(f'Local compute capability is determined to be @cc@')
        compute_cap = cc[0] + '0'
    endif
endif

# Target only one GPU
compute_cap = '-arch=sm_' + compute_cap

nvcc_flags = [
    compute_cap,
    '-rdc',
    'true',
    '-DPERIODIC',
    '-DLOCALRATES',
    '-use_fast_math',
]
nvlink_flags = [compute_cap]

# CUDA sources for ASORA
asora_sources = ['memory.cpp', 'rates.cu', 'raytracing.cu', 'utils.cu']
cuda_dep = dependency('cuda', version: '>=12', static: false)
asora_core = static_library(
    'asora_core',
    asora_sources,
    dependencies: cuda_dep,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)

# Build ASORA module with nvcc
py.extension_module(
    'libasora',
    'python_module.cu',
    dependencies: [py_dep, np_dep, cuda_dep],
    install: true,
    subdir: 'pyc2ray/lib',
    link_with: asora_core,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)

# Build ASORA test module with nvcc
py.extension_module(
    'libasoratest',
    ['tests.cu', 'python_test_module.cu'],
    dependencies: [py_dep, np_dep, cuda_dep],
    install: true,
    subdir: 'pyc2ray/lib',
    link_with: asora_core,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)
