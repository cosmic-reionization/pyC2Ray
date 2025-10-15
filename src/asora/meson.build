# CUDA compiler is optional, required to build ASORA
if not has_cuda
    subdir_done()
endif

# CUDA sources for ASORA
asora_sources = ['memory.cu', 'rates.cu', 'raytracing.cu', 'python_module.cu']

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

compute_cap = 'sm_' + compute_cap
nvcc_flags = [
    '--gpu-architecture=' + compute_cap,
    '-rdc',
    'true',
    '-DPERIODIC',
    '-DLOCALRATES',
]
nvlink_flags = ['-arch', compute_cap]

rocm_root = get_option('rocm-root')
hip_headers = include_directories(rocm_root / 'include')

# Build ASORA module with nvcc
py.extension_module(
    'libasora',
    asora_sources,
    dependencies: [py_dep, np_dep],
    include_directories: hip_headers,
    install: true,
    subdir: 'pyc2ray/lib',
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)
