# GPU compiler is optional, required to build ASORA
if not has_gpu
    subdir_done()
endif

# GPU sources for ASORA
asora_sources = ['memory.cu', 'rates.cu', 'raytracing.cu', 'utils.cu']
module_sources = ['python_module.cu']
test_sources = ['tests.cu', 'python_test_module.cu']

compute_cap = get_option('gpu-architecture')

if gpu_backend == 'cuda'
    # Get CUDA architecture to be used for compilation
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
    ]
    nvlink_flags = [compute_cap]

    gpu_cuda_args = nvcc_flags
    gpu_cpp_args = []
    gpu_link_args = nvlink_flags
else
    # Get AMD GPU architecture to be used for compilation
    if compute_cap == ''
        message('AMD GPU architecture was not specified by the user')
        r_compute_cap = run_command('rocm_agent_enumerator', check: false)
        gfx_targets = []
        if r_compute_cap.returncode() == 0
            foreach target : r_compute_cap.stdout().strip().split('\n')
                # gfx000 is the CPU agent
                if target.strip() not in ['', 'gfx000']
                    gfx_targets += target.strip()
                endif
            endforeach
        endif
        if gfx_targets.length() == 0
            warning('No AMD GPU detected; defaulting to gfx90a (MI250X)')
            compute_cap = 'gfx90a'
        else
            compute_cap = gfx_targets[0]
            message(f'Local GPU architecture is determined to be @compute_cap@')
        endif
    endif

    # Target only one GPU
    compute_cap = '--offload-arch=' + compute_cap

    gpu_cuda_args = []
    gpu_cpp_args = [
        '-x',
        'hip',
        compute_cap,
        '-fgpu-rdc',
        '-DPERIODIC',
        '-DLOCALRATES',
    ]
    gpu_link_args = [compute_cap, '-fgpu-rdc', '--hip-link']

    # Meson picks the compiler from the file suffix: copy the .cu sources to
    # .cpp in the build directory so that they are compiled by hipcc
    fs = import('fs')
    hip_asora_sources = []
    foreach src : asora_sources
        hip_asora_sources += fs.copyfile(src, src.replace('.cu', '.cpp'))
    endforeach
    hip_module_sources = []
    foreach src : module_sources
        hip_module_sources += fs.copyfile(src, src.replace('.cu', '.cpp'))
    endforeach
    hip_test_sources = []
    foreach src : test_sources
        hip_test_sources += fs.copyfile(src, src.replace('.cu', '.cpp'))
    endforeach
    asora_sources = hip_asora_sources
    module_sources = hip_module_sources
    test_sources = hip_test_sources
endif

asora_core = static_library(
    'asora_core',
    asora_sources,
    cuda_args: gpu_cuda_args,
    cpp_args: gpu_cpp_args,
    link_args: gpu_link_args,
)

# Build ASORA module
py.extension_module(
    'libasora',
    module_sources,
    dependencies: [py_dep, np_dep],
    install: true,
    subdir: 'pyc2ray/lib',
    link_with: asora_core,
    cuda_args: gpu_cuda_args,
    cpp_args: gpu_cpp_args,
    link_args: gpu_link_args,
)

# Build ASORA test module
py.extension_module(
    'libasoratest',
    test_sources,
    dependencies: [py_dep, np_dep],
    install: true,
    subdir: 'pyc2ray/lib',
    link_with: asora_core,
    cuda_args: gpu_cuda_args,
    cpp_args: gpu_cpp_args,
    link_args: gpu_link_args,
)
