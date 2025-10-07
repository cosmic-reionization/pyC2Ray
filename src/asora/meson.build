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
        'nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader', check: false)
  if r_compute_cap.returncode() != 0
    warning('nvidia-smi not detected; defaulting to compute capability 6.0')
    compute_cap = '60'
  else
    cc = r_compute_cap.stdout().strip()
    message(f'Local compute capability is determined to be @cc@')
    compute_cap = cc[0] + '0'
  endif
endif

gpu_arch_flag = '--gpu-architecture=sm_' + compute_cap

# Build ASORA module with nvcc
py.extension_module(
    'libasora',
    asora_sources,
    include_directories: inc_np,
    dependencies: py_dep,
    install: true,
    subdir: 'pyc2ray/lib',
    cuda_args: gpu_arch_flag,
)
