import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


if __name__ == '__main__':
    # Check for EFA-DP-direct mode (GPU-direct EFA, no NVSHMEM)
    use_efa_dp_direct = int(os.getenv('USE_EFA_DP_DIRECT', 0))

    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'

    if use_efa_dp_direct:
        # EFA-DP-direct mode: bypass NVSHMEM entirely
        disable_nvshmem = True
        print('EFA-DP-direct mode enabled: bypassing NVSHMEM, using efa-dp-direct for GPU-direct RDMA\n')
    elif nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
            )
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem and not use_efa_dp_direct:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    if use_efa_dp_direct:
        # EFA-DP-direct: link efa-dp-direct library + libibverbs + libefa
        efa_dp_dir = os.getenv('EFA_DP_DIRECT_DIR', os.path.join(os.path.dirname(__file__), 'third-party', 'efa-dp-direct'))
        efa_home = os.getenv('EFA_HOME', '/opt/amazon/efa')

        cxx_flags.append('-DUSE_EFA_DP_DIRECT')
        nvcc_flags.append('-DUSE_EFA_DP_DIRECT')

        # Include efa-dp-direct headers
        include_dirs.append(os.path.join(efa_dp_dir, 'CUDA', 'src'))

        # Link efa-dp-direct library + EFA verbs
        efa_dp_lib_dir = os.path.join(efa_dp_dir, 'CUDA', 'build')
        library_dirs.append(efa_dp_lib_dir)
        library_dirs.append('/usr/lib/x86_64-linux-gnu')
        extra_link_args.extend([
            '-lefacudadp',
            '-libverbs', '-lefa',
            f'-Wl,-rpath,{efa_dp_lib_dir}',
        ])

        # Add internode sources + EFA runtime
        sources.extend([
            'csrc/kernels/internode.cu',
            'csrc/kernels/internode_ll.cu',
            'csrc/kernels/efa_dp_direct_runtime.cu',
        ])

        # Device linking: resolve cross-TU __device__ symbols (required with -rdc=true)
        nvcc_dlink.extend(['-dlink', f'-L{efa_dp_lib_dir}', '-lefacudadp'])

        # Include EFA verbs headers
        if os.path.isdir(os.path.join(efa_home, 'include')):
            include_dirs.append(os.path.join(efa_home, 'include'))

    elif disable_nvshmem:
        # No NVSHMEM and no EFA-DP-direct: disable internode
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        # NVSHMEM mode (original)
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    if use_efa_dp_direct:
        print(f' > Backend: EFA-DP-direct (GPU-direct EFA RDMA)')
        print(f' > EFA-DP dir: {efa_dp_dir}')
    else:
        print(f' > Backend: NVSHMEM')
        print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='deep_ep',
                     version='1.2.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=[
                         CUDAExtension(name='deep_ep_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
