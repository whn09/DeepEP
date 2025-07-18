echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee -a /etc/modprobe.d/nvidia.conf
sudo update-initramfs -u
sudo reboot

sudo apt update
sudo apt install -y ninja-build
sudo apt install -y python3.10-venv

export CUDA_HOME=/usr/local/cuda
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=1
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_IBRC_SUPPORT=1
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_LIBFABRIC_SUPPORT=1
export MPI_HOME=/opt/amazon/openmpi
export PMIX_HOME=/opt/amazon/pmix
export GDRCOPY_HOME=/usr/bin
export LIBFABRIC_HOME=/opt/amazon/efa

cmake -G Ninja -S . -B build -DCMAKE_INSTALL_PREFIX=/home/ubuntu/nvshmem
cmake --build build/ --target install

# Use for DeepEP installation
export NVSHMEM_DIR=/home/ubuntu/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"

nvshmem-info -a
