# echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee -a /etc/modprobe.d/nvidia.conf
# sudo update-initramfs -u
# sudo reboot

sudo apt update
sudo apt install -y ninja-build
sudo apt install -y python3.10-venv

sudo apt-get install -y build-essential devscripts debhelper fakeroot pkg-config dkms
wget -O gdrcopy-v2.4.4.tar.gz https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
tar xf gdrcopy-v2.4.4.tar.gz
cd gdrcopy-2.4.4/
sudo make prefix=/opt/gdrcopy -j$(nproc) install

cd packages/
CUDA=/usr/local/cuda ./build-deb-packages.sh
sudo dpkg -i gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb \
             gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.8.deb \
             gdrcopy_2.4.4_amd64.Ubuntu22_04.deb \
             libgdrapi_2.4.4_amd64.Ubuntu22_04.deb

/opt/gdrcopy/bin/gdrcopy_copybw

export CUDA_HOME=/usr/local/cuda
# disable all features except IBGDA
export NVSHMEM_IBGDA_SUPPORT=1

export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_LIBFABRIC_SUPPORT=1
export GDRCOPY_HOME=/opt/gdrcopy
export LIBFABRIC_HOME=/opt/amazon/efa

cmake -G Ninja -S . -B build -DCMAKE_INSTALL_PREFIX=/home/ubuntu/nvshmem
cmake --build build/ --target install

export NVSHMEM_DIR=/home/ubuntu/nvshmem  # Use for DeepEP installation
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa

nvshmem-info -a
