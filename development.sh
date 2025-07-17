# Build and make symbolic links for SO files
NVSHMEM_DIR=/home/ubuntu/nvshmem python3 setup.py build
# You may modify the specific SO names according to your own platform
ln -s build/lib.linux-x86_64-cpython-310/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so

# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils.py`
# according to your own cluster settings, and launch into multiple nodes 
python3 tests/test_intranode.py
python3 tests/test_low_latency.py

NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa MASTER_ADDR=172.31.61.101 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python3 tests/test_internode.py
NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa MASTER_ADDR=172.31.61.101 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python3 tests/test_internode.py
