pip install torch numpy

# Build and make symbolic links for SO files
NVSHMEM_DIR=/home/ubuntu/nvshmem python3 setup.py build
# You may modify the specific SO names according to your own platform
ln -s build/lib.linux-x86_64-cpython-310/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so

# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils.py`
# according to your own cluster settings, and launch into multiple nodes 
python3 tests/test_intranode.py
python3 tests/test_low_latency.py

export MASTER_ADDR=172.31.55.118  # 主节点IP
export MASTER_PORT=29500       # 主节点端口
export WORLD_SIZE=2            # 节点总数
export RANK=0  # 0,1
python3 tests/test_internode.py
