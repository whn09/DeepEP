# cd /usr/local/cuda-12.8/efa/test-cuda-12.8

# ./all_reduce_perf -b 8 -e 128M -f 2 -g 8

# mpirun -np 16 -H 172.31.61.101:8,172.31.53.208:8 -N 8 ./all_reduce_perf -b 8 -e 8G -f 2 -g 1
# #  -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH -x NCCL_DEBUG=INFO -mca pml ob1 -mca btl ^openib

/opt/amazon/openmpi/bin/mpirun \
-x FI_EFA_USE_DEVICE_RDMA=1 \
-x LD_LIBRARY_PATH=/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH \
-x NCCL_DEBUG=INFO \
--hostfile my-hosts -n 16 -N 8 \
--mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
/usr/local/cuda-12.8/efa/test-cuda-12.8/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100
