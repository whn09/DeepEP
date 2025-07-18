echo -e "Host * \n    ForwardAgent yes \nHost * \n    StrictHostKeyChecking no" >> ~/.ssh/config
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
echo "XXXX" >> ~/.ssh/authorized_keys
ssh member_node_private_ip

cd /usr/local/cuda-12.8/efa/test-cuda-12.8

./all_reduce_perf -b 8 -e 128M -f 2 -g 8

# Maybe bug, just input your private IPs to my-hosts
# TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` \
# && curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/local-ipv4 >> my-hosts

/opt/amazon/openmpi/bin/mpirun \
-x FI_EFA_USE_DEVICE_RDMA=1 \
-x LD_LIBRARY_PATH=/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH \
-x NCCL_DEBUG=INFO \
--hostfile my-hosts -n 16 -N 8 \
--mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
/usr/local/cuda-12.8/efa/test-cuda-12.8/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100
