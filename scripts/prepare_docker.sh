docker load -i /home/jeesoo/jira/rocm_pytorch_modified
docker pull rdma/container_tools_installer
docker pull rdma/sriov-plugin

docker run --net=host -v /usr/bin:/tmp rdma/container_tools_installer

docker network create -d sriov --subnet=192.169.16.0/24 -o netdevice=ib0 ibnet
