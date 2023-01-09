docker load -i /home/jeesoo/jira/rocm_pytorch
docker pull rdma/container_tools_installer
docker pull rdma/sriov-plugin

docker run --net=host -v /usr/bin:/tmp rdma/container_tools_installer
docker run -d -v /run/docker/plugins:/run/docker/plugins -v /etc/docker:/etc/docker -v /var/run:/var/run --net=host --privileged rdma/sriov-plugin

docker network create -d sriov --subnet=192.169.16.0/24 -o netdevice=ib0 ibnet
