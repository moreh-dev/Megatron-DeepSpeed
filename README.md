# RDM (ROCm + Megatron-DeepSpeed)
---------------
## Prerequisite
- Docker
  - 본인 계정으로 docker command가 정상 실행 가능한지 확인
## Setup
* N개의 노드를 1개의 master 노드와 (N - 1)개의 worker 노드로 구분합니다.
1. 모든 노드에서 `./scripts/prepare_docker.sh` script를 실행합니다.
    * ssh command를 사용한 스크립트를 작성하는 것을 추천드립니다.
        * ```
          ssh aiot-mi250-001 ./scripts/prepare_docker.sh
          ssh aiot-mi250-002 ./scripts/prepare_docker.sh
          ... (or use for loop)
          ```
    * 해당 스크립트는 다음과 같은 작업을 합니다.
    1. 아래와 같은 docker image들을 설치합니다.
        1. `rocm/pytorch`
            * (aiot cluster `/home/jeesoo/jira/rocm_pytorch` file을 load 합니다. 다른 cluster에서 setup하는 경우 주의)
        2. `rdma/sriov-plugin`
        3. `rdma/container_tools_installer`
    2. `rdma/container_tools_installer`container를 실행해 sriov 사용 가능한 docker command를 설치합니다.
    3. infiniband를 사용하는 docker network를 생성합니다. `192.169.16.0/24` 대역을 사용합니다.
        * host 가 사용하는 ip 대역과 겹치지 않는지 **반드시** 확인합니다.
2. 모든 노드에서 `./scripts/run_plugin.sh` script를 실행합니다.
3. 노드 별로 `MY_IP` 환경변수를 셋팅합니다.
    * 해당 환경 변수는 https://github.com/moreh-dev/Megatron-DeepSpeed/blob/RDM/scripts/run_docker.sh 에서 사용됩니다.
    * `192.169.16.100`이 마스터 노드로 동작하고
        * https://github.com/moreh-dev/Megatron-DeepSpeed/blob/RDM/scripts/run-t5-23b.sh#L115
        * `deepspeed --force_multi --num_nodes=$NODES --hostfile $HF --no_ssh_check --master_addr 192.169.16.100 ... `
    * `192.169.16.101` 부터 차례대로 worker 노드에 할당합니다.
        * https://github.com/moreh-dev/Megatron-DeepSpeed/blob/RDM/hostfile
        * 노드가 156대 이상 있어, 대역이 부족하다면 100이 아닌 1번을 마스터 노드로 할당하고 그보다 더 부족하다면, subnet mask를 16으로 바꾸는 등 설정이 필요합니다.
    * home 어딘가에 `hostname:docker_ip` dictionary를 두고 `bashrc` 에서 해당 파일을 읽어 `MY_IP`를 export 하는 방식을 추천드립니다. 
4. 노드 별로 `./scripts/run_docker.sh` script를 실행합니다.
    * 스크립트의 volume 인자를 본인 local의 해당 repo에 접근 가능하도록 주어야 합니다.
    * 사용이 끝난 뒤엔 노드 별로 `./scripts/clean_docker.sh` script를 실행해 모든 컨테이너를 내려줍니다.

## Prepare dataset
### T5, GPT 공통
* `dataset/download_book.sh` 와 `dataset/download_vocab.sh` 를 실행합니다.


## Run
