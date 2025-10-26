echo $RESOURCE_NUM_GPU
echo $DISTRIBUTED_NODE_COUNT
echo $DISTRIBUTED_NODE_RANK
echo $DISTRIBUTED_MASTER_HOSTS
echo $DISTRIBUTED_PYTORCH_PORT

GLOBAL_NUM_PROCESSES=$(($RESOURCE_NUM_GPU * $DISTRIBUTED_NODE_COUNT))
echo "Launching training with $GLOBAL_NUM_PROCESSES total processes"

NETWORK_INTERFACE="eth0"
if ip addr | grep -q "ens3"; then
    NETWORK_INTERFACE="ens3"
elif ip addr | grep -q "eno1"; then
    NETWORK_INTERFACE="eno1"
fi
echo "[INFO] Using network interface: $NETWORK_INTERFACE"

export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($RESOURCE_NUM_GPU - 1)))

echo "[INFO] Starting accelerate launch..."
CFG=${1:-"configs/train/video_finetune_luban.json"}

WORKSPACE=${2:-"workspace/video_finetune_luban"}

accelerate launch \
    --multi_gpu \
    --num_processes=$GLOBAL_NUM_PROCESSES \
    --num_machines=$DISTRIBUTED_NODE_COUNT \
    --machine_rank=$DISTRIBUTED_NODE_RANK \
    --main_process_ip=$DISTRIBUTED_MASTER_HOSTS \
    --main_process_port=$LUBAN_AVAILBLE_PORT_0 \
    moge/scripts/train.py \
        --config $CFG \
        --workspace $WORKSPACE \
        --gradient_accumulation_steps 1 \
        --batch_size_forward 2 \
        --checkpoint pretrained_moge/pretrained_moge.pt \
        --enable_gradient_checkpointing False \
        --vis_every 500 \
        --enable_mlflow True \
        --enable_mixed_precision False \
        --num_iterations 400000 \
        --save_every 1000 \
        --log_every 500