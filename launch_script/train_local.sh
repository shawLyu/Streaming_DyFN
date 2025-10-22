# source /home/luban/miniconda3/etc/profile.d/conda.sh
# conda activate video_depth

CFG=${1:-"configs/train/video_finetune_local_tartanair.json"}

WORKSPACE=${2:-"workspace/video_finetune_local_bug_fixed"}

accelerate launch \
    --num_processes 2 \
    moge/scripts/train.py \
    --config $CFG \
    --workspace $WORKSPACE \
    --checkpoint workspace/v_MoGE_tune/conv_gru_temporal_loss_residual/checkpoint/00010000.pt \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 2 \
    --enable_gradient_checkpointing False \
    --vis_every 500 \
    --enable_mlflow True \
    --enable_mixed_precision False \
    --save_every 500 \
    --log_every 100
