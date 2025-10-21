source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_depth

echo "[INFO] Starting accelerate launch..."

CFG=${1:-"configs/train/video_finetune_local_tartanair.json"}

WORKSPACE=${2:-"workspace/video_finetune_luban"}

accelerate launch \
    --num_processes 8 \
    moge/scripts/train.py \
    --config $CFG \
    --workspace $WORKSPACE \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 4 \
    --checkpoint workspace/v_MoGE_tune/gru_moge_scale_all_every_frame_gram_scale_first_frame/checkpoint/00039500.pt \
    --enable_gradient_checkpointing False \
    --vis_every 500 \
    --enable_mlflow True \
    --enable_mixed_precision False \
    --num_iterations 40000 \
    --save_every 500