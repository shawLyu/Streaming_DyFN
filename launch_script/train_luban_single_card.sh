source /home/luban/miniconda3/etc/profile.d/conda.sh
conda activate video_depth

echo "[INFO] Starting accelerate launch..."

CFG=${1:-"configs/train/v1_train_single.json"}

WORKSPACE=${2:-"workspace/train_luban_single"}

accelerate launch \
    --num_processes 8 \
    moge/scripts/train.py \
    --config $CFG \
    --workspace $WORKSPACE \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 16 \
    --checkpoint latest \
    --enable_gradient_checkpointing False \
    --vis_every 500 \
    --enable_mlflow True \
    --enable_mixed_precision True \
    --num_iterations 400000 \
    --save_every 1000