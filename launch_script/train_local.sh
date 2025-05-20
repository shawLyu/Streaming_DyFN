source /home/luban/miniconda3/etc/profile.d/conda.sh
conda activate video_depth

CFG=${1:-"configs/train/video_finetune_local.json"}

WORKSPACE=${2:-"workspace/test_debug_local_test_nan"}

accelerate launch \
    --num_processes 2 \
    moge/scripts/train.py \
    --config $CFG \
    --workspace $WORKSPACE \
    --checkpoint pretrained_moge/image_pretrained_moge.pt \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 1 \
    --enable_gradient_checkpointing False \
    --vis_every 500 \
    --enable_mlflow True \
    --enable_mixed_precision True \
    --save_every 500 \
    --log_every 100
