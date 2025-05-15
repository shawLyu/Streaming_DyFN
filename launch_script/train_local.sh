source /home/luban/miniconda3/etc/profile.d/conda.sh
conda activate video_depth
accelerate launch \
    --num_processes 2 \
    moge/scripts/train.py \
    --config configs/train/v1_test_local.json \
    --workspace workspace/test_debug_local \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 16 \
    --checkpoint latest \
    --enable_gradient_checkpointing False \
    --vis_every 500 \
    --enable_mlflow True \
    --enable_mixed_precision True \
    --save_every 2000
