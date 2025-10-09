#!/bin/bash

# Configuration
CHECKPOINT_DIR="../workspace/v_MoGE_tune/gru_moge_scale_all_every_frame_gram/checkpoint"
VIDEO_DIR="../demos/video_eval"
OUTPUT_BASE_DIR="../exps/video_evaluation/gru_moge_scale_all_every_frame_gram"
EVAL_SCRIPT="../moge/scripts/eval_video_baseline.py"

# Create output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Function to evaluate a single checkpoint
evaluate_checkpoint() {
    local step=$1
    local checkpoint_file="${CHECKPOINT_DIR}/${step}.pt"
    local output_dir="${OUTPUT_BASE_DIR}/step_${step}"
    
    echo "Evaluating checkpoint at step $step..."
    echo "Checkpoint: $checkpoint_file"
    echo "Output directory: $output_dir"
    echo "----------------------------------------"
    
    # Create output directory for this step
    mkdir -p "$output_dir"
    
    # Run evaluation
    python "$EVAL_SCRIPT" \
        --video_dir_path "$VIDEO_DIR" \
        --pretrained "$checkpoint_file" \
        --output_dir "$output_dir"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully evaluated step $step"
    else
        echo "❌ Failed to evaluate step $step"
    fi
    echo "========================================"
}

# Function to evaluate multiple checkpoints
evaluate_multiple_checkpoints() {
    local start_step=$1
    local end_step=$2
    local step_interval=$3
    
    echo "Starting evaluation from step $start_step to $end_step (interval: $step_interval)"
    echo "========================================"
    
    for step in $(seq $start_step $step_interval $end_step); do
        # Format step with leading zeros (8 digits)
        formatted_step=$(printf "%08d" $step)
        checkpoint_file="${CHECKPOINT_DIR}/${formatted_step}.pt"
        
        # Check if checkpoint exists
        if [ -f "$checkpoint_file" ]; then
            evaluate_checkpoint "$formatted_step"
        else
            echo "⚠️  Checkpoint $formatted_step not found, skipping..."
        fi
    done
}

# Function to evaluate specific steps
evaluate_specific_steps() {
    local steps=("$@")
    
    echo "Evaluating specific steps: ${steps[*]}"
    echo "========================================"
    
    for step in "${steps[@]}"; do
        # Format step with leading zeros (8 digits)
        formatted_step=$(printf "%08d" $step)
        checkpoint_file="${CHECKPOINT_DIR}/${formatted_step}.pt"
        
        # Check if checkpoint exists
        if [ -f "$checkpoint_file" ]; then
            evaluate_checkpoint "$formatted_step"
        else
            echo "⚠️  Checkpoint $formatted_step not found, skipping..."
        fi
    done
}

# Main script logic
case "$1" in
    "range")
        # Usage: ./evaluate_checkpoints.sh range start_step end_step interval
        if [ $# -ne 4 ]; then
            echo "Usage: $0 range <start_step> <end_step> <interval>"
            echo "Example: $0 range 5000 20000 2500"
            exit 1
        fi
        evaluate_multiple_checkpoints $2 $3 $4
        ;;
    "specific")
        # Usage: ./evaluate_checkpoints.sh specific step1 step2 step3 ...
        if [ $# -lt 2 ]; then
            echo "Usage: $0 specific <step1> <step2> <step3> ..."
            echo "Example: $0 specific 5000 10000 15000 20000"
            exit 1
        fi
        shift  # Remove 'specific' from arguments
        evaluate_specific_steps "$@"
        ;;
    "single")
        # Usage: ./evaluate_checkpoints.sh single step
        if [ $# -ne 2 ]; then
            echo "Usage: $0 single <step>"
            echo "Example: $0 single 11500"
            exit 1
        fi
        formatted_step=$(printf "%08d" $2)
        evaluate_checkpoint "$formatted_step"
        ;;
    "all")
        # Evaluate all available checkpoints (every 500 steps)
        echo "Evaluating all available checkpoints..."
        evaluate_multiple_checkpoints 0 39000 500
        ;;
    *)
        echo "Usage: $0 {range|specific|single|all} [arguments...]"
        echo ""
        echo "Commands:"
        echo "  range <start> <end> <interval>  - Evaluate checkpoints in a range"
        echo "  specific <step1> <step2> ...    - Evaluate specific steps"
        echo "  single <step>                   - Evaluate a single step"
        echo "  all                             - Evaluate all available checkpoints"
        echo ""
        echo "Examples:"
        echo "  $0 range 5000 20000 2500        # Evaluate steps 5000, 7500, 10000, 12500, 15000, 17500, 20000"
        echo "  $0 specific 5000 10000 15000    # Evaluate specific steps"
        echo "  $0 single 11500                 # Evaluate step 11500"
        echo "  $0 all                          # Evaluate all checkpoints (0, 500, 1000, ..., 39000)"
        exit 1
        ;;
esac

echo "Evaluation completed!"