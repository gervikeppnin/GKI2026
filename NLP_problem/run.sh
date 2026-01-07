#!/bin/bash
# =============================================================================
# Wav2Vec2 CTC Icelandic ASR - Training Entry Point
# =============================================================================
# HARDWARE: RTX 4060 8GB VRAM
# CRITICAL: DO NOT modify fp16, batch_size, or max_audio_duration
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Wav2Vec2 CTC Icelandic ASR Training${NC}"
echo -e "${CYAN}============================================${NC}"

# =============================================================================
# Hardware-safe defaults (DO NOT CHANGE for 8GB VRAM)
# =============================================================================
FP16="--fp16"
BATCH_SIZE=2
GRADIENT_ACCUMULATION=8
LEARNING_RATE=3e-4
MAX_STEPS=5000
MAX_AUDIO_DURATION=10.0

OUTPUT_DIR="./output"

# =============================================================================
# Check dependencies
# =============================================================================
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python3 is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}Python:${NC} $(python3 --version)"
}

check_cuda() {
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}')")
        echo -e "${GREEN}GPU:${NC} $GPU_NAME ($GPU_MEM GB)"
        return 0
    else
        echo -e "${YELLOW}WARNING: CUDA not available. Training on CPU (very slow).${NC}"
        FP16=""  # Disable FP16 on CPU
        return 1
    fi
}

install_requirements() {
    echo -e "\n${YELLOW}Installing requirements...${NC}"
    pip install -q -r requirements.txt
}

# =============================================================================
# Main entry point
# =============================================================================
main() {
    check_python
    check_cuda

    # Parse arguments
    if [ "$1" == "--benchmark" ]; then
        # =================================================================
        # BENCHMARK MODE: 100-step smoke test
        # =================================================================
        echo -e "\n${CYAN}============================================${NC}"
        echo -e "${CYAN}  BENCHMARK MODE (100 steps)${NC}"
        echo -e "${CYAN}============================================${NC}"
        echo -e "Running smoke test to validate:"
        echo -e "  - VRAM fits within 8GB"
        echo -e "  - Measure throughput (samples/sec)"
        echo -e "  - Estimate full training time"
        echo -e "${CYAN}============================================${NC}\n"

        python3 src/benchmark.py \
            $FP16 \
            --batch_size $BATCH_SIZE \
            --steps 100 \
            --target_steps $MAX_STEPS

    elif [ "$1" == "--eval" ]; then
        # =================================================================
        # EVALUATION MODE
        # =================================================================
        MODEL_PATH="${2:-$OUTPUT_DIR/best_model}"

        echo -e "\n${CYAN}============================================${NC}"
        echo -e "${CYAN}  EVALUATION MODE${NC}"
        echo -e "${CYAN}============================================${NC}"
        echo -e "Model: $MODEL_PATH"
        echo -e "${CYAN}============================================${NC}\n"

        python3 src/evaluate.py \
            --model_path "$MODEL_PATH" \
            --processor_path "$OUTPUT_DIR/processor" \
            --split test \
            --num_samples 500 \
            --output_file "$OUTPUT_DIR/evaluation_results.txt"

    else
        # =================================================================
        # FULL TRAINING MODE
        # =================================================================
        echo -e "\n${CYAN}============================================${NC}"
        echo -e "${CYAN}  FULL TRAINING MODE${NC}"
        echo -e "${CYAN}============================================${NC}"
        echo -e "Configuration (8GB VRAM safe):"
        echo -e "  FP16: ${GREEN}enabled${NC} (mandatory)"
        echo -e "  Batch size: ${GREEN}$BATCH_SIZE${NC}"
        echo -e "  Gradient accumulation: ${GREEN}$GRADIENT_ACCUMULATION${NC}"
        echo -e "  Effective batch: ${GREEN}$((BATCH_SIZE * GRADIENT_ACCUMULATION))${NC}"
        echo -e "  Learning rate: ${GREEN}$LEARNING_RATE${NC}"
        echo -e "  Max steps: ${GREEN}$MAX_STEPS${NC}"
        echo -e "  Max audio duration: ${GREEN}${MAX_AUDIO_DURATION}s${NC}"
        echo -e "  Output: ${GREEN}$OUTPUT_DIR${NC}"
        echo -e "${CYAN}============================================${NC}\n"

        # Create output directory
        mkdir -p "$OUTPUT_DIR"

        # Run training
        python3 src/train.py \
            $FP16 \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
            --learning_rate $LEARNING_RATE \
            --max_steps $MAX_STEPS \
            --max_audio_duration $MAX_AUDIO_DURATION \
            --output_dir "$OUTPUT_DIR" \
            --save_steps 1000 \
            --logging_steps 50 \
            --eval_steps 500 \
            --eval_samples 200

        # Check if training succeeded
        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}============================================${NC}"
            echo -e "${GREEN}  TRAINING COMPLETE${NC}"
            echo -e "${GREEN}============================================${NC}"
            echo -e "Models saved to: $OUTPUT_DIR"
            echo -e "  - best_model/"
            echo -e "  - final_model/"
            echo -e "  - checkpoint-*/"
            echo -e "\nTo evaluate, run:"
            echo -e "  ${CYAN}./run.sh --eval${NC}"
            echo -e "\nTo view TensorBoard logs:"
            echo -e "  ${CYAN}tensorboard --logdir $OUTPUT_DIR/tensorboard${NC}"
            echo -e "${GREEN}============================================${NC}"
        else
            echo -e "\n${RED}Training failed!${NC}"
            exit 1
        fi
    fi
}

# =============================================================================
# Help
# =============================================================================
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "${CYAN}Wav2Vec2 CTC Icelandic ASR Training${NC}"
    echo ""
    echo "Usage:"
    echo "  ./run.sh              # Full training (5000 steps)"
    echo "  ./run.sh --benchmark  # 100-step smoke test"
    echo "  ./run.sh --eval       # Evaluate best model"
    echo "  ./run.sh --eval PATH  # Evaluate specific model"
    echo ""
    echo "Hardware constraints (RTX 4060 8GB):"
    echo "  - FP16: enabled (mandatory)"
    echo "  - Batch size: 2 (max safe)"
    echo "  - Gradient accumulation: 8"
    echo "  - Max audio duration: 10s"
    echo ""
    echo "Time estimates:"
    echo "  - Benchmark (100 steps): ~5-10 min"
    echo "  - Full training (5000 steps): ~3-5 hours"
    exit 0
fi

# Run main
main "$@"
