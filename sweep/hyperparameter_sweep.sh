#!/bin/bash

################################################################################
# Hyperparameter Sweep Manager for bhumi_exp_2
#
# This script orchestrates W&B hyperparameter sweeps for training with planning.
# It handles initialization, execution, monitoring, and analysis of sweeps.
#
# Usage:
#   ./hyperparameter_sweep.sh init              # Initialize a new sweep
#   ./hyperparameter_sweep.sh run <SWEEP_ID>    # Run a single sweep agent
#   ./hyperparameter_sweep.sh run <SWEEP_ID> 4  # Run 4 agents in parallel
#   ./hyperparameter_sweep.sh monitor           # Monitor active sweeps
#   ./hyperparameter_sweep.sh status            # Show sweep status
#   ./hyperparameter_sweep.sh help              # Show this help message
#
################################################################################

set -e

# Configuration
WORK_DIR="/scratch/yp2693/world_models/bhumi_exp_2"
SWEEP_YAML="$WORK_DIR/sweep.yaml"
RUN_SWEEP_SCRIPT="$WORK_DIR/run_sweep.sh"
PYTHON_ENV="/scratch/yp2693/world_models/penv"
LOG_DIR="$WORK_DIR/sweep_logs"
SWEEP_ID_FILE="$WORK_DIR/.sweep_id"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         HYPERPARAMETER SWEEP (bhumi_exp_2)                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

setup_environment() {
    print_info "Setting up environment..."

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate "$PYTHON_ENV"

    print_success "Environment ready"
}

################################################################################
# Initialize Sweep
################################################################################

init_sweep() {
    print_header
    print_info "Initializing new hyperparameter sweep..."
    echo ""

    setup_environment

    # Check if sweep.yaml exists
    if [ ! -f "$SWEEP_YAML" ]; then
        print_error "sweep.yaml not found at $SWEEP_YAML"
        echo "Please ensure sweep.yaml exists in the experiment folder."
        exit 1
    fi

    print_info "Sweep Configuration:"
    echo "  📊 File: $(basename $SWEEP_YAML)"
    echo "  📂 Location: $WORK_DIR"
    echo ""

    # Display sweep parameters
    echo "  Swept Parameters:"
    echo "    • Quantization Loss Weight: 0.5, 1.0, 2.0"
    echo "    • VCReg Loss Weight: 1, 5, 10"
    echo "    • State Quantizer LR: 5e-6 to 5e-3 (log-uniform)"
    echo "    • Predictor LR: 5e-6 to 5e-3 (log-uniform)"
    echo "    • Encoder LR: 5e-6 to 5e-3 (log-uniform)"
    echo ""

    # Create sweep
    cd "$WORK_DIR"
    print_info "Creating sweep on W&B..."

    SWEEP_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1)
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP '(?<=/)[\w\-]+$' | tail -1)

    if [ -z "$SWEEP_ID" ]; then
        print_error "Failed to create sweep. Output:"
        echo "$SWEEP_OUTPUT"
        exit 1
    fi

    # Save sweep ID
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"

    echo ""
    echo "═════════════════════════════════════════════════════════════"
    print_success "Sweep created successfully!"
    echo "═════════════════════════════════════════════════════════════"
    echo ""
    echo "Sweep ID: ${BLUE}$SWEEP_ID${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1️⃣  Run a single agent (sequential):"
    echo "     ${BLUE}./hyperparameter_sweep.sh run $SWEEP_ID${NC}"
    echo ""
    echo "  2️⃣  Or run multiple agents in parallel (faster):"
    echo "     ${BLUE}./hyperparameter_sweep.sh run $SWEEP_ID 4${NC}"
    echo ""
    echo "  3️⃣  Monitor progress:"
    echo "     ${BLUE}./hyperparameter_sweep.sh monitor${NC}"
    echo ""
    echo "  4️⃣  View on W&B Dashboard:"
    echo "     https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps"
    echo ""
}

################################################################################
# Run Sweep Agent(s)
################################################################################

run_sweep() {
    local SWEEP_ID="$1"
    local NUM_AGENTS="${2:-1}"

    if [ -z "$SWEEP_ID" ]; then
        print_error "Sweep ID not provided"
        echo "Usage: ./hyperparameter_sweep.sh run <SWEEP_ID> [NUM_AGENTS]"
        exit 1
    fi

    print_header
    print_info "Running sweep agents..."
    echo ""

    setup_environment

    print_info "Sweep Configuration:"
    echo "  🎯 Sweep ID: $SWEEP_ID"
    echo "  🔄 Number of agents: $NUM_AGENTS"
    echo "  📂 Working directory: $WORK_DIR"
    echo ""

    cd "$WORK_DIR"

    # Save sweep ID
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"

    if [ "$NUM_AGENTS" -eq 1 ]; then
        print_info "Launching single agent..."
        echo ""
        bash "$RUN_SWEEP_SCRIPT" "$SWEEP_ID" 2>&1 | tee "$LOG_DIR/sweep_agent.log"
    else
        print_info "Launching $NUM_AGENTS agents in parallel..."
        echo ""
        bash "$RUN_SWEEP_SCRIPT" "$SWEEP_ID" "$NUM_AGENTS" 2>&1 | tee "$LOG_DIR/sweep_agents_parallel.log"
    fi

    echo ""
    print_success "Sweep agents started"
    print_info "Monitor progress with: ./hyperparameter_sweep.sh monitor"
}

################################################################################
# Monitor Sweeps
################################################################################

monitor_sweeps() {
    print_header
    print_info "Monitoring active sweeps..."
    echo ""

    setup_environment

    # Get sweep ID if available
    local SWEEP_ID=""
    if [ -f "$SWEEP_ID_FILE" ]; then
        SWEEP_ID=$(cat "$SWEEP_ID_FILE")
        print_info "Current Sweep ID: $SWEEP_ID"
        echo ""
    fi

    # Display monitoring options
    echo "📊 Monitoring Options:"
    echo ""
    echo "  1. W&B Dashboard:"
    echo "     https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps"
    echo ""
    echo "  2. Check agent logs:"
    echo "     tail -f $LOG_DIR/sweep_agent*.log"
    echo ""
    echo "  3. Check training logs:"
    echo "     tail -f $WORK_DIR/outputs/*/training.log"
    echo ""
    echo "  4. Check planning output:"
    echo "     ls -la $WORK_DIR/plan_outputs/"
    echo ""

    # Show current runs
    print_info "Active training runs:"
    if [ -d "$WORK_DIR/outputs" ]; then
        find "$WORK_DIR/outputs" -maxdepth 2 -type d -name "*-*" | sort -r | head -5 | while read dir; do
            if [ -f "$dir/train.log" ]; then
                echo "  📁 $(basename $(dirname $dir))/$(basename $dir)"
            fi
        done
    else
        echo "  No runs yet"
    fi
    echo ""
}

################################################################################
# Sweep Status
################################################################################

sweep_status() {
    print_header
    print_info "Hyperparameter Sweep Status"
    echo ""

    setup_environment

    # Check if sweep ID is available
    if [ ! -f "$SWEEP_ID_FILE" ]; then
        print_warning "No sweep has been initialized yet"
        echo "Initialize a sweep with: ./hyperparameter_sweep.sh init"
        exit 0
    fi

    local SWEEP_ID=$(cat "$SWEEP_ID_FILE")
    print_info "Current Sweep: $SWEEP_ID"
    echo ""

    # Count completed runs
    local COMPLETED=0
    if [ -d "$WORK_DIR/outputs" ]; then
        COMPLETED=$(find "$WORK_DIR/outputs" -maxdepth 2 -type d -name "*-*" | wc -l)
    fi

    print_info "Summary:"
    echo "  📊 Total runs completed: $COMPLETED"
    echo "  📂 Output directory: $WORK_DIR/outputs"
    echo "  📋 Log directory: $LOG_DIR"
    echo ""

    # Show recent runs
    if [ "$COMPLETED" -gt 0 ]; then
        print_info "Recent runs:"
        find "$WORK_DIR/outputs" -maxdepth 2 -type d -name "*-*" | sort -r | head -3 | while read dir; do
            if [ -f "$dir/train.log" ]; then
                local name=$(basename $(dirname $dir))/$(basename $dir)
                echo "  ✓ $name"
            fi
        done
        echo ""
    fi

    echo "Next steps:"
    echo "  1. Monitor on W&B: https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps"
    echo "  2. View logs: tail -f $LOG_DIR/sweep_agent*.log"
    echo "  3. Check training: tail -f $WORK_DIR/outputs/*/training.log"
    echo ""
}

################################################################################
# Help
################################################################################

show_help() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║    Hyperparameter Sweep Manager (bhumi_exp_2) - Help      ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: ./hyperparameter_sweep.sh <COMMAND> [OPTIONS]"
    echo ""
    echo "Commands:"
    echo ""
    echo "  init                    Initialize a new sweep"
    echo "                          Creates sweep on W&B and returns sweep ID"
    echo ""
    echo "  run <SWEEP_ID> [N]      Run sweep agent(s)"
    echo "                          N = number of parallel agents (default: 1)"
    echo "                          Example: run <SWEEP_ID> 4"
    echo ""
    echo "  monitor                 Monitor active sweeps"
    echo "                          Shows logs and W&B dashboard links"
    echo ""
    echo "  status                  Show sweep status"
    echo "                          Lists completed runs and summary"
    echo ""
    echo "  help                    Show this help message"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Quick Start:"
    echo ""
    echo "  1. Initialize sweep:"
    echo "     ./hyperparameter_sweep.sh init"
    echo ""
    echo "  2. Run with 4 parallel agents:"
    echo "     ./hyperparameter_sweep.sh run <SWEEP_ID> 4"
    echo ""
    echo "  3. Monitor progress:"
    echo "     ./hyperparameter_sweep.sh monitor"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Sweep Parameters:"
    echo ""
    echo "  Loss Weights (Grid):"
    echo "    • Quantization Loss: 0.5, 1.0, 2.0"
    echo "    • VCReg Loss: 1, 5, 10"
    echo ""
    echo "  Learning Rates (Log-Uniform):"
    echo "    • State Quantizer LR: 5e-6 to 5e-3"
    echo "    • Predictor LR: 5e-6 to 5e-3"
    echo "    • Encoder LR: 5e-6 to 5e-3"
    echo ""
    echo "  Total combinations: ~9 (grid) × variations (random LRs)"
    echo "  Estimated time: 9-12 hours"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Configuration Files:"
    echo "  • $SWEEP_YAML"
    echo "  • $RUN_SWEEP_SCRIPT"
    echo ""
    echo "For detailed information, see: HYPERPARAMETER_SWEEP_GUIDE.md"
    echo ""
}

################################################################################
# Main
################################################################################

main() {
    local COMMAND="${1:-help}"

    case "$COMMAND" in
        init)
            init_sweep
            ;;
        run)
            run_sweep "$2" "$3"
            ;;
        monitor)
            monitor_sweeps
            ;;
        status)
            sweep_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
