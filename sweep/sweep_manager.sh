#!/bin/bash

################################################################################
# Complete Hyperparameter Sweep Manager
#
# This script handles the complete sweep workflow:
# 1. Allows updating hyperparameters
# 2. Creates a new sweep
# 3. Prompts for single vs parallel execution
# 4. Runs the sweep agents automatically
#
# Usage:
#   ./sweep_manager.sh              # Interactive mode
#   ./sweep_manager.sh --help       # Show help
#
################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # sweep/ directory
WORK_DIR="$(dirname "$SCRIPT_DIR")"                          # bhumi_exp_2/ directory
SWEEP_YAML="$SCRIPT_DIR/sweep.yaml"                          # In sweep/ dir
PYTHON_BIN="/scratch/yp2693/world_models/penv/bin/python"
SWEEP_OUTPUTS_DIR="$WORK_DIR/sweep_outputs"                 # In bhumi_exp_2/
LOG_DIR="$SWEEP_OUTPUTS_DIR/logs"
SWEEP_ID_FILE="$WORK_DIR/.sweep_id"
SWEEP_CONFIG_FILE="$WORK_DIR/.sweep_config"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        HYPERPARAMETER SWEEP MANAGER (bhumi_exp_2)          ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
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

prompt_yes_no() {
    local question="$1"
    local answer

    while true; do
        read -p "$(echo -e ${YELLOW}$question [y/n]: ${NC})" answer
        case $answer in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer y or n.";;
        esac
    done
}

################################################################################
# Review/Edit Hyperparameters
################################################################################

review_hyperparameters() {
    print_section "Current Hyperparameter Configuration"

    echo "Loss Weights:"
    echo "  • Quantization Loss: 0.5, 1.0, 2.0"
    echo "  • VCReg Loss: 1, 5, 10"
    echo ""
    echo "Learning Rates (log-uniform):"
    echo "  • State Quantizer LR: 5e-6 to 5e-3"
    echo "  • Predictor LR: 5e-6 to 5e-3"
    echo "  • Encoder LR: 5e-6 to 5e-3"
    echo ""
    echo "Training Configuration:"
    echo "  • Epochs: 30"
    echo "  • Batch Size: 32"
    echo "  • Dataset Size: 400 rollouts"
    echo ""

    if prompt_yes_no "Do you want to edit the sweep.yaml?"; then
        print_info "Opening sweep.yaml in editor..."
        ${EDITOR:-nano} "$SWEEP_YAML"
        print_success "Sweep configuration updated"
    else
        print_info "Using current configuration"
    fi
}

################################################################################
# Create Sweep
################################################################################

create_sweep() {
    print_section "Creating Hyperparameter Sweep"

    print_info "Initializing new sweep on W&B..."

    SWEEP_OUTPUT=$("$PYTHON_BIN" -m wandb sweep \
        --entity yashdeep18121-new-york-university \
        --project dino_wm_sweeps \
        "$SWEEP_YAML" 2>&1)

    # Extract sweep ID - get the full qualified path from the last line
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K.+' | tail -1)

    if [ -z "$SWEEP_ID" ]; then
        print_error "Failed to create sweep"
        echo "W&B Output:"
        echo "$SWEEP_OUTPUT"
        exit 1
    fi

    # Save sweep ID
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"

    # Extract short ID for dashboard URL (last component of the full path)
    SHORT_ID=$(echo "$SWEEP_ID" | awk -F'/' '{print $NF}')

    # Save sweep configuration to configs directory for reference
    mkdir -p "$SWEEP_OUTPUTS_DIR/configs"
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    cp "$SWEEP_YAML" "$SWEEP_OUTPUTS_DIR/configs/sweep_${SHORT_ID}_${TIMESTAMP}.yaml"

    print_success "Sweep created successfully!"
    echo ""
    echo -e "${GREEN}Sweep ID: ${CYAN}$SWEEP_ID${NC}"
    echo ""
    echo "W&B Dashboard:"
    echo -e "  ${CYAN}https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps/sweeps/$SHORT_ID${NC}"
    echo ""
}

################################################################################
# Choose Execution Mode
################################################################################

choose_execution_mode() {
    print_section "Choose Execution Mode"

    echo "How do you want to run the sweep agents?"
    echo ""
    echo "  1) Single Agent (Sequential)"
    echo "     - Runs one hyperparameter combination at a time"
    echo "     - Takes ~9-12 hours for all combinations"
    echo "     - Simpler, less resource intensive"
    echo ""
    echo "  2) Multiple Agents in Parallel"
    echo "     - Runs multiple combinations simultaneously"
    echo "     - Faster completion (~2-3 hours with 4 agents)"
    echo "     - Higher resource usage"
    echo ""

    while true; do
        read -p "Enter choice (1 or 2): " choice
        case $choice in
            1)
                EXECUTION_MODE="single"
                NUM_AGENTS=1
                break
                ;;
            2)
                read -p "How many parallel agents? (recommended: 2-4): " num_agents_input
                if [[ "$num_agents_input" =~ ^[0-9]+$ ]] && [ "$num_agents_input" -ge 1 ]; then
                    NUM_AGENTS="$num_agents_input"
                    EXECUTION_MODE="parallel"
                    break
                else
                    echo "Please enter a valid number"
                fi
                ;;
            *)
                echo "Invalid choice. Please enter 1 or 2."
                ;;
        esac
    done

    echo ""
    if [ "$EXECUTION_MODE" = "single" ]; then
        print_info "Single agent mode selected"
        echo "  One agent will run sequentially"
    else
        print_info "Parallel mode selected"
        echo "  $NUM_AGENTS agents will run in parallel"
    fi
}

################################################################################
# Run Sweep Agents
################################################################################

run_sweep_agents() {
    print_section "Starting Sweep Agents"

    SWEEP_ID=$(cat "$SWEEP_ID_FILE")
    AGENT_COMMAND="$PYTHON_BIN -m wandb agent $SWEEP_ID"

    echo "Command:"
    echo -e "  ${CYAN}$AGENT_COMMAND${NC}"
    echo ""

    mkdir -p "$LOG_DIR"

    if [ "$EXECUTION_MODE" = "single" ]; then
        print_info "Starting single agent..."
        echo ""
        echo "To stop the agent: Press Ctrl+C"
        echo ""

        bash -c "$AGENT_COMMAND" 2>&1 | tee "$LOG_DIR/sweep_agent_single.log"

    else
        print_info "Starting $NUM_AGENTS agents in parallel..."
        echo ""
        echo "To stop all agents: Press Ctrl+C"
        echo ""

        # Launch multiple agents in background
        for i in $(seq 1 "$NUM_AGENTS"); do
            print_info "Launching agent $i..."
            bash -c "$AGENT_COMMAND" 2>&1 | tee "$LOG_DIR/sweep_agent_$i.log" &
            AGENT_PIDS[$i]=$!
            sleep 2  # Stagger agent starts
        done

        echo ""
        print_success "All $NUM_AGENTS agents started"
        echo ""
        echo "Agent process IDs: ${AGENT_PIDS[@]}"
        echo ""
        echo "To view logs:"
        for i in $(seq 1 "$NUM_AGENTS"); do
            echo "  Agent $i: tail -f $LOG_DIR/sweep_agent_$i.log"
        done
        echo ""

        # Wait for all agents
        wait
    fi
}

################################################################################
# Summary
################################################################################

show_summary() {
    print_section "Sweep Execution Summary"

    SWEEP_ID=$(cat "$SWEEP_ID_FILE" 2>/dev/null || echo "N/A")
    SHORT_ID=$(echo "$SWEEP_ID" | awk -F'/' '{print $NF}')

    echo "Configuration:"
    echo "  📊 Sweep ID: $SWEEP_ID"
    echo "  🔄 Execution Mode: $EXECUTION_MODE"
    if [ "$EXECUTION_MODE" = "parallel" ]; then
        echo "  👥 Number of Agents: $NUM_AGENTS"
    fi
    echo ""

    echo "Outputs & Monitoring:"
    echo "  🗂️  Training Outputs: sweep_outputs/runs/"
    echo "  📋 Agent Logs: sweep_outputs/logs/"
    echo "  ⚙️  Sweep Configs: sweep_outputs/configs/"
    echo ""
    echo "  🌐 W&B Dashboard: https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps/sweeps/$SHORT_ID"
    echo "  📖 Full Guide: sweep_outputs/README.md"
    echo ""

    echo "Useful Commands:"
    echo "  • View agent logs (real-time):   tail -f sweep_outputs/logs/sweep_agent_1.log"
    echo "  • View all training runs:        ls -ltra sweep_outputs/runs/*/*/"
    echo "  • View training outputs summary: find sweep_outputs/runs -name '.hydra' -type d"
    echo ""

    echo "Next time, use:"
    echo -e "  ${CYAN}./sweep_manager.sh${NC}"
    echo ""
}

################################################################################
# Help
################################################################################

show_help() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   Hyperparameter Sweep Manager - Help                      ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage:"
    echo -e "  ${CYAN}./sweep_manager.sh${NC}              # Run interactive sweep"
    echo -e "  ${CYAN}./sweep_manager.sh --help${NC}       # Show this help"
    echo -e "  ${CYAN}./sweep_manager.sh --status${NC}     # Check sweep status"
    echo ""
    echo "Workflow:"
    echo "  1. Review/edit hyperparameters in sweep.yaml"
    echo "  2. Create a new sweep on W&B"
    echo "  3. Choose single or parallel execution"
    echo "  4. Run sweep agents automatically"
    echo ""
    echo "Configuration Files:"
    echo "  • Sweep config: $SWEEP_YAML"
    echo "  • Sweep ID: $SWEEP_ID_FILE"
    echo "  • Logs: $LOG_DIR/"
    echo ""
}

################################################################################
# Status
################################################################################

show_status() {
    print_header
    print_section "Sweep Status"

    if [ -f "$SWEEP_ID_FILE" ]; then
        SWEEP_ID=$(cat "$SWEEP_ID_FILE")
        echo "Current Sweep ID: $SWEEP_ID"
        echo ""
        echo "Dashboard: https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps/sweeps/$SWEEP_ID"
        echo ""
    else
        print_warning "No active sweep found"
        echo "Run ./sweep_manager.sh to create a new sweep"
    fi

    # Count completed runs
    COMPLETED=0
    if [ -d "$WORK_DIR/outputs" ]; then
        COMPLETED=$(find "$WORK_DIR/outputs" -maxdepth 2 -type d -name "*-*" 2>/dev/null | wc -l)
    fi

    echo "Completed Runs: $COMPLETED"
    echo ""
}

################################################################################
# Main
################################################################################

main() {
    case "${1:-}" in
        --help|-h|help)
            show_help
            ;;
        --status|-s|status)
            show_status
            ;;
        *)
            print_header

            # Step 1: Review hyperparameters
            review_hyperparameters

            # Step 2: Create sweep
            create_sweep

            # Step 3: Choose execution mode
            choose_execution_mode

            # Step 4: Confirm and run
            echo ""
            if prompt_yes_no "Ready to start the sweep?"; then
                run_sweep_agents
                show_summary
            else
                print_warning "Sweep cancelled"
                echo "To start later, run:"
                echo -e "  ${CYAN}$PYTHON_BIN -m wandb agent $(cat $SWEEP_ID_FILE)${NC}"
            fi
            ;;
    esac
}

# Run main
main "$@"
