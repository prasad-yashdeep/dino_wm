#!/bin/bash

################################################################################
# W&B Hyperparameter Sweep Runner
################################################################################
#
# This script launches a Weights & Biases hyperparameter sweep for the 3-stage
# world model training pipeline.
#
# USAGE:
#   bash run_sweep.sh <sweep_id> [num_agents]
#
# EXAMPLES:
#   # Initialize new sweep and get ID
#   bash run_sweep.sh --init
#
#   # Run sweep agent (after getting ID from --init)
#   bash run_sweep.sh yashdeep18121-new-york-university/dino_wm_sweeps/<sweep_id>
#
#   # Run multiple agents in parallel
#   bash run_sweep.sh <sweep_id> 4
#
################################################################################

set -e  # Exit on error

cd "$(dirname "${BASH_SOURCE[0]}")"

# Configuration
PYTHON=${PYTHON:-python}
CONFIG_FILE="sweep.yaml"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       W&B Hyperparameter Sweep for World Model Training        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if sweep config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}❌ Error: $CONFIG_FILE not found!${NC}"
    echo "Please ensure sweep.yaml exists in the current directory."
    exit 1
fi

# Parse command-line arguments
if [ "$1" == "--init" ] || [ "$1" == "-init" ]; then
    # Initialize new sweep
    echo -e "${GREEN}📊 Initializing new W&B sweep...${NC}"
    echo "Configuration file: $CONFIG_FILE"
    echo ""

    # Show sweep configuration
    echo -e "${YELLOW}Sweep Parameters:${NC}"
    echo "  • Quantization Loss Weight: 0.5, 1.0, 2.0"
    echo "  • VCReg Loss Weight: 1, 5, 10"
    echo "  • State Quantizer LR: 5e-6 to 5e-3 (log-uniform)"
    echo "  • Predictor LR: 5e-6 to 5e-3 (log-uniform)"
    echo "  • Encoder LR: 5e-6 to 5e-3 (log-uniform)"
    echo ""

    # Initialize sweep and capture output
    echo -e "${YELLOW}Creating sweep...${NC}"
    SWEEP_OUTPUT=$(wandb sweep --entity yashdeep18121-new-york-university --project dino_wm_sweeps "$CONFIG_FILE" 2>&1)

    # Extract the sweep ID from output
    # Output format: "Create sweep with ID: yashdeep18121-new-york-university/dino_wm_sweeps/abcxyz"
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP '(?<=Create sweep with ID: )[^\s]+' | head -1)

    if [ -z "$SWEEP_ID" ]; then
        # Try alternative format
        SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP '(?<=wandb agent )[^\s]+' | head -1)
    fi

    if [ -z "$SWEEP_ID" ]; then
        echo -e "${YELLOW}⚠️  Could not extract sweep ID automatically${NC}"
        echo "W&B Output:"
        echo "$SWEEP_OUTPUT"
        echo ""
        echo "Try running manually:"
        echo -e "${GREEN}wandb sweep --project dino_wm_sweeps $CONFIG_FILE${NC}"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}✅ Sweep created successfully!${NC}"
    echo ""
    echo -e "${BLUE}Sweep ID:${NC} $SWEEP_ID"
    echo ""
    echo -e "${YELLOW}To run sweep agents:${NC}"
    echo -e "${GREEN}  bash run_sweep.sh $SWEEP_ID${NC}"
    echo ""
    echo -e "${YELLOW}To run multiple agents in parallel:${NC}"
    echo -e "${GREEN}  bash run_sweep.sh $SWEEP_ID 4${NC}"
    echo ""

elif [ -z "$1" ]; then
    echo -e "${YELLOW}❌ Error: Sweep ID required!${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  Initialize sweep: ${GREEN}bash run_sweep.sh --init${NC}"
    echo -e "  Run agent: ${GREEN}bash run_sweep.sh <sweep_id>${NC}"
    echo -e "  Run multiple agents: ${GREEN}bash run_sweep.sh <sweep_id> 4${NC}"
    exit 1

else
    # Run sweep agent
    SWEEP_ID="$1"
    NUM_AGENTS="${2:-1}"

    echo -e "${GREEN}🚀 Starting W&B Sweep Agent${NC}"
    echo -e "${BLUE}Sweep ID:${NC} $SWEEP_ID"
    echo -e "${BLUE}Number of agents:${NC} $NUM_AGENTS"
    echo ""

    # Function to run a single agent
    run_agent() {
        local agent_num=$1
        echo -e "${YELLOW}Starting agent $agent_num...${NC}"

        # Run the agent (will exit when sweep is complete)
        wandb agent "$SWEEP_ID" \
            --count 0 \
            2>&1 | tee "sweep_agent_${agent_num}.log"

        echo -e "${GREEN}✅ Agent $agent_num completed${NC}"
    }

    # Run single or multiple agents
    if [ "$NUM_AGENTS" -eq 1 ]; then
        echo -e "${YELLOW}Running single agent (will run until sweep completes)...${NC}"
        echo "Press Ctrl+C to stop the agent."
        echo ""
        run_agent 1
    else
        echo -e "${YELLOW}Running $NUM_AGENTS agents in parallel...${NC}"
        echo ""

        # Start all agents in background
        for i in $(seq 1 "$NUM_AGENTS"); do
            run_agent "$i" &
            PIDS+=($!)
            sleep 2  # Stagger startup
        done

        # Wait for all agents to complete
        echo -e "${YELLOW}Waiting for all agents to complete...${NC}"
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done

        echo -e "${GREEN}✅ All agents completed${NC}"
    fi
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Sweep Execution Complete                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📊 View results on Weights & Biases:${NC}"
echo -e "${GREEN}https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps${NC}"
