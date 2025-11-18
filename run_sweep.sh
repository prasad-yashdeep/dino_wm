#!/bin/bash

################################################################################
# Hyperparameter Sweep Launcher
#
# Simple wrapper that runs the sweep manager from the sweep/ directory
# Usage: ./run_sweep.sh
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="$SCRIPT_DIR/sweep/sweep_manager.sh"

if [ ! -f "$SWEEP_SCRIPT" ]; then
    echo "❌ Error: sweep/sweep_manager.sh not found"
    echo "Location: $SWEEP_SCRIPT"
    exit 1
fi

# Run the sweep manager
exec "$SWEEP_SCRIPT" "$@"
