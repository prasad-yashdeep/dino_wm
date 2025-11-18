#!/usr/bin/env python
"""
Wrapper script for W&B Hyperparameter Sweep

Converts W&B arguments (--key=value format) to Hydra format (key=value)
and calls train.py with sweep outputs directed to sweep_outputs/runs
"""

import sys
import subprocess
import os

# Detect if we're in the sweep directory and navigate to parent
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Change to parent directory so train.py can be found
os.chdir(parent_dir)

# Convert W&B arguments to Hydra format
# W&B passes: --key=value
# Hydra expects: key=value

hydra_args = []
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        # Remove the -- prefix
        hydra_args.append(arg[2:])
    else:
        hydra_args.append(arg)

# Add sweep-specific output directory (saves to sweep_outputs/runs instead of outputs)
# This keeps all sweep runs organized in one place
hydra_args.append('outputs_dir=sweep_outputs/runs')

# Call train.py with converted arguments (from parent directory)
cmd = ['python', 'train.py'] + hydra_args
print(f"Running from: {os.getcwd()}")
print(f"Running: {' '.join(cmd)}")
sys.exit(subprocess.call(cmd))
