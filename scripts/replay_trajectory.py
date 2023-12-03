"""
replay_trajectory file

This script expects the path relative to the data directory of a pickle file
corresponding to the stored states, actions, and rewards of a single trajectory
and rolls out that trajectory in sim by replaying the actions.

Note: this assumes the same starting location as the original trajectory and
that actions deterministically affect the state, which should be true
"""

import sys
import os
import torch
from learning.autopilot import AutopilotLearner
from simulation.simulate import FullIntegratedSim
from simulation.jsbsim_aircraft import x8

# Get file path
traj_file_path = sys.argv[1]
file = os.path.join(*(["data"] + traj_file_path.split('/')))
print(file)

# Load states, actions, rewards
states, actions, rewards = torch.load(file)

# Play sim
integrated_sim = FullIntegratedSim(x8, AutopilotLearner(), 60.0)
integrated_sim.simulation_replay(actions)