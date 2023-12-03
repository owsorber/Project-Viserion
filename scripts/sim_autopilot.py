"""
sim_autopilot file

This script expects the path relative to the data directory of a pth file
corresponding to the saved policy network of an autopilot learner
and rolls out a trajectory in sim for that autopilot learner.
"""

from learning.autopilot import AutopilotLearner
from simulation.simulate import FullIntegratedSim
from simulation.jsbsim_aircraft import x8
import sys
import os

# Get file path
policy_file_path = sys.argv[1]
file = os.path.join(*(["data"] + policy_file_path.split('/')))

# Initialize autopilot
autopilot = AutopilotLearner()
autopilot.init_from_saved(file)

# Play sim
integrated_sim = FullIntegratedSim(x8, autopilot, 60.0)
integrated_sim.simulation_loop()