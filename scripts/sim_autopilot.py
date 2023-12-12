"""
sim_autopilot file

This script expects the path relative to the data directory of a pth file
corresponding to the saved policy network of an autopilot learner
and rolls out a trajectory in sim for that autopilot learner.
"""

from learning.autopilot import AutopilotLearner, SlewRateAutopilotLearner
from simulation.simulate import FullIntegratedSim
from simulation.jsbsim_aircraft import x8
import sys
import os

# Get file path
policy_file_path = sys.argv[1]
file = os.path.join(*(["data"] + policy_file_path.split('/')))

# Get in-flight reset
in_flight_reset = int(sys.argv[2])

# Initialize autopilot
autopilot = SlewRateAutopilotLearner()
autopilot.init_from_saved(file)

# Play sim
integrated_sim = FullIntegratedSim(x8, autopilot, 60.0, auto_deterministic=True, in_flight_reset=in_flight_reset)
integrated_sim.simulation_loop()

integrated_sim.mdp_data_collector.save(os.path.join('vision'), 'data')