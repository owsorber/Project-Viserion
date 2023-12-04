
from learning.autopilot import StochasticPIDControlLearner
from simulation.simulate import FullIntegratedSim
from simulation.jsbsim_aircraft import x8


if __name__ == "__main__":
  # A one-minute simulation with an untrained autopilot
  autopilot = StochasticPIDControlLearner(dt = 1/240)
  integrated_sim = FullIntegratedSim(x8, autopilot, 60.0, state_dim=13, action_dim=3)
  integrated_sim.simulation_loop()