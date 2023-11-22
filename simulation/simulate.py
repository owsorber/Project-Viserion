from jsbsim_simulator import Simulation
from jsbsim_aircraft import Aircraft, x8
import jsbsim_properties as prp
from learning.autopilot import AutopilotLearner
import mdp

"""
A class to integrate JSBSim and AirSim to roll-out a full trajectory for an
autopilot.
"""
class FullIntegratedSim:
  def __init__(self,
                aircraft: Aircraft,
                autopilot: AutopilotLearner,
                sim_time: float,
                display_graphics: bool = True,
                agent_interaction_frequency: float = 12.0,
                airsim_frequency_hz: float = 392.0,
                sim_frequency_hz: float = 240.0,
                init_conditions: bool = None,
                debug_level: int = 0):
    # Aircraft and autopilot
    self.aircraft = aircraft
    self.autopilot = autopilot
    
    # Sim params
    self.sim: Simulation = Simulation(sim_frequency_hz, aircraft, init_conditions, debug_level)
    self.sim_time = sim_time
    self.display_graphics = display_graphics
    self.sim_frequency_hz = sim_frequency_hz
    self.airsim_frequency_hz = airsim_frequency_hz

    # For data collection
    self.mdp_data_collector = mdp.MDPDataCollector(self, mdp.bb_reward, self.sim_time * self.sim_frequency_hz)
    
    # Currently unused, but could be used for how often the agent selects a new action
    self.agent_interaction_frequency = agent_interaction_frequency

    # Triggered when sim is complete
    self.done: bool = False
  
  """
    Run loop for one simulation.
  """
  def simulation_loop(self):
    update_num = int(self.sim_time * self.sim_frequency_hz)  # how many simulation steps to update the simulation
    relative_update = self.airsim_frequency_hz / self.sim_frequency_hz  # rate between airsim and JSBSim
    graphic_update = 0

    i = 0
    while i < update_num:
      # Do autopilot controls
      state, action, log_prob = mdp.enact_autopilot(self.sim, self.autopilot)

      # Run another sim step
      self.sim.run()

      # Increment timestep
      i += 1

      # Airsim update
      graphic_i = relative_update * i
      graphic_update_old = graphic_update
      graphic_update = graphic_i // 1.0
      if self.display_graphics and graphic_update > graphic_update_old:
        self.sim.update_airsim()
      
      # Check for collisions via airsim and terminate if there is one
      if self.sim.get_collision_info().has_collided:
        print('Aircraft has collided.')
        self.done = True

      # Get new state
      next_state = mdp.state_from_sim(self.sim)

      # Data collection update for this step
      self.mdp_data_collector.update(i-1, state, action, log_prob, next_state, self.done)

      # End if collided
      if self.done == True:
        break
    
    self.done = True
    self.mdp_data_collector.terminate(i)
    print('Simulation complete.')

if __name__ == "__main__":
  # A one-minute simulation with an untrained autopilot
  autopilot = AutopilotLearner()
  integrated_sim = FullIntegratedSim(x8, autopilot, 60.0)
  integrated_sim.simulation_loop()