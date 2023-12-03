from simulation.jsbsim_simulator import Simulation
from simulation.jsbsim_aircraft import Aircraft, x8
import simulation.jsbsim_properties as prp
from learning.autopilot import AutopilotLearner
import torch
import simulation.mdp as mdp
import os
import numpy as np

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
                agent_interaction_frequency: int = 120,
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
    self.sim_steps = int(self.sim_time * self.sim_frequency_hz)
    
    # How often the agent (in terms of sim timesteps) selects a new
    # action (e.g. 10 means it selects a new action every 20 timesteps)
    self.agent_interaction_frequency = agent_interaction_frequency
    assert self.sim_steps % self.agent_interaction_frequency == 0

    # For data collection
    self.mdp_data_collector = mdp.MDPDataCollector(self, mdp.get_wp_reward(self.sim), int(self.sim_steps / self.agent_interaction_frequency))

    # Triggered when sim is complete
    self.done: bool = False

    self.initial_collision = False
  """
    Run loop for one simulation.
  """
  def simulation_loop(self):
    update_num = self.sim_steps  # how many simulation steps to update the simulation
    relative_update = self.airsim_frequency_hz / self.sim_frequency_hz  # rate between airsim and JSBSim
    graphic_update = 0


    # Occasionally, airsim lags behind jsbsim, causing aircraft to spawn inside obstacles.
    # Checking this and reinitializing helps to alleviate these cases.
    pose = self.sim.client.simGetVehiclePose()

    # Experimentally determined, in UE4 coordinate system
    ic_position = np.array([0, 0, -1.3411200046539307])
    current_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    retry_period = 10
    retry_counter = 0

    while (np.abs(ic_position - current_position) > np.finfo(float).eps).all():
      if retry_counter % retry_period == 0:
        self.sim.reinitialize()
      retry_counter += 1

      pose = self.sim.client.simGetVehiclePose()
      current_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])

    i = 0
    while i < update_num:
      # Do autopilot controls          
      try:
        state, action, log_prob = mdp.enact_autopilot(self.sim, self.autopilot)
        # state, action, log_prob = mdp.enact_predetermined_controls(self.sim, self.autopilot)
        if torch.isnan(state).any():
          break
      except Exception as e:
        print(e)
        # If enacting the autopilot fails, end the simulation immediately
        break
      
      # Update sim while waiting for next agent interaction
      while True:
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
          if self.initial_collision:
            print('Aircraft has collided.')
            self.done = True
            self.sim.reinitialize()
          else:
            print("Aircraft completed initial landing")
            self.initial_collision = True

        # Exit if sim is over or it's time for another agent interaction
        if self.done or i % self.agent_interaction_frequency == 0:
          break
      
      # Get new state
      try:
        next_state = mdp.state_from_sim(self.sim)
        if torch.isnan(next_state).any():
          next_state = state
          self.done = True
      except:
        # If we couldn't acquire the state, something crashed with jsbsim
        # We treat that as the end of the simulation and don't update the state
        next_state = state
        self.done = True

      # Data collection update for this step
      self.mdp_data_collector.update(int(i/self.agent_interaction_frequency)-1, state, action, log_prob, next_state, self.done)

      # End if collided
      if self.done == True:
        break
    
    self.done = True
    self.mdp_data_collector.terminate(int(i/self.agent_interaction_frequency))
    print('Simulation complete.')
    print('Cum reward:', self.mdp_data_collector.cum_reward)
          

  """
  Replays a simulation
  """
  def simulation_replay(self, actions):
    i = 0
    for action in actions:
      # Do the control
      mdp.update_sim_from_control(self.autopilot.get_control(action))

      while True:
        # Run another sim step
        self.sim.run()
        i += 1

        # Airsim update
        self.sim.update_airsim()

        # NOTE: for replays, the agent interaction frequency must match what
        # it was when the trajectory was created
        if i % self.agent_interaction_frequency == 0:
          print("Step")
          break

if __name__ == "__main__":
  # A one-minute simulation with an untrained autopilot
  autopilot = AutopilotLearner()
  integrated_sim = FullIntegratedSim(x8, autopilot, 60.0)
  integrated_sim.simulation_loop()