from shared import HidePrints
from simulation.jsbsim_simulator import Simulation
from simulation.jsbsim_aircraft import Aircraft, x8
import simulation.jsbsim_properties as prp
from AirSimClient import *
from learning.autopilot import AutopilotLearner, SlewRateAutopilotLearner
import torch
import simulation.mdp as mdp
import os
import numpy as np
from shared import THROTTLE_CLAMP, AILERON_CLAMP, ELEVATOR_CLAMP, RUDDER_CLAMP
from vision.vision import Imager, VisionGuidanceSystem, VisionProcessor

"""
A class to integrate JSBSim and AirSim to roll-out a full trajectory for an
autopilot.
"""
class FullIntegratedSim:
  def __init__(self,
                aircraft: Aircraft,
                autopilot: SlewRateAutopilotLearner,
                sim_time: float,
                display_graphics: bool = True,
                agent_interaction_frequency: int = 15,
                airsim_frequency_hz: float = 392.0,
                sim_frequency_hz: float = 240.0,
                in_flight_reset: int = 0, # nonzero if we initialize from a non-takeoff reset distribution
                auto_deterministic: bool = True, # whether the autopilot picks its mode action (deterministic) or samples
                acquire_images: bool = False, # whether the sim acquires images
                vision_avoidance: bool = False,
                debug_level: int = 0):
    # Aircraft and autopilot
    self.aircraft = aircraft
    self.autopilot = autopilot
    self.auto_deterministic = auto_deterministic
    
    # Sim params
    self.in_flight_reset = in_flight_reset
    self.sim: Simulation = Simulation(sim_frequency_hz, aircraft, in_flight_reset=in_flight_reset, debug_level=debug_level)
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

    # 
    self.unhealthy_termination: bool = False
    self.unhealthy_penalty: float = 0.0

    self.initial_collision = False

    # Imaging/Vision
    self.acquire_images = acquire_images
    self.vision_avoidance = vision_avoidance
    self.imager = Imager(self.sim)
    self.avoidance_system = VisionGuidanceSystem()

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
    if self.in_flight_reset == 0:
      ic_position = [0, 0, -1.3411200046539307]
    elif self.in_flight_reset == 3:
      ic_position = [ 2.50183762e+02, -1.58594549e-01, -8.38120174e+00]
    elif self.in_flight_reset == 4:
      ic_position =  [ 3.97393585e+02,  4.14202549e-02, -3.14456406e+01]
    elif self.in_flight_reset == 5:
      ic_position = [ 5.47565369e+02, -7.68899895e-08, -5.24536057e+01]
    elif self.in_flight_reset == 6:
      ic_position = [ 681.66052246, 8.80005455, -69.21742249]
    
    ic_position = np.array(ic_position)
    current_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    retry_period = 10
    retry_counter = 0

    while (np.abs(ic_position - current_position)/ np.linalg.norm(current_position) > 0.00001).all():
      if retry_counter % retry_period == 0:
        self.sim.reinitialize()
      retry_counter += 1
      # print("cur pos", current_position)

      pose = self.sim.client.simGetVehiclePose()
      current_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    # Initialize to max throttle; the agent then learns when/how to decrease for cruise throttle
    self.sim[prp.throttle_cmd] = THROTTLE_CLAMP

    # If we're initializing in flight, randomly set controls
    if self.in_flight_reset > 0:
      self.sim[prp.throttle_cmd] = np.clip(np.random.normal(0.35,THROTTLE_CLAMP/10), 0, THROTTLE_CLAMP)
      self.sim[prp.aileron_cmd] = np.clip(np.random.normal(0, AILERON_CLAMP/200), -AILERON_CLAMP, AILERON_CLAMP)
      self.sim[prp.elevator_cmd] = np.clip(np.random.normal(0, ELEVATOR_CLAMP/20), -ELEVATOR_CLAMP, ELEVATOR_CLAMP)
      self.sim[prp.rudder_cmd] = np.clip(np.random.normal(0, RUDDER_CLAMP/20), -RUDDER_CLAMP, RUDDER_CLAMP)
    
    i = 0
    while i < update_num:
      # Do autopilot controls          
      try:
        #state, action, log_prob = mdp.enact_autopilot(self.sim, self.autopilot)
        state, action, log_prob, control = mdp.query_slewrate_autopilot(self.sim, self.autopilot, deterministic=self.auto_deterministic)
        if torch.isnan(state).any():
          break
        
        # Waypoint guidance
        if self.vision_avoidance and self.imager.acquired_enough():
          image, prev_image = self.imager.last_two_images()
          wp = self.avoidance_system.guide(VisionProcessor(image, prev_image, len(self.imager.images)), state)

          # utilize new wp for control instead if there is one
          if wp is not None:
            state[8:11] = wp
            action, log_prob = self.autopilot.get_deterministic_action(state)
            control = autopilot.get_control(action)
      except Exception as e:
        print(e)
        self.unhealthy_penalty = float(str(e).split(":")[-1])
        # print("penalty", self.unhealthy_penalty)
        self.unhealthy_termination = True
        # If enacting the autopilot fails, end the simulation immediately
        break
      
      # Update sim while waiting for next agent interaction
      while True:
        mdp.update_sim_from_slewrate_control(self.sim, control, self.autopilot)

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
          if self.acquire_images:
            self.imager.acquire_image()
        
        # Check for collisions via airsim and terminate if there is one
        if self.sim.get_collision_info().has_collided:
          if self.initial_collision:
            # print('Aircraft has collided.')
            self.done = True
            self.sim.reinitialize()
          else:
            # print("Aircraft completed initial landing")
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
      except Exception as e:
        # If we couldn't acquire the state, something crashed with jsbsim
        # We treat that as the end of the simulation and don't update the state
        print(f"\t\t\t\t\t\t\t\t\t\t{e}")
        next_state = state
        self.done = True
        self.unhealthy_termination = True
        self.unhealthy_penalty = float(str(e).split(":")[-1])

      # Data collection update for this step
      self.mdp_data_collector.update(int(i/self.agent_interaction_frequency)-1, state, action, 
                                     log_prob, next_state, self.unhealthy_termination, self.unhealthy_penalty)

      # End if collided
      if self.done == True:
        break
    
    self.done = True
    self.mdp_data_collector.terminate(int(i/self.agent_interaction_frequency))
    print('Simulation complete.')
    print('\t\t\t\t\t\t\t\t\t\tCum reward:', self.mdp_data_collector.cum_reward)
          

  """
  Replays a simulation
  """
  def simulation_replay(self, actions):
    # THESE ARE HARCODED by reading from prints of the initial controls state. TODO: make this a possible input
    self.sim[prp.throttle_cmd] = 0.5372 * THROTTLE_CLAMP
    self.sim[prp.aileron_cmd] = -0.0024 * AILERON_CLAMP
    self.sim[prp.elevator_cmd] = -0.0352 * ELEVATOR_CLAMP
    self.sim[prp.rudder_cmd] = -0.0080 * RUDDER_CLAMP
    
    i = 0
    for action in actions:
      state = mdp.state_from_sim(self.sim)
      # Do the control
      # mdp.update_sim_from_control(self.autopilot.get_control(action))
      control = self.autopilot.get_control(action)
      while True:
        # Run another sim step
        mdp.update_sim_from_slewrate_control(self.sim, control, self.autopilot)
        #print('control', control)
        self.sim.run()
        i += 1

        # Airsim update
        self.sim.update_airsim()

        # NOTE: for replays, the agent interaction frequency must match what
        # it was when the trajectory was created
        if i % self.agent_interaction_frequency == 0:
          print("Step")
          break
      try:
        next_state = mdp.state_from_sim(self.sim)
        self.mdp_data_collector.update(int(i/self.agent_interaction_frequency)-1, state, action, 
                                     0, next_state, self.unhealthy_termination, self.unhealthy_penalty)
      except:
        break
    print('cum reward', self.mdp_data_collector.cum_reward)

if __name__ == "__main__":
  # A one-minute simulation with an untrained autopilot
  autopilot = AutopilotLearner()
  integrated_sim = FullIntegratedSim(x8, autopilot, 60.0)
  integrated_sim.simulation_loop()