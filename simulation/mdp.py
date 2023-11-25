"""
Helper functions for abstractin a simulation as an MDP (Markov Decision Process)
including converting between simulation data and MDP constructs and recording of
states/observations, actions, and rewards.
"""

import torch
import jsbsim_properties as prp

"""
Extracts agent state data from the sim.
"""
def state_from_sim(sim):
  state = torch.zeros(15,)

  M_TO_FT = 0.3048
  
  # altitude
  state[0] = sim[prp.altitude_sl_ft] # z

  # velocity
  state[1] = sim[prp.v_east_fps] # x velocity
  state[2] = sim[prp.v_north_fps] # y velocity
  state[3] = -sim[prp.v_down_fps] # z velocity

  # angles
  state[4] = sim[prp.roll_rad] # roll
  state[5] = sim[prp.pitch_rad] # pitch
  state[6] = sim[prp.heading_rad] # yaw

  # angle rates
  state[7] = sim[prp.p_radps] # roll rate
  state[8] = sim[prp.q_radps] # pitch rate
  state[9] = sim[prp.r_radps] # yaw rate

  # next waypoint (relative)
  state[10] = 0.0
  state[11] = 0.0
  state[12] = 0.0

  return state

"""
Updates sim according to an action, assumes [action] is a 4-item tensor of
throttle, aileron cmd, elevator cmd, rudder cmd.
"""
def update_sim_from_action(sim, action):
  sim[prp.throttle_cmd] = action[0]
  sim[prp.aileron_cmd] = action[1]
  sim[prp.elevator_cmd] = action[2]
  sim[prp.rudder_cmd] = action[3]

"""
Enacts the [autopilot] on the current state of the simulation [sim].
Basically just updates sim throttle / control surfaces according to the autopilot.
"""
def enact_autopilot(sim, autopilot):
  state = state_from_sim(sim)
  action, log_prob = autopilot.get_controls(state)
  update_sim_from_action(sim, action)

  return state, action, log_prob

"""
The reward function for the bb autopilots. Since they won't know how to fly, 
they will get reward as follows (for every timestep prior to collision/termination):
  + 1 if moving with some velocity threshold vel_reward_threshold
  + alt_reward_threshold if flying above ground with some threshold alt_reward_threshold
  - quadratic control cost
"""
def bb_reward(action, next_state, collided, alt_reward_coeff=100, alt_reward_threshold=2, vel_reward_threshold=1):
  moving_reward = 1 if (next_state[3]**2 + next_state[4]**2 + next_state[5]**2) > vel_reward_threshold else 0
  alt_reward = alt_reward_coeff if next_state[2] > alt_reward_threshold else 0
  return moving_reward + alt_reward - torch.dot(action, action) if not collided else 0

"""
This class provides tooling for collecting MDP-related data about a simulation
rollout, including states/actions/rewards the agent experienced.
"""
class MDPDataCollector:
  def __init__(self, sim, reward_fn, expected_trajectory_length, state_dim=15, action_dim=4):
    # parameters
    self.sim = sim
    self.reward_fn = reward_fn

    # data accumulation
    self.states = torch.zeros(expected_trajectory_length, state_dim)
    self.actions = torch.zeros(expected_trajectory_length, action_dim)
    self.sample_log_probs = torch.zeros(expected_trajectory_length,)
    self.dones = torch.zeros(expected_trajectory_length, 1, dtype=torch.bool)
    self.rewards =  torch.zeros(expected_trajectory_length, 1)
    self.cum_reward = 0
  
  # t = timestep of the current state-action pair, in [0, expected_trajectory_length-1]
  def update(self, t, state, action, log_prob, next_state, collided):
    self.states[t, :] = state.T
    self.actions[t, :] = action.T
    self.sample_log_probs[t] = log_prob
    
    # this function should only be used when sim is still running, so not done
    # and not collided
    self.dones[t] = 0
    reward = self.reward_fn(action, next_state, collided)
    self.rewards[t] = reward
    self.cum_reward += reward
  
  # T = the number of timesteps that the trajectory executed for
  def terminate(self, T):
    # Make the last "done" flag 1 instead of 0 because that was the final state
    self.dones[T-1] = 1

    # Clip off the rest of the output
    self.states = self.states[:T]
    self.actions = self.actions[:T]
    self.sample_log_probs = self.sample_log_probs[:T]
    self.rewards = self.rewards[:T]

  def return_trajectory_data(self):
    return self.states, self.actions, self.sample_log_probs, self.rewards, self.dones
  
  def get_cum_reward(self):
    return self.cum_reward