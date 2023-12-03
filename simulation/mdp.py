"""
Helper functions for abstractin a simulation as an MDP (Markov Decision Process)
including converting between simulation data and MDP constructs and recording of
states/observations, actions, and rewards.
"""

import torch
import simulation.jsbsim_properties as prp
import numpy as np
import os

"""
Extracts agent state data from the sim.
"""
def state_from_sim(sim, debug=False):
  state = torch.zeros(13,)
  
  FT_TO_M = 0.3048
  
  # altitude
  state[0] = sim[prp.altitude_sl_ft] * FT_TO_M # z

  # velocity
  state[1] = sim[prp.v_north_fps] * FT_TO_M # x velocity
  state[2] = sim[prp.v_east_fps] * FT_TO_M # y velocity
  state[3] = -sim[prp.v_down_fps] * FT_TO_M # z velocity

  # angles
  state[4] = sim[prp.roll_rad] # roll
  state[5] = sim[prp.pitch_rad] # pitch
  state[6] = sim[prp.heading_rad] # yaw

  # angle rates
  state[7] = sim[prp.p_radps] # roll rate
  state[8] = sim[prp.q_radps] # pitch rate
  state[9] = sim[prp.r_radps] # yaw rate

  # next waypoint (relative)
  position = np.array(sim.get_local_position())
  waypoint = np.array(sim.waypoints[sim.waypoint_id])
  
  displacement = waypoint - position

  if state[0] >= 2:
    if not sim.completed_takeoff:
      print("\t\t\t\t\t\t\t\t\t\tTake off!")
    sim.completed_takeoff = True

  if np.linalg.norm(displacement) <= sim.waypoint_threshold:
    print(f"\t\t\t\t\t\t\t\t\t\tWaypoint {sim.waypoint_id} Hit!")
    sim.waypoint_id += 1
    sim.waypoint_rewarded = False

  state[10] = displacement[0]
  state[11] = displacement[1]
  state[12] = displacement[2]

  if debug:
    print('State!')
    print('Altitude:', state[0])
    print('Velocity: (', state[1], state[2], state[3], ')')
    print('Roll:', state[4], '; Pitch:', state[5], '; Yaw:', state[6])
    print('RollRate:', state[7], '; PitchRate:', state[8], '; YawRate:', state[9])
    print('Relative WP: (', state[10], state[11], state[12], ')')

  return state


"""
Updates sim according to a control, assumes [control] is a 4-item tensor of
throttle, aileron cmd, elevator cmd, rudder cmd.
"""
def update_sim_from_control(sim, control, debug=False):
  sim[prp.throttle_cmd] = control[0]
  sim[prp.aileron_cmd] = control[1]
  sim[prp.elevator_cmd] = control[2]
  sim[prp.rudder_cmd] = control[3]
  if debug:
    print('Control Taken:', control)

"""
Follows a predetermined sequence of controls, instead of using autopilot.
"""
def enact_predetermined_controls(sim, autopilot):
  t = sim.t

  durations = [400, 200, 100, 300, 900, 400, 400,
                  1500, 400, 1500, 400]
  controls = [[0.8, 0., 0., 0.],
          [0.7, 0., 0., 0.],
          [0.5, 0., -0.2, 0.],
          [0.4, 0., -0.13, 0.],
          [0.29, 0., -0.01, 0.],
          [0.29, 0., -0.01, 0.],
          [0.29, 0., -0.01, 0.],
          [0.22, 0.01, -0.01, 0.3],
          [0.22, -0.01, -0.005, 0.0],
          [0.22, 0.01, -0.01, 0.3],
          [0.22, -0.01, -0.005, 0.0]]
  durations = np.cumsum(durations)
  control = [0.22, 0.0, 0.0, 0.0]

  for i, duration in enumerate(durations):
    if t < duration:
      control = controls[i]
      break
  t += 1
  if t >= durations[-1]: 
   t = 0
  control = torch.tensor(control)
  update_sim_from_control(sim, control)

  sim.t = t

  return state_from_sim(sim), control, 0
  

"""
Enacts the [autopilot] on the current state of the simulation [sim].
Basically just updates sim throttle / control surfaces according to the autopilot.
"""
def enact_autopilot(sim, autopilot):
  state = state_from_sim(sim, debug=False)
  action, log_prob = autopilot.get_action(state)
  update_sim_from_control(sim, autopilot.get_control(action))

  return state, action, log_prob

# Takes in the action outputted directly from the network and outputs the 
# normalized quadratic action cost from 0-1
def quadratic_action_cost(action):
  action_cost_weights = torch.tensor([1.0, 20.0, 10.0, 1.0])
  action[0] = 0.5 * (action[0] + 1) # converts throttle to be 0-1
  return float(torch.dot(action ** 2, action_cost_weights).detach()) / sum(action_cost_weights) # divide by 4 to be 0-1

"""
The reward function for the bb autopilots. Since they won't know how to fly, 
they will get reward as follows (for every timestep prior to collision/termination):
  + 1 if moving with some velocity threshold vel_reward_threshold
  + alt_reward_threshold if flying above ground with some threshold alt_reward_threshold
  - action_coeff * the quadratic control cost
"""
def bb_reward(action, next_state, collided, alt_reward_coeff=10, action_coeff=1, alt_reward_threshold=2, vel_reward_threshold=1):
  moving_reward = 1 if (next_state[3]**2 + next_state[4]**2 + next_state[5]**2) > vel_reward_threshold else 0
  alt_reward = alt_reward_coeff if next_state[2] > alt_reward_threshold else 0
  action_cost = action_coeff * quadratic_action_cost(action)
  return moving_reward + alt_reward - action_cost if not collided else 0

"""
A reward function that tries to incentivize the plane to go towards the waypoint
at each timestep, while also rewarding for being above ground and penalizing
for high control effort
"""
def new_init_wp_reward(action, next_state, collided, wp_coeff=1, action_coeff=1, alt_reward_threshold=5):
  alt_reward = 1 if next_state[2] > alt_reward_threshold else 0
  action_cost = action_coeff * quadratic_action_cost(action)

  waypoint_rel_unit = next_state[10:13] / torch.norm(next_state[10:13])
  vel = next_state[1:4]
  toward_waypoint_reward = wp_coeff * float(torch.dot(vel, waypoint_rel_unit).detach())
  return toward_waypoint_reward + alt_reward - action_cost if not collided else 0

"""
A reward function.
"""
def get_wp_reward(sim):
  def wp_reward(action, next_state, collided, wp_coeff=0.1, action_coeff=10):
    if not sim.waypoint_rewarded:
      sim.waypoint_rewarded = True
      wp_reward = 1_000
    else: wp_reward = 0
    action_cost = action_coeff * quadratic_action_cost(action)
    waypoint_rel_unit = torch.nn.functional.normalize(next_state[10:13], dim=0)
    vel = next_state[1:4]
    toward_waypoint_reward = wp_coeff * float(torch.dot(vel, waypoint_rel_unit).detach())
    toward_waypoint_reward = 0
    if not sim.takeoff_rewarded and sim.completed_takeoff:
      takeoff_reward = 500
      sim.takeoff_rewarded = True
    else:
      takeoff_reward = 0
    # toward_waypoint_reward = 0
    # toward_waypoint_reward = torch.min(torch.tensor(10), 0.01 * 1 / torch.max(torch.tensor(0.1), torch.sum(next_state[10:13]**2)))
    #print("wp_reward: ", wp_reward, " toward_waypoint_reward: ", toward_waypoint_reward)
    #print("alt_reward: ", alt_reward, " action_cost: ", -action_cost)
    #print("\t reward", wp_reward + toward_waypoint_reward + alt_reward - action_cost if not collided else 0)
    return wp_reward + toward_waypoint_reward + takeoff_reward - action_cost if not collided else 0
  return wp_reward

"""
This class provides tooling for collecting MDP-related data about a simulation
rollout, including states/actions/rewards the agent experienced.
"""
class MDPDataCollector:
  def __init__(self, sim, reward_fn, expected_trajectory_length, state_dim=13, action_dim=4):
    # parameters
    self.sim = sim
    self.reward_fn = reward_fn

    # data accumulation
    self.states = torch.zeros(expected_trajectory_length, state_dim)
    self.next_states = torch.zeros(expected_trajectory_length, state_dim)
    self.actions = torch.zeros(expected_trajectory_length, action_dim)
    self.sample_log_probs = torch.zeros(expected_trajectory_length,)
    self.dones = torch.zeros(expected_trajectory_length, 1, dtype=torch.bool)
    self.rewards =  torch.zeros(expected_trajectory_length, 1)
    self.cum_reward = 0
  
  # t = timestep of the current state-action pair, in [0, expected_trajectory_length-1]
  def update(self, t, state, action, log_prob, next_state, collided):
    self.states[t, :] = torch.transpose(state, 0, -1)
    self.next_states[t, :] = torch.transpose(next_state, 0, -1)
    self.actions[t, :] = torch.transpose(action, 0, -1)
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
    self.dones = self.dones[:T]
    self.states = self.states[:T]
    self.next_states = self.next_states[:T]
    self.actions = self.actions[:T]
    self.sample_log_probs = self.sample_log_probs[:T]
    self.rewards = self.rewards[:T]

  # Save the states, actions, and rewards to a file in data/[dir] with [name].pkl
  # Can later use the following to load:
  def save(self, dir, name):
    file = os.path.join('data', dir, name + '.pkl')
    torch.save([self.states, self.actions, self.rewards], file)

  def get_trajectory_data(self):
    return self.states, self.next_states, self.actions, self.sample_log_probs, self.rewards, self.dones
  
  def get_cum_reward(self):
    return self.cum_reward