"""
Helper functions for abstractin a simulation as an MDP (Markov Decision Process)
including converting between simulation data and MDP constructs and recording of
states/observations, actions, and rewards.
"""

import math
import torch
import simulation.jsbsim_properties as prp
import numpy as np
import os
from shared import THROTTLE_CLAMP, AILERON_CLAMP, ELEVATOR_CLAMP, RUDDER_CLAMP

"""
Extracts agent state data from the sim.
"""
def state_from_sim(sim):
  state = torch.zeros(15,)
  
  FT_TO_M = 0.3048
  
  # altitude
  state[0] = sim[prp.altitude_sl_ft] * FT_TO_M # z

  # speed
  state[1] = FT_TO_M * np.linalg.norm(np.array([sim[prp.v_north_fps], sim[prp.v_east_fps], -sim[prp.v_down_fps]]))
  
  # roll
  state[2] = sim[prp.roll_rad]

  # pitch 
  state[3] = sim[prp.pitch_rad] 
  
  # angle rates
  state[4] = sim[prp.p_radps] # roll rate
  state[5] = sim[prp.q_radps] # pitch rate
  state[6] = sim[prp.r_radps] # yaw rate

  state[7] = -sim[prp.v_down_fps] * FT_TO_M # z velocity

  # calculate next waypoint (relative)
  position = np.array(sim.get_local_position())
  waypoint = np.array(sim.waypoints[sim.waypoint_id])
  displacement = waypoint - position
  
  # if sim.waypoint_id == 3:
  #   input()
  #   print("lat", sim[prp.lat_geod_deg])
  #   print("lon", sim[prp.lng_geoc_deg])
  #   print("u_fps", sim[prp.u_fps])
  #   print("v_fps", sim[prp.v_fps])
  #   print("w_fps", sim[prp.w_fps])
  #   print("roll_rad", sim[prp.roll_rad])
  #   print("pitch_rad", sim[prp.pitch_rad])
  #   print("heading_rad", sim[prp.heading_rad])
  #   print("altitude", sim[prp.altitude_sl_ft])

  
  # determine whether next waypoint needs to be updated if we're currently inside one
  if np.linalg.norm(displacement) <= sim.waypoint_threshold:
    print(f"\t\t\t\t\t\t\t\t\t\tWaypoint {sim.waypoint_id} Hit!")
    sim.waypoint_id += 1
    sim.waypoint_entered = True
    sim.waypoint_reward = True
    if sim.waypoint_id == 7:
      raise Exception(f"Finish! :{50}")
    # raise Exception(f"Unhealthy state, do better. Penalty :{penalty}")


  # elif sim.waypoint_entered:
  #   prev_waypoint = np.array(sim.waypoints[sim.waypoint_id-1])
  #   prev_displacement = prev_waypoint - position
  #   if np.linalg.norm(prev_displacement) > sim.waypoint_threshold:
  #     # if we've left the previous waypoint, give reward for it
  #     print(f"\t\t\t\t\t\t\t\t\t\tWaypoint {sim.waypoint_id-1} Rewarded!")
  #     sim.waypoint_reward = True
  #     sim.waypoint_entered = False

  waypoint = np.array(sim.waypoints[sim.waypoint_id])
  displacement = waypoint - position

  # waypoint ground distance
  state[8] = np.linalg.norm(np.array(displacement[0:2]))

  # waypoint altitude distance
  state[9] = displacement[2]

  # heading diff from waypoint
  waypoint_heading = np.arctan2(displacement[1], displacement[0]) 
  heading = sim[prp.heading_rad] # yaw
  state[10] = waypoint_heading - heading
  if state[10] < -math.pi:
    state[10] += 2 * math.pi 
  elif state[10] > math.pi:
    state[10] -= 2* math.pi
  

  # print("\t\t\t\t\t\t\t\t\t\t YAW", heading/math.pi*180)
  # print("\t\t\t\t\t\t\t\t\t\t waypoint_heading", waypoint_heading/math.pi*180)


  # print("\t\t\t\t\t\t\t\t\t\theading", state[10]/math.pi*180)
  # print("\t\t\t\t\t\t\t\t\t\tdist", state[8])
  # print("\t\t\t\t\t\t\t\t\t\theight", state[9])
  

  # Controls state
  state[11] = sim[prp.throttle_cmd] / THROTTLE_CLAMP
  state[12] = sim[prp.aileron_cmd] / AILERON_CLAMP
  state[13] = sim[prp.elevator_cmd] / ELEVATOR_CLAMP
  state[14] = sim[prp.rudder_cmd] / RUDDER_CLAMP


  # Record whether takeoff completed
  if state[0] >= 2:
    if not sim.completed_takeoff:
      print("\t\t\t\t\t\t\t\t\t\tTake off!")
    sim.completed_takeoff = True

  unhealthy_state, penalty = is_unhealthy_state(state)
  if unhealthy_state: 
    raise Exception(f"Unhealthy state, do better. Penalty :{penalty}")


  return state

"""
Returns a bool for whether the state is unhealthy
"""
def is_unhealthy_state(state):
  MAX_BANK =  math.pi / 3
  MAX_PITCH =  math.pi / 4
  if np.cos(state[10]) < 0:
    return True, -10
  if not -MAX_BANK < state[2] < MAX_BANK:
    return True, -50
  if not -MAX_PITCH < state[3] < MAX_PITCH:
    return True, -25
  return False, 0
  

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
  

# Get the state/action/log_prob and control from the slewrate autopilot
count = 0
def query_slewrate_autopilot(sim, autopilot, deterministic=False):
  global count
  state = state_from_sim(sim)
  action, log_prob = autopilot.get_deterministic_action(state) if deterministic else autopilot.get_action(state)
  control = autopilot.get_control(action)
  
  #print('state', state)
  #print(state[10:])
  #print('elevator', state[13])
  """
  if state[1] <= 18 and not sim.completed_takeoff:
    print('throttle', state[11])
    control = torch.Tensor([1, 0, 0, 0]) 
  elif state[1] <= 20 and not sim.completed_takeoff:
    print('throttle down', state[11])
    control = torch.Tensor([-1, 0, 0, 0]) 
  elif count < 20:
    #print('throttle', state[11])
    print('holding pitch', state[13])
    count += 1
    control = torch.Tensor([0, 0, 0, 0])
  elif count < 40:
    print('pitch up')
    count += 1
    control = torch.Tensor([0, 0, -1, 0])
  elif count < 400:
    count += 1 
    control = torch.Tensor([0, 0, 0, 0])
  else:
    control = torch.Tensor([0, 0, 1, 0])
  """
  

  return state, action, log_prob, control

# Called every single sim step to enact slewrate
# Assumes autopilot is a SlewRateAutopilotLearner
def update_sim_from_slewrate_control(sim, control, autopilot):
  sim[prp.throttle_cmd] = np.clip(sim[prp.throttle_cmd] + control[0] * autopilot.throttle_slew_rate, 0, THROTTLE_CLAMP)
  sim[prp.aileron_cmd] = np.clip(sim[prp.aileron_cmd] + control[1] * autopilot.aileron_slew_rate, -AILERON_CLAMP, AILERON_CLAMP)
  sim[prp.elevator_cmd] = np.clip(sim[prp.elevator_cmd] + control[2] * autopilot.elevator_slew_rate, -ELEVATOR_CLAMP, ELEVATOR_CLAMP)
  sim[prp.rudder_cmd] = np.clip(sim[prp.rudder_cmd] + control[3] * autopilot.rudder_slew_rate, -RUDDER_CLAMP, RUDDER_CLAMP)

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
  state = state_from_sim(sim)
  action, log_prob = autopilot.get_action(state)
  update_sim_from_control(sim, autopilot.get_control(action))

  return state, action, log_prob


# Takes in the action outputted directly from the network and outputs the 
# normalized quadratic action cost from 0-1
def quadratic_action_cost(action):
  action_cost_weights = torch.tensor([3.0, 10.0, 5.0, 1.0])
  action[0] = 0.5 * (action[0] + 1) # converts throttle to be 0-1
  return float(torch.dot(action ** 2, action_cost_weights).detach() / sum(action_cost_weights)) # divide by 4 to be 0-1

# Takes in the action outputted directly from the network and outputs the 
# normalized quadratic action cost from 0-1
def quadratic_control_cost(control):
  # control_cost_weights = torch.tensor([1.0, 10.0, 5.0, 1.0])
  # 0.05882352941	0.5882352941	0.2941176471	0.05882352941
  control_cost_weights = torch.tensor([0.6, 10.0, 0.3, 0.06])
  return float(torch.dot(control ** 2, control_cost_weights).detach()) # divide by 4 to be 0-1

"""
The reward function for the bb autopilots. Since they won't know how to fly, 
they will get reward as follows (for every timestep prior to collision/termination):
  + 1 if moving with some velocity threshold vel_reward_threshold
  + alt_reward_threshold if flying above ground with some threshold alt_reward_threshold
  - action_coeff * the quadratic control cost
  !!! WARNING: Deprecated
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
  !!! WARNING: Deprecated
"""
def new_init_wp_reward(action, next_state, collided, wp_coeff=1, action_coeff=1, alt_reward_threshold=5):
  alt_reward = 1 if next_state[2] > alt_reward_threshold else 0
  action_cost = action_coeff * quadratic_action_cost(action)

  waypoint_rel_unit = next_state[10:13] / torch.norm(next_state[10:13])
  vel = next_state[1:4]
  toward_waypoint_reward = wp_coeff * float(torch.dot(vel, waypoint_rel_unit).detach())
  return toward_waypoint_reward + alt_reward - action_cost if not collided else 0

"""
A reward function getter.
"""
a = 0
def get_wp_reward(sim):
  def wp_reward(action, next_state, collided, wp_coeff=0.5, action_coeff=0.5, alt_coeff=0.5, roll_coeff=0.9):
    global a
    if sim.waypoint_reward:
      sim.waypoint_reward = False
      wp_reward = 50
    else: wp_reward = 0
    action_cost = action_coeff * quadratic_control_cost(next_state[11:15])
    #print('\t\t\t\t\t\tACTION COST:', action_cost, next_state[11:15])
    # waypoint_rel_unit = torch.nn.functional.normalize(next_state[10:13], dim=0)
    # vel = next_state[1:4]
    # toward_waypoint_reward = wp_coeff * float(torch.dot(vel, waypoint_rel_unit).detach())
    # cosine(relative heading) * ground speed
    #toward_waypoint_reward = min(wp_coeff * torch.cos(next_state[10]) * np.sqrt(next_state[1] ** 2 - next_state[7] ** 2), 4)
    away_waypoint_cost = wp_coeff * abs(next_state[10])
    alitude_diff_cost = alt_coeff * abs(np.arctan2(next_state[9], next_state[8]) - next_state[3]) # pitch "error": relative pitch between current pitch and straight line to waypoint
    roll_cost = roll_coeff * (abs(next_state[2]))**3
    #print('\t\t\t\t\t\AWAY WP COST:', away_waypoint_cost)
    #print('\t\t\t\t\t\ALT COST:', alitude_diff_cost)
    if not sim.takeoff_rewarded and sim.completed_takeoff:
      takeoff_reward = 25
      sim.takeoff_rewarded = True
    else:
      takeoff_reward = 0
    
    staying_alive_reward = 0.5
    # toward_waypoint_reward = 0
    rewards = staying_alive_reward + wp_reward + takeoff_reward
    costs = action_cost + away_waypoint_cost + alitude_diff_cost + roll_cost
    # if a % 100 == 0:
    #   print("\t\t\t\t\t\t\t\taway_waypoint_cost", away_waypoint_cost, "alitude_diff_cost",alitude_diff_cost)
    #   print("\t\t\t\t\t\t\t\troll_cost", roll_cost, "action_cost",action_cost, "action",next_state[11:15])
    #   print("\t\t\t\t\t\t\t\tRewards:",rewards, "Costs:", -costs)
    #   print("\t\t\t\t\t\t\t\tNet Reward", rewards - costs if not collided else 0)
    # toward_waypoint_reward = torch.min(torch.tensor(10), 0.01 * 1 / torch.max(torch.tensor(0.1), torch.sum(next_state[10:13]**2)))
    return rewards - costs if not collided else 0
  return wp_reward

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
    self.next_states = torch.zeros(expected_trajectory_length, state_dim)
    self.actions = torch.zeros(expected_trajectory_length, action_dim)
    self.sample_log_probs = torch.zeros(expected_trajectory_length,)
    self.dones = torch.zeros(expected_trajectory_length, 1, dtype=torch.bool)
    self.rewards =  torch.zeros(expected_trajectory_length, 1)
    self.cum_reward = 0
  
  # t = timestep of the current state-action pair, in [0, expected_trajectory_length-1]
  def update(self, t, state, action, log_prob, next_state, collided, unhealthy_penalty=0.0):
    self.states[t, :] = torch.transpose(state, 0, -1)
    self.next_states[t, :] = torch.transpose(next_state, 0, -1)
    self.actions[t, :] = torch.transpose(action, 0, -1)
    self.sample_log_probs[t] = log_prob
    
    # this function should only be used when sim is still running, so not done
    # and not collided
    self.dones[t] = 0
    reward = self.reward_fn(action, next_state, collided) + unhealthy_penalty
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