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
  
  # position
  state[0] = sim[prp.lng_travel_m] # x
  state[1] = sim[prp.lat_travel_m] # y
  state[2] = sim[prp.altitude_sl_ft] # z

  # velocity
  state[3] = sim[prp.v_east_fps] # x velocity
  state[4] = sim[prp.v_north_fps] # y velocity
  state[5] = -sim[prp.v_down_fps] # z velocity

  # angles
  state[6] = sim[prp.roll_rad] # roll
  state[7] = sim[prp.pitch_rad] # pitch
  state[8] = sim[prp.heading_rad] # yaw

  # angle rates
  state[9] = sim[prp.p_radps] # roll rate
  state[10] = sim[prp.q_radps] # pitch rate
  state[11] = sim[prp.r_radps] # yaw rate

  # next waypoint (relative)
  state[12] = 0.0
  state[13] = 0.0
  state[14] = 0.0

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
  action = autopilot.get_controls(state)
  update_sim_from_action(sim, action)