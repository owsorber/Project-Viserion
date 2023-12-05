import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict
from torchrl.modules import ValueOperator
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from learning.autopilot import StochasticAutopilotLearner, SlewRateAutopilotLearner
from simulation.simulate import FullIntegratedSim
from simulation.jsbsim_aircraft import x8
import numpy as np
import os

device = "cpu" if not torch.has_cuda else "cuda:0"

"""
Gathers rollout data and returns it in the way the Proximal Policy Optimization loss_module expects
"""
def gather_rollout_data(autopilot_learner, policy_num, num_trajectories=100, sim_time=60.0):
  # Do rollouts
  data_size = 0
  observations = torch.empty(0)
  next_observations = torch.empty(0)
  actions = torch.empty(0)
  sample_log_probs = torch.empty(0)
  rewards = torch.empty(0)
  dones = torch.empty(0, dtype=torch.bool)
  best_cum_reward = -float('inf')
  worst_cum_reward = float('inf')
  total_cum_reward = 0
  total_timesteps = 0
  for t in range(num_trajectories):
    in_flight_reset =  np.random.rand() <= 1.0
    integrated_sim = FullIntegratedSim(x8, autopilot_learner, sim_time, in_flight_reset)
    integrated_sim.simulation_loop()
    
    # Acquire data
    observation, next_observation, action, sample_log_prob, reward, done = integrated_sim.mdp_data_collector.get_trajectory_data()
    cum_reward = integrated_sim.mdp_data_collector.get_cum_reward()
    total_cum_reward += cum_reward
    total_timesteps += reward.shape[0]
    
    # Save trajectory if worst or best so far
    if cum_reward > best_cum_reward:
      integrated_sim.mdp_data_collector.save(os.path.join('ppo', 'trajectories'), 'best_rollout#' + str(policy_num))
      best_cum_reward = cum_reward
    if cum_reward < worst_cum_reward:
      integrated_sim.mdp_data_collector.save(os.path.join('ppo', 'trajectories'), 'worst_rollout#' + str(policy_num))
      worst_cum_reward = cum_reward

    # Add to the data
    observations = torch.cat((observations, observation))
    next_observations = torch.cat((next_observations, next_observation))
    actions = torch.cat((actions, action))
    sample_log_probs = torch.cat((sample_log_probs,sample_log_prob))
    rewards = torch.cat((rewards, reward / torch.std(reward)))
    dones = torch.cat((dones, done))
    data_size += observation.shape[0]
    
  # Each entry tensor should be data_size x d where d is the dimension of
  # that entry for one step in a rollout.
  data = TensorDict({
    "observation": observations,
    "action": TensorDict({
      "throttle": actions[:,0].detach(),
      "aileron": actions[:,1].detach(),
      "elevator": actions[:,2].detach(),
      "rudder": actions[:,3].detach(),
      "throttle_log_prob": torch.zeros(data_size).detach(),
      "aileron_log_prob": torch.zeros(data_size).detach(),
      "elevator_log_prob": torch.zeros(data_size).detach(),
      "rudder_log_prob": torch.zeros(data_size).detach(),
      }, [data_size,]),
    "sample_log_prob": sample_log_probs.detach(), # log probability that each action was selected
    # "sample_log_prob": TensorDict({
    # }, [data_size,]), # log probability that each action was selected
    ("next", "done"): dones,
    ("next", "terminated"): dones,
    ("next", "reward"): rewards,
    ("next", "observation"): next_observations,
  }, [data_size,])

  # Write to stats file
  stats_file = open(os.path.join('data', 'ppo', 'stats.txt'), 'a')
  stats_file.write('Policy Number #' + str(policy_num) + ':\n')
  stats_file.write('Average Time Steps =' + str(total_timesteps/num_trajectories) + ':\n')
  stats_file.write('Average Reward: ' + str(total_cum_reward/num_trajectories) + '\n')
  stats_file.write('Best Reward: ' + str(best_cum_reward) + '\n')
  stats_file.write('Worst Reward: ' + str(worst_cum_reward) + '\n')
  stats_file.write('\n\n')

  return data, data_size

"""
Outputs an RL module for value estimation that holds a neural network trained to
estimate the value of a state/observation of size n_obs. This value estimator 
is used as a critic to subtract a baseline from the cumulative reward of the 
action taken during PPO learning.
"""
def make_value_estimator_module(n_obs):
  global device

  value_net = nn.Sequential(
    nn.Linear(n_obs, 2*n_obs, device=device),
    nn.Tanh(),
    nn.Linear(2*n_obs, 1, device=device), # one value is computed for the state
  )

  return ValueOperator(
      module=value_net,
      in_keys=["observation"],
  )

"""
Perform PPO to improve the policy over ONE "generation" of rollouts.
  Inspired By: https://pytorch.org/rl/tutorials/coding_ppo.html

In a generation, we generate [num_trajectories] trajectories using the policy 
defined by [autopilot_learner] to gather an on-policy dataset. Then, we learn 
from this data to update the policy over [num_epochs] epochs.

In each epoch, the dataset is traversed through entriely  once over a series of 
batches of [batch_size], where in each batch the PPO gradient update is performed 
using the [loss_module] which computes the loss from the  policy for that 
dataset, the critic (value function estimator), and an [optimizer].

This function returns the updated policy.
"""
def train_ppo_once(policy_num, autopilot_learner, loss_module, advantage_module, optimizer, num_trajectories, num_epochs, batch_size):
  # Rollout policy to gather trajectories
  dataset, data_size = gather_rollout_data(autopilot_learner, policy_num, num_trajectories)

  # Define replay buffer to store trajectory data in during training
  replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(data_size),
    sampler=SamplerWithoutReplacement(),
  )
  
  for epoch in range(num_epochs):
    print("Training Epoch", epoch+1)
    # Re-compute advantage at each epoch as its value depends on the value
    # network which is updated in the inner loop
    with torch.no_grad():
      advantage_module(dataset)

    # Gather the data into a replay buffer for sampling
    data_view = dataset.reshape(-1)
    replay_buffer.extend(data_view.cpu())
    for b in range(0, data_size // batch_size):
      # Gather batch and calculate PPO loss on it
      batch = replay_buffer.sample(batch_size)
      loss_vals = loss_module(batch.to(device))
      loss_value = (
        loss_vals["loss_objective"]
        + loss_vals["loss_critic"]
      )

      # Optimize via gradient descent with the optimizer
      loss_value.backward()

      torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
      optimizer.step()
      optimizer.zero_grad()


if __name__ == "__main__":
  # Parameters (see https://pytorch.org/rl/tutorials/coding_ppo.html#ppo-parameters)
  batch_size = 100
  num_epochs = 100
  device = "cpu" if not torch.has_cuda else "cuda:0"

  clip_epsilon = 0.2 # for PPO loss
  gamma = 1.0
  lmbda = 0.95
  entropy_eps = 1e-4
  lr = 1e-4
  num_trajectories = 200
  num_policy_iterations = 100

  # Build the modules
  #autopilot_learner = StochasticAutopilotLearner()
  autopilot_learner = SlewRateAutopilotLearner()
  # autopilot_learner.init_from_params(np.random.normal(0, 1, 350))
  value_module = make_value_estimator_module(autopilot_learner.inputs)
  advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
  )
  loss_module = ClipPPOLoss(
    actor=autopilot_learner.policy_module,
    critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=False,
    entropy_coef=0,
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=gamma,
    loss_critic_type="smooth_l1",
  )
  optimizer = torch.optim.Adam(loss_module.parameters(), lr)

  autopilot_learner.save(os.path.join('data', 'ppo', 'policies'), 'learner#0')
  for i in range(num_policy_iterations):
    train_ppo_once(i, autopilot_learner, loss_module, advantage_module, optimizer, num_trajectories, num_epochs, batch_size)
    autopilot_learner.save(os.path.join('data', 'ppo', 'policies'), 'learner#' + str(i+1))