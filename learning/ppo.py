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
import time
import statistics

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
    # Reset distribution
    print(f"Trajectory #{t+1}")
    rand = np.random.rand()
    if rand <= 0.25:
      in_flight_reset = 0
    elif rand <= 0.5:
      in_flight_reset = 3
    elif rand <= 0.75:
      in_flight_reset = 4
    else:
      in_flight_reset = 5
    
    # Run a sim with a stochastic version of the autopilot so it can explore
    integrated_sim = FullIntegratedSim(x8, autopilot_learner, sim_time, in_flight_reset=in_flight_reset, auto_deterministic=False)
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
    rewards = torch.cat((rewards, reward)) 
    dones = torch.cat((dones, done))
    data_size += observation.shape[0]

  # we divide by std for reward scaling
  print("Reward scale", torch.std(rewards))
  rewards /= torch.std(rewards)
  print(f"Reward min: {torch.min(rewards)}, mean: {torch.mean(rewards)}, median: {torch.median(rewards)},  max: {torch.max(rewards)}" )
  # Each entry tensor should be data_size x d where d is the dimension of
  # that entry for one step in a rollout.
  data = TensorDict({
    "observation": observations,
    "action": TensorDict({
      "throttle": actions[:,0].detach(),
      "aileron": actions[:,1].detach(),
      "elevator": actions[:,2].detach(),
      "rudder": actions[:,3].detach(),
      # "throttle_log_prob": torch.zeros(data_size).detach(),
      # "aileron_log_prob": torch.zeros(data_size).detach(),
      # "elevator_log_prob": torch.zeros(data_size).detach(),
      # "rudder_log_prob": torch.zeros(data_size).detach(),
      }, [data_size,]),
    "sample_log_prob": sample_log_probs.detach(), # log probability that each action was selected
    # "sample_log_prob": TensorDict({
    # }, [data_size,]), # log probability that each action was selected
    ("next", "done"): dones,
    ("next", "terminated"): dones,
    ("next", "reward"): rewards,
    ("next", "observation"): next_observations,
  }, [data_size,])
  torch.save(data, os.path.join('data', 'ppo', 'all_data.pkl'))

  # Write to stats file
  stats_file = open(os.path.join('data', 'ppo', 'stats.txt'), 'a')
  stats_file.write('Policy Number #' + str(policy_num) + ':\n')
  stats_file.write('\tAverage Time Steps: ' + str(total_timesteps/num_trajectories) + '\n')
  stats_file.write('\tAverage Reward: ' + str(total_cum_reward/num_trajectories) + '\n')
  stats_file.write('\tBest Reward: ' + str(best_cum_reward) + '\n')
  stats_file.write('\tWorst Reward: ' + str(worst_cum_reward) + '\n')
  stats_file.close()

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
  dataset = torch.load(os.path.join("data","ppo","all_data.pkl")) 
  print(dataset)
  # data_size = 56114
  # print(max(dataset.get(("next", "reward"))))
  # print(dataset)
  # data_size = 9484
  # Define replay buffer to store trajectory data in during training
  replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(data_size),
    sampler=SamplerWithoutReplacement(),
  )
  
  for epoch in range(num_epochs):
    # print("Training Epoch", epoch+1)
    # Re-compute advantage at each epoch as its value depends on the value
    # network which is updated in the inner loop
    with torch.no_grad():
      advantage_module(dataset)

    # Gather the data into a replay buffer for sampling
    data_view = dataset.reshape(-1)
    replay_buffer.extend(data_view.cpu())
    batch_loss = 0
    batch_obj_loss = 0
    batch_critic_loss = 0
    for b in range(0, data_size // batch_size):
      # Gather batch and calculate PPO loss on it
      batch = replay_buffer.sample(batch_size)
      loss_vals = loss_module(batch.to(device))
      loss_value = (
        loss_vals["loss_objective"]
        + loss_vals["loss_critic"]
      )
      batch_loss += loss_value * batch_size
      batch_obj_loss += loss_vals["loss_objective"] * batch_size
      batch_critic_loss += loss_vals["loss_critic"] * batch_size

      # Optimize via gradient descent with the optimizer
      loss_value.backward()

      torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
      optimizer.step()
      optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
      print('loss_value in epoch #', epoch+1, ':', float(batch_loss/ 1000), '(objective:', float(batch_obj_loss/ 1000), ' critic: ', float(batch_critic_loss/ 1000), ')')


if __name__ == "__main__":

  # Parameters (see https://pytorch.org/rl/tutorials/coding_ppo.html#ppo-parameters)
  batch_size = 256
  num_epochs = 100
  device = "cpu" if not torch.has_cuda else "cuda:0"

  clip_epsilon = 0.2 #0.08 # for PPO loss
  gamma = 0.99 # keep 1.0
  lmbda = 0.95
  # entropy_eps = 1e-4
  lr = 5e-5

  # lr = 5e-5
# loss_value in epoch # 1 : 67.5849609375 (objective: -0.002501344308257103  critic:  67.58744049072266 )
# loss_value in epoch # 2 : 66.79192352294922 (objective: -0.022511614486575127  critic:  66.81440734863281 )
# loss_value in epoch # 3 : 65.99295806884766 (objective: -0.023394417017698288  critic:  66.01636505126953 )
# loss_value in epoch # 10 : 63.2775764465332 (objective: -0.03471797704696655  critic:  63.31227493286133 )
# loss_value in epoch # 20 : 58.33335876464844 (objective: -0.03877483680844307  critic:  58.37212371826172 )
# loss_value in epoch # 30 : 53.69935607910156 (objective: -0.04969186335802078  critic:  53.749061584472656 )
# loss_value in epoch # 40 : 48.16702651977539 (objective: -0.048812463879585266  critic:  48.21586227416992 )
# loss_value in epoch # 50 : 42.11222839355469 (objective: -0.06517862528562546  critic:  42.17741394042969 )
# loss_value in epoch # 60 : 36.78435134887695 (objective: -0.06043088063597679  critic:  36.844749450683594 )
# loss_value in epoch # 70 : 33.0183219909668 (objective: -0.062020089477300644  critic:  33.08034133911133 )
# loss_value in epoch # 80 : 32.66770935058594 (objective: -0.05988125503063202  critic:  32.72758865356445 )
# loss_value in epoch # 90 : 32.728004455566406 (objective: -0.04434733837842941  critic:  32.772335052490234 )
# loss_value in epoch # 100 : 32.90699768066406 (objective: -0.03433294966816902  critic:  32.94131851196289 )



  # 3e-5: roughly 160 (27.3, 133)
  # 1e-4: roughly 190 (27, 133)
  # 1e-4: roughly 190 (27, 133)

  num_trajectories = 500
  num_policy_iterations = 100

  # Build the modules
  autopilot_learner = StochasticAutopilotLearner()
  #autopilot_learner = SlewRateAutopilotLearner()
  #autopilot_learner.init_from_saved(os.path.join("data", "cross_entropy", "generation14", "learner#67.pth"))
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
    start = time.time()
    train_ppo_once(i, autopilot_learner, loss_module, advantage_module, optimizer, num_trajectories, num_epochs, batch_size)
    autopilot_learner.save(os.path.join('data', 'ppo', 'policies'), 'learner#' + str(i+1))

    # Evaluate the learner deterministically
    integrated_sim = FullIntegratedSim(x8, autopilot_learner, 60.0, in_flight_reset=0, auto_deterministic=True)
    integrated_sim.simulation_loop()
    cum_reward = integrated_sim.mdp_data_collector.get_cum_reward()
    stats_file = open(os.path.join('data', 'ppo', 'stats.txt'), 'a')
    stats_file.write('\tDeterministic Reward Evaluation: ' + str(cum_reward))
    end = time.time()
    duration = end - start
    stats_file.write(f'\n\tTime: {str(int(duration//60))}m {str(int(duration) %60)}')
    stats_file.write('\n\n')
    stats_file.close()
