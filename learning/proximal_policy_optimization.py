import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict
from torchrl.modules import ValueOperator
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from autopilot import StochasticAutopilotLearner

"""
Gathers rollout data and returns it in the way the PPO loss_module expects
"""
def gather_rollout_data(autopilot_learner, num_trajectories):
  n_obs = autopilot_learner.inputs
  n_act = autopilot_learner.outputs

  # TODO: Here we need to gather [num_trajectories] rollouts.
  # Need to sample actions using the autopilot_learner.policy_module.
  # From the total number of steps of all rollouts, we will get a data_size.
  data_size = 10000 # update
  
  # Each entry tensor should be data_size x d where d is the dimension of
  # that entry for one step in a rollout. TODO: actually fill in correct values
  data = TensorDict({
    "observation": torch.rand(data_size, n_obs),
    "action": torch.rand(data_size, n_act),
    "sample_log_prob": -torch.rand(data_size,), # log probability that each action was selected
    ("next", "done"): torch.zeros(data_size, 1, dtype=torch.bool),
    ("next", "terminated"): torch.zeros(data_size, 1, dtype=torch.bool),
    ("next", "reward"): torch.rand(data_size, 1),
    ("next", "observation"): torch.rand(data_size, n_obs),
  }, [data_size,])

  return data, data_size

"""
Outputs an RL module for value estimation that holds a neural network trained to
estimate the value of a state/observation of size n_obs. This value estimator 
is used as a critic to subtract a baseline from the cumulative reward of the 
action taken during PPO learning.
"""
def make_value_estimator_module(n_obs):
  value_net = nn.Sequential(
    nn.Linear(n_obs, n_obs),
    nn.Tanh(),
    nn.Linear(n_obs, n_obs),
    nn.Tanh(),
    nn.Linear(n_obs, 1), # one value is computed for the state
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
def train_ppo_once(autopilot_learner, loss_module, advantage_module, optimizer, num_trajectories, num_epochs, batch_size):
  # Rollout policy to gather trajectories
  dataset, data_size = gather_rollout_data(autopilot_learner, num_trajectories)

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
      loss_vals = loss_module(batch.to("cpu"))
      loss_value = (
        loss_vals["loss_objective"]
        + loss_vals["loss_critic"]
        + loss_vals["loss_entropy"]
      )

      # Optimize via gradient descent with the optimizer
      loss_value.backward()
      optimizer.step()
      optimizer.zero_grad()
    
    print(autopilot_learner.policy_network[0].weight)


if __name__ == "__main__":
  # Parameters (see https://pytorch.org/rl/tutorials/coding_ppo.html#ppo-parameters)
  batch_size = 100
  num_epochs = 10
  clip_epsilon = 0.2 # for PPO loss
  gamma = 0.99
  lmbda = 0.95
  entropy_eps = 1e-4
  lr = 3e-4
  num_trajectories = 10

  # Build the modules
  autopilot_learner = StochasticAutopilotLearner()
  value_module = make_value_estimator_module(autopilot_learner.inputs)
  advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
  )
  loss_module = ClipPPOLoss(
    actor=autopilot_learner.policy_module,
    critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
  )
  optimizer = torch.optim.Adam(loss_module.parameters(), lr)
  train_ppo_once(autopilot_learner, loss_module, advantage_module, optimizer, num_trajectories, num_epochs, batch_size)