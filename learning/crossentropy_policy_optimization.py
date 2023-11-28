import torch
import numpy as np
from learning.autopilot import AutopilotLearner
from simulation.simulate import FullIntegratedSim
from simulation.jsbsim_aircraft import x8
import os
from random import randint
import sys

"""
A generation of learners. It takes the form of a list of Learners with infra for
calculating statistics and generating a new generation from a gaussian distribution.
"""
class Generation:
  def __init__(self, learners, num_params):
    # Set of learners in the generation, each having a policy network
    self.learners = learners

    # The fixed number of parameters each policy network has
    self.num_params = num_params

  # Initializes a generation (likely the first generation for most use cases)
  # by using the torch default initialization for each of the policy networks
  def init_using_torch_default(generation_size, num_params):
    learners = []
    for i in range(generation_size):
      learners.append(AutopilotLearner())
    return Generation(learners, num_params)

  # Utilizes [rewards], which contains the reward obtained by each learner as a
  # np array, and preserves only the best [num_survive] learners.
  # Returns the best, median, and worst ids and rewards of the original gen.
  def preserve(self, rewards, num_survive):
    new_learners = []
    best_iis = np.flip(np.argsort(rewards))
    for i in best_iis[:num_survive]:
      new_learners.append(self.learners[i])
    
    # Update learners to only include those preserved
    self.learners = new_learners

    # Accumulate performance stats for return
    best_learner_id = best_iis[0]
    best_reward = rewards[best_learner_id]
    median_learner_id = len(best_iis)//2
    median_reward = rewards[median_learner_id]
    worst_learner_id = len(best_iis)-1
    worst_reward = rewards[worst_learner_id]
    return (best_learner_id+1, median_learner_id+1, worst_learner_id+1), (best_reward, median_reward, worst_reward)
  
  # Saves all learners' networks from the generation into a directory
  # parent_dir = parent of all generations
  def save_learners(self, parent_dir, generation):
    dir = os.path.join('data', parent_dir, 'generation' + str(generation))
    os.mkdir(dir)
    for i in range(len(self.learners)):
      self.learners[i].save(dir, 'learner#' + str(i+1))

  # Calculate stats for the generation's learner parameters
  # NOTE: This should be used after preserve() if forming a new generation
  def calculate_stats(self):
    # Initialize matrix of paramaters of the entire learner set
    # n x d, where n is the learner set size and d is the param size of the neural networks
    set_params = torch.empty((0, self.num_params))

    for learner in self.learners:
      # Build flattened list of parameters for this learner
      parameters = learner.policy_network.parameters()
      flattened_params = torch.empty((0,1))
      for param in parameters:
        param = torch.flatten(param).unsqueeze(1)
        flattened_params = torch.cat((flattened_params, param))
      
      # Add flattened list of parameters for this learner to the set_params
      set_params = torch.cat((set_params, flattened_params.T), 0)
    
    # Return mean and covariance of the params
    # means = dx1 tensor, where d is the number of network parameters, to sample from
    # covs = dxd tensor that encodes the covariance to sample from
    set_params = set_params.detach().numpy()
    return np.mean(set_params,0), np.cov(set_params.T)
  
  # Make a new generation of size generation_size using mean/cov for sampling 
  def make_new_generation(mean, cov, generation_size, num_params):
    # gen_size x d
    selected_params = np.random.multivariate_normal(mean, cov, generation_size)
    
    # Generate each learner from params
    learners = []
    for param_list in selected_params:
      l = AutopilotLearner()
      l.init_from_params(param_list)
      learners.append(l)
    return Generation(learners, num_params)

def cross_entropy_train(epochs, generation_size, num_survive, num_params=238, sim_time=60.0, save_dir='cross_entropy'):
  # Create save_dir (and if one already exists, rename it with some rand int)
  if os.path.exists(os.path.join('data', save_dir)):
    os.rename(os.path.join('data', save_dir), os.path.join('data', save_dir + '_old' + str(randint(0, 100000))))
  os.mkdir(os.path.join('data', save_dir))
  stats_file = open(os.path.join('data', save_dir, 'stats.txt'))
  
  # Baseline to be updated after first generation
  mean = np.zeros((num_params))
  cov = 0.1 * np.identity(num_params)

  for epoch in range(epochs):
    print('Generation #', (epoch+1))

    # Sample the new generation
    generation = Generation.make_new_generation(mean, cov, generation_size, num_params)

    # Save generation
    generation.save_learners(save_dir, epoch+1)

    # Evaluate generation through rollouts
    rewards = []
    for i in range(len(generation.learners)):
      id = str(100*(epoch+1) + (i+1))
      learner = generation.learners[i]
      print('Evaluating Learner #', id)
      with HidePrints():
        integrated_sim = FullIntegratedSim(x8, learner, sim_time)
      integrated_sim.simulation_loop()
      rewards.append(integrated_sim.mdp_data_collector.get_cum_reward())
      print('Reward for Learner #', id, ': ', integrated_sim.mdp_data_collector.get_cum_reward())

    # Let the best "survive"
    print('Preserving the best learners from generation #', (epoch+1))
    ids, rew = generation.preserve(np.array(rewards), num_survive)

    # Find the new distribution with the actual best
    mean, cov = generation.calculate_stats()
    cov += 0.01 * np.identity(mean.shape[0])

    # Save important info in the save_dir stats file
    stats_file.write('Generation #' + str(epoch+1) + ':\n')
    stats_file.write('Best, Median, and Worst Learner: ' + str(ids) + '\n')
    stats_file.write('Best, Median, and Worst Reward: ' + str(rew) + '\n')
    stats_file.write('Mean Weights:' + str(mean) + '\n')
    stats_file.write('Cov Weights:' + str(cov) + '\n')
    stats_file.write('\n\n\n')
    
class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
  os.environ["JSBSIM_DEBUG"]=str(0)
  # epochs, generation_size, num_survive
  cross_entropy_train(3, 20, 4)