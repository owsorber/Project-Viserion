import torch
import numpy as np
from autopilot import AutopilotLearner

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
  def init_using_torch_default(generation_size):
    learners = []
    for i in range(generation_size):
      learners.append(AutopilotLearner())
    return Generation(learners)

  # Utilizes [rewards], which contains the reward obtained by each learner, and 
  # and preserves only the best [num_survive] learners
  def preserve(self, rewards, num_survive):
    new_learners = []
    best_iis = np.flip(np.argsort(rewards))
    for i in best_iis[:num_survive]:
      new_learners.append(self.learners[i])
    
    self.learners = new_learners
  
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
  def make_new_generation(mean, cov, generation_size):
    # gen_size x d
    selected_params = np.random.multivariate_normal(mean, cov, generation_size)
    
    # Generate each learner from params
    learners = []
    for param_list in selected_params:
      l = AutopilotLearner()
      l.init_from_params(param_list)
      learners.append(l)
    return Generation(learners)

def cross_entropy_train(epochs, generation_size, num_params):
  # To be updated after the first generation
  mean = None
  cov = None

  for epoch in range(epochs):
    # Sample the new generation
    if epoch == 0:
      # Initialize generation as default
      generation = Generation.init_using_torch_default(generation_size)
    else:
      # Initialize generation from the previous best
      generation = Generation.make_new_generation(mean, cov, generation_size)

    # TODO: Evaluate generation through rollouts

    # TODO: Let the best "survive"
    generation.preserve(np.array([]))

    # TODO: Find the new distribution with the actual best
    mean, cov = generation.calculate_stats()
    


if __name__ == "__main__":
  # Basic proof of concept
  l1 = AutopilotLearner()
  l2 = AutopilotLearner()
  g = Generation([l1, l2], l1.get_num_params())
  mean, cov = g.calculate_stats()
  new_gen = Generation.make_new_generation(mean, cov, 1)