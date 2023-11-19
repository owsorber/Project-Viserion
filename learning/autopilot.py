import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import os

"""
An autopilot learner. It takes the form of a policy network that outputs
actions for a given state.
"""
class AutopilotLearner:
  def __init__(self):
    """
    State: 
    - 3d position: x, y, z (alt)
    - 3d velocity: vx, vy, vz
    - 3 angles: roll, pitch, yaw
    - 3 angular velocities: w_roll, w_pitch, w_yaw
    - 3d relative position of next waypoint: wx, wy, wz
    """
    self.inputs = 15

    """
    Action:
    - Throttle
    - Aileron Position
    - Elevator Position
    - Rudder Position
    """
    self.outputs = 4

    # Default initialization
    self.policy_network = nn.Sequential(
      nn.Linear(self.inputs, self.inputs),
      nn.ReLU(),
      nn.Linear(self.inputs, self.outputs),
      nn.Sigmoid(),
    )

  # Returns the control selected
  def get_controls(self, sim_state):
    observation = sim_state # need to actually extract observation from sim eventually
    return self.policy_network(observation)

  # flattened_params = flattened dx1 numpy array of all params to init from
  # NOTE: the way the params are broken up into the weights/biases of each layer
  #        would need to be manually edited for changes in network architecture
  def init_from_params(self, flattened_params):
    flattened_params = torch.from_numpy(flattened_params)
    
    pl, pr = 0, 0
    layer1 = nn.Linear(self.inputs, self.inputs)
    pr += layer1.weight.nelement()
    layer1.weight = nn.Parameter(flattened_params[pl:pr].reshape(layer1.weight.shape))
    pl = pr
    pr += layer1.bias.nelement()
    layer1.bias = nn.Parameter(flattened_params[pl:pr].reshape(layer1.bias.shape))

    layer2 = nn.Linear(self.inputs, self.outputs)
    pl = pr
    pr += layer2.weight.nelement()
    layer2.weight = nn.Parameter(flattened_params[pl:pr].reshape(layer2.weight.shape))
    pl = pr
    pr += layer2.bias.nelement()
    layer2.bias = nn.Parameter(flattened_params[pl:pr].reshape(layer2.bias.shape))

    self.policy_network = nn.Sequential(
      layer1,
      nn.ReLU(),
      layer2,
      nn.Sigmoid(),
    )
  
  # Loads the network from dir/name.pth
  def init_from_saved(self, dir, name):
    path = os.path.join(dir, name + '.pth')
    self.policy_network = torch.load(path)

  # Saves the network to dir/name.pth
  def save(self, dir, name):
    path = os.path.join(dir, name + '.pth')
    torch.save(self.policy_network, path)

  # Gets the number of params in the policy network
  def get_num_params(self):
    count = 0
    for parameters in self.policy_network.parameters():
      count += parameters.nelement()
    return count


"""
A stochastic autopilot learner. It takes the form of a policy network that outputs
action mean/sigma for a given state, where controls are sampled from the 
corresponding Tanh-normal distribution, and is used for any stochastic-based 
on-policy learning algorithms.
"""
class StochasticAutopilotLearner(AutopilotLearner):
  def __init__(self):
    super().__init__()
    self.transform_from_deterministic_learner()

  # Helper function to transform stochastic architecture, assuming that
  # self.policy_network is the deterministic version
  def transform_from_deterministic_learner(self):
    # Save the weights/bias of the last layer
    w = self.policy_network[-2].weight
    b = self.policy_network[-2].bias

    # Double the output to account for means of outputs and sigmas of outputs
    self.policy_network[-2] = nn.Linear(self.inputs, self.outputs * 2)

    # Update the output layer to include the original weights and biases
    self.policy_network[-2].weight = nn.Parameter(torch.cat((w, torch.zeros(w.shape)), 0))
    self.policy_network[-2].bias = nn.Parameter(torch.cat((b, torch.zeros(b.shape)), 0))

    # Add a normal param extractor to the network to extract (means, sigmas) tuple
    self.policy_network.append(NormalParamExtractor())

    # Make a probabalistic actor wrapper so the stochastic learner can easily
    # output controls while returning the log probability of that action,
    # which is necessary for gradient-based optimization
    policy_module = TensorDictModule(
      self.policy_network, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    self.policy_module = ProbabilisticActor(
      module=policy_module,
      in_keys=["loc", "scale"],
      distribution_class=TanhNormal,
      distribution_kwargs={
          "min": 0, # minimum control
          "max": 1, # maximum control
      },
      return_log_prob=True,
    )
  
  # Returns the control selected and the log_prob of that control
  def get_controls(self, sim_state):
    observation = sim_state # need to actually extract observation from sim eventually

    data = TensorDict({"observation": observation}, [])
    policy_forward = self.policy_module(data)
    return policy_forward["action"], policy_forward["sample_log_prob"]

  # NOTE: This initializes from *deterministic* learner parameters and picks
  # random parameters for the stochastic portion
  def init_from_params(self, flattened_params):
    super().init_from_params(flattened_params)
    self.transform_from_deterministic_learner()
