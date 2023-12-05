import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torch.distributions import Categorical
from tensordict.nn import CompositeDistribution
from learning.utils import CategoricalControlsExtractor
import os

"""
An autopilot learner. It takes the form of a policy network that outputs
actions for a given state.
"""
class AutopilotLearner:
  def __init__(self):
    """
    State: 
    - altitude: z
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
      nn.Tanh(),
    )

  # Returns the action selected and 0, representing the log-prob of the 
  # action, which is zero in the default deterministic setting
  def get_action(self, observation):
    return self.policy_network(observation), 0
  
  def get_control(self, action):
    """
    Transforms network-outputted action tensor to the correct cmds.
    Clamps various control outputs and sets the mean for control surfaces to 0.
    Assumes [action] is a 4-item tensor of throttle, aileron cmd, elevator cmd, rudder cmd.
    """
    action[0] = 0.8 * (0.5*(action[0] + 1))
    action[1] = 0.1 * action[1]
    action[2] = 0.4 * action[2]
    action[3] = 0.1 * action[3]
    return action

  # flattened_params = flattened dx1 numpy array of all params to init from
  # NOTE: the way the params are broken up into the weights/biases of each layer
  #        would need to be manually edited for changes in network architecture
  def init_from_params(self, flattened_params):
    flattened_params = torch.from_numpy(flattened_params).to(torch.float32)
    
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
      nn.Tanh()
    )
  
  # Loads the network from path
  def init_from_saved(self, path):
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
      default_interaction_type=InteractionType.RANDOM,
      return_log_prob=True,
    )
  
  # Returns the action selected and the log_prob of that action
  def get_action(self, observation):
    data = TensorDict({"observation": observation}, [])
    policy_forward = self.policy_module(data)
    return policy_forward["action"], policy_forward["sample_log_prob"]

  # NOTE: This initializes from *deterministic* learner parameters and picks
  # random parameters for the stochastic portion
  def init_from_params(self, flattened_params):
    super().init_from_params(flattened_params)
    self.transform_from_deterministic_learner()

  def init_from_saved(self, path):
    super().init_from_saved(path)
    
    # Update policy module after policy network is updated
    policy_module = TensorDictModule(
      self.policy_network, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    self.policy_module = ProbabilisticActor(
      module=policy_module,
      in_keys=["loc", "scale"],
      distribution_class=TanhNormal,
      default_interaction_type=InteractionType.RANDOM,
      return_log_prob=True,
    )

"""
Slew rate autopilot learner.
Each control (throttle/aileron/elevator/rudder) has three options:
stay constant, go down, or go up.
"""
class SlewRateAutopilotLearner:
  def __init__(self):
    self.inputs = 15
    self.outputs = 4

    # Slew rates are wrt sim clock
    self.throttle_slew_rate = 0.005
    self.aileron_slew_rate = 0.0001
    self.elevator_slew_rate = 0.00025
    self.rudder_slew_rate = 0.0001
    
    self.policy_network = nn.Sequential(
      nn.Linear(self.inputs, self.inputs),
      nn.ReLU(),
      nn.Linear(self.inputs, 3 * self.outputs),
      nn.Sigmoid(),
      CategoricalControlsExtractor()
    )

    self.instantiate_policy_module()
  
  def instantiate_policy_module(self):
    policy_module = TensorDictModule(self.policy_network, in_keys=["observation"], out_keys=[("params", "throttle", "probs"),("params", "aileron", "probs"),("params", "elevator", "probs"),("params", "rudder", "probs")])
    self.policy_module = policy_module = ProbabilisticActor(
      module=policy_module,
      in_keys=["params"],
      distribution_class=CompositeDistribution, 
      distribution_kwargs={
        "distribution_map": {
          "throttle": Categorical,
          "aileron": Categorical,
          "elevator": Categorical,
          "rudder": Categorical,
        }
      },
      default_interaction_type=InteractionType.RANDOM, 
      return_log_prob=True
    )

  # Returns the action selected and the log_prob of that action
  def get_action(self, observation):
    data = TensorDict({"observation": observation}, [])
    policy_forward = self.policy_module(data)
    action = torch.Tensor([policy_forward['throttle'], policy_forward['aileron'], policy_forward['elevator'], policy_forward['rudder']])
    return action, policy_forward["sample_log_prob"]

  # Always samples the mode of the output for each control
  def get_deterministic_action(self, observation):
    throttle_probs, aileron_probs, elevator_probs, rudder_probs = self.policy_network(observation)
    action = torch.Tensor([torch.argmax(throttle_probs), torch.argmax(aileron_probs), torch.argmax(elevator_probs), torch.argmax(rudder_probs)])
    return action, 0
 
  # Apply a -1 transformation to the action to create control tensor such that:
  # -1 means go down, 0 means stay same, and +1 means go up
  def get_control(self, action):
    return action - 1

  # flattened_params = flattened dx1 numpy array of all params to init from
  # NOTE: the way the params are broken up into the weights/biases of each layer
  #        would need to be manually edited for changes in network architecture
  def init_from_params(self, flattened_params):
    flattened_params = torch.from_numpy(flattened_params).to(torch.float32)
    
    pl, pr = 0, 0
    layer1 = nn.Linear(self.inputs, self.inputs)
    pr += layer1.weight.nelement()
    layer1.weight = nn.Parameter(flattened_params[pl:pr].reshape(layer1.weight.shape))
    pl = pr
    pr += layer1.bias.nelement()
    layer1.bias = nn.Parameter(flattened_params[pl:pr].reshape(layer1.bias.shape))

    layer2 = nn.Linear(self.inputs, 3*self.outputs)
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
      CategoricalControlsExtractor()
    )
    self.instantiate_policy_module()
  
  # Loads the network from path
  def init_from_saved(self, path):
    self.policy_network = torch.load(path)
    self.instantiate_policy_module()
  
  # Saves the network to dir/name.pth
  def save(self, dir, name):
    path = os.path.join(dir, name + '.pth')
    torch.save(self.policy_network, path)

  

  

