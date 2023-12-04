import torch
import torch.nn as nn

class CategoricalControlsExtractor(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    tensor, *others = tensors
    throttle, aileron, elevator, rudder = tensor.chunk(4, -1)
    return (nn.functional.softmax(throttle), nn.functional.softmax(aileron), nn.functional.softmax(elevator), nn.functional.softmax(rudder), *others)
