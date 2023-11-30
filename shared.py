import sys, os

"""
Context manager object that helps prevent print outputs from 3rd party software.
"""
class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


"""
Transforms network-outputted action tensor to the correct cmds.
Clamps various control outputs and sets the mean for control surfaces to 0.
Assumes [action] is a 4-item tensor of throttle, aileron cmd, elevator cmd, rudder cmd.
"""
def action_transform(action):
  action[0] = 0.9 + 0.3 * action[0] * 0
  action[1] = 0.01 * (action[1] - 0.5)
  action[2] = 0.1 + 0.000 * (action[2] - 0.5)
  action[3] = 0.1 * (action[3] - 0.5) 
  return action