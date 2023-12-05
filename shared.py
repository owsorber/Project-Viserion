import sys, os

THROTTLE_CLAMP = 0.8
AILERON_CLAMP = 0.1
ELEVATOR_CLAMP = 0.1
RUDDER_CLAMP = 0.1

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