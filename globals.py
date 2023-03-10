
""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

globals.py - global defines that don't change.

"""

import torch

DTYPE = torch.float32
EPSILON = 0.001

def badness(t: torch.Tensor) -> bool:
    return not(torch.all(torch.isnan(t) == False) and torch.all(torch.isinf(t) == False))