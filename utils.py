import random
import numpy as np
import torch


def set_rand_seed(seed=3407):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)