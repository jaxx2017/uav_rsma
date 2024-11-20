import numpy as np

from scipy import integrate

import torch

from mha_drqn_env import maps

def regis_map(range_pos):
    if range_pos == 400:
        return maps.Rang400MapSpecial()

    return None

def wrapper_obs(obs):
    observation_other_features = []
    observation_gt_features = []
    for o in obs:
        observation_agent = list(o['agent'])
        observation_gt_features.append(o['gt'])
        # for gt in o['gt']:
        #     observation_gt_features.append(list(gt))
        #     observation_agent += list(gt)
        for ubs in o['ubs']:
            observation_agent += list(ubs)
        observation_other_features.append(observation_agent)
    observation_gt_features = np.array(observation_gt_features, dtype=np.float32)

    return torch.tensor(observation_other_features), torch.from_numpy(observation_gt_features)

def compute_jain_fairness_index(x):
    """Computes the Jain's fairness index of entries in given ndarray."""
    if x.size > 0:
        x = np.clip(x, 1e-6, np.inf)
        return np.square(x.sum()) / (x.size * np.square(x).sum())
    else:
        return 1

def wrapper_state(state):
    s_tensor = torch.tensor(state)

    return s_tensor.unsqueeze(0)

if __name__ == '__main__':
    pass