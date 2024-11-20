import random
from collections import deque

import torch

import numpy as np

def check_args_sanity(args):
    """Checks sanity and avoids conflicts of arguments."""

    # Ensure specified cuda is used when it is available.
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = f'cuda:{args.cuda_index}'
    else:
        args.device = 'cpu'
    print(f"Choose to use {args.device}.")

    # When QMix is used, ensure a scalar reward is used.
    if hasattr(args, 'mixer'):
        if args.mixer and not args.share_reward:
            args.share_reward = True
            print("Since QMix is used, all agents are forced to share a scalar reward.")

    return args


def cat(data_list):
    """Concatenates list of inputs"""
    if isinstance(data_list[0], torch.Tensor):
        return torch.cat(data_list)
    # elif isinstance(data_list[0], dgl.DGLGraph):
    #     return dgl.batch(data_list)
    else:
        raise TypeError("Unrecognised observation type.")


def set_rand_seed(seed=3407):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import pickle
def save_var(var_path, var):
    var_n = var_path + '.pickle'
    with open(var_n, 'wb') as f:
        pickle.dump(var, f)
def load_var(var_path):
    # 从文件中读取变量
    var_n = var_path + '.pickle'
    with open(var_n, 'rb') as f:
        my_var = pickle.load(f)

    return my_var

import json
def convert_json(obj):
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except:
            return False

    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

import os
def save_config(output_dir, config):
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
    print(output)
    with open(os.path.join(output_dir, "config.json"), 'w') as out:
        out.write(output)

import glob
import os
def del_file(file_suffix):
    file_list = glob.glob('*.' + file_suffix)
    for file_path in file_list:
        os.remove(file_path)

if __name__ == "__main__":
    # print("期望值 E[|x|^2] =", calEstimateChannelError(std=0.01))
    del_file('pickle')
    # buffer = ReplayBuffer(capacity=500, max_seq_len=5)
    # for i in range(30):
    #     obs = np.random.rand(1, 128)
    #     h = np.random.rand(1, 256)
    #     state = np.random.rand(1, 256)
    #     act = np.random.rand(1, 5)
    #     rew = np.random.rand(1, 1)
    #     next_obs = np.random.rand(1, 128)
    #     next_h = np.random.rand(1, 256)
    #     next_state = np.random.rand(1, 256)
    #     done = np.random.rand(1, 1)
