# A Multi-Agent Collaboration Framework for Secure RSMA-Enabled Multi-UAV Networks

This is an an open implementation of our paper "A Multi-Agent Collaboration Framework for Secure RSMA-Enabled Multi-UAV Networks".


## Required packages
```
conda env create -f environment.yaml
```
We developed and tested only on Linux-based systems. In principle, it should also run on Windows, but there might be some compatibility issues. 



## How to use

**Note:**
The `ltf` in the code is equal to the `FSF` in the paper, and the `nltf` is equal to the `CQF` in the paper. And pay attention to modify the absolute path in code.

### 1. Clone
```
git clone https://github.com/jaxx2017/uav_rsma.git
cd uav_rsma
```

### 2. Create a Folder to Store Data
```
mkdir mha_drqn_data
```


The relevant training data, such as 
* `checkpoint`: Weight parameters saved during training.
* `tensorboard-log`: Visualization of the training process: `tensorboard --logdir=pathname`
* `config.json`: Hyperparameter configuration files. 
* `other data`: Training return and test return.

will be stored in `/uav_rsma/mha_drqn_data/exp x`. `x` is the number of experiments.

### 3. Train
The hyperparameter parameters are in `/algo/mha_multi_drqn/config.py`. The training procedure is run as follows：
* exp1 (two-UAV) in Paper:
```
cd ./experiment/experiment3
python main.py
```
* exp2 (four-UAV) in Paper:
```
cd ./experiment/experiment2
python main.py
```
After the training is completed, the relevant data will be saved in `/uav_rsma/mha_drqn_data/exp x`.

### 4. Test
For exp1 (two-UAV) in Paper
```
cd ./experiment/experiment3
```
Modify the `config_path, test_ret_path, train_ret_path, model_path, test_kwargs` in file `test_policy.py`. Such as for the `config_path`
```
config_path = '/uav_rsma-master/mha_drqn_data/expx/config.json'
```
You need to modify `expx` to the folder where the training data is stored. Then modify parameter:
* saveif: Whether to save the drawing？ `True` or `False`. If `True`, data will be saved in `./experiment/experiment3`.
* fair_service: Is the service fair? `True` or `False`.
```
python test_policy.py
```

### 5. Results Plot
For exp1 (two-UAV) in Paper
```
cd ./experiment/experiment3
```
Parameters:
* fair_service: Is the service fair? `True` or `False`.
* save: Whether to save the drawing？ `True` or `False`.
Then 
```
python draw.py
```
To draw a comparison between CQF and FSF, you need to run
```
python draw_compared.py
```

## Reference 
If using this code for research purposes, please cite:
```
...
```
