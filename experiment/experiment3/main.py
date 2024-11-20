from experiment.experiment3.maps import GeneralMap
from algo.mha_multi_drqn.run import train


if __name__ == '__main__':
    train_kwargs = {"map": GeneralMap,
                    "fair_service": False,
                    "n_layers": 2,
                    "n_heads": 2,
                    "batch_size": 128,
                    "steps_per_epoch": 30000,
                    "update_after": 20000,
                    "epochs": 500,
                    "n_ubs": 2,
                    "range_pos": 400,
                    "n_gts": 20,
                    "n_eve": 16,
                    "polyak": 0.999,
                    "double_q": True
                    }
    train(train_kwargs=train_kwargs)

    # print(train_kwargs)
    # print(type(train_kwargs))