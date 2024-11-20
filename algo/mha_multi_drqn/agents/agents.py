import torch
import torch.nn as nn
import math


class Agents(nn.Module):
    def __init__(self, gt_features_dim, num_heads, other_features_dim, move_dim, power_dim, n_layers=2, hidden_size=256):
        super(Agents, self).__init__()
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self.mha_layer = nn.MultiheadAttention(embed_dim=gt_features_dim, num_heads=num_heads, batch_first=True)

        layers = [nn.Linear(gt_features_dim + other_features_dim, self._hidden_size), nn.ReLU()]
        for l in range(self._n_layers - 1):
            layers += [nn.Linear(self._hidden_size, self._hidden_size), nn.ReLU()]
        self.enc = nn.Sequential(*layers)
        self.rnn = nn.GRUCell(self._hidden_size, self._hidden_size)

        self.move_head = nn.Linear(self._hidden_size, move_dim)
        self.power_head = nn.Linear(self._hidden_size, power_dim)

    def init_hidden(self):
        return torch.zeros(1, self._hidden_size)

    def forward(self, gt_features, other_features, h):
        # 使用 MHA 处理gt_geatures
        mha_gt_features, _ = self.mha_layer(gt_features, gt_features, gt_features)
        # 平均 pool
        mha_gt_features = mha_gt_features.mean(dim=1)
        x = torch.cat([mha_gt_features, other_features], dim=1)
        x = self.enc(x)
        h = self.rnn(x, h)

        move = self.move_head(h)
        power = self.power_head(h)

        return move, power, h


if __name__ == '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    batch_size = 2
    num_gts = 12
    gt_features_dim = 4  # 每个GT的特征维度
    num_heads = 2
    other_dim = 6
    moves_dim = 16
    jamming_power_dim = 10
    hidden_dim = 256

    agent = Agents(gt_features_dim=gt_features_dim,
                   num_heads=num_heads,
                   other_features_dim=other_dim,
                   move_dim=moves_dim,
                   power_dim=jamming_power_dim)
    gt_features = torch.zeros(batch_size, num_gts, gt_features_dim)
    other_features = torch.rand(batch_size, other_dim)
    hidden_state = agent.init_hidden().expand(batch_size, -1)  # 扩展隐藏状态到批大小

    # 模型向前传播
    output = agent(gt_features, other_features, hidden_state)
    print(output)
    # print("Output Shape", output[0].shape)
    # print(agent)


