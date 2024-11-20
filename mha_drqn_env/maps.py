import numpy as np

import matplotlib.pyplot as plt


def plot(map):
    uav_pos = map['pos_ubs']
    eve_pos = map['pos_eve']
    gts_pos = map['pos_gts']
    range_pos = map['range_pos']
    area = map['area']

    fig, ax = plt.subplots()
    ax.axis([0, range_pos, 0, range_pos])
    for (x, y) in uav_pos:
        ubs, = ax.plot(x, y, marker='o', color='b')
    for (x, y) in eve_pos:
        eve, = ax.plot(x, y, marker='o', color='r', markersize=5)
    for (x, y) in gts_pos:
        gts, = ax.plot(x, y, marker='o', color='y', markersize=5)
    # plot axis0
    for x in range(0, range_pos, int(area)):
        if x == 0:
            continue
        ax.plot([0, range_pos], [x, x], linestyle='--', color='b')

    # plot axis1
    for y in range(0, range_pos, int(area)):
        if y == 0:
            continue
        ax.plot([y, y], [0, range_pos], linestyle='--', color='b')

    ax.legend(handles=[ubs, eve, gts],
              labels=['uav_pos', 'eve_pos', 'gts_pos'],
              loc="center left", bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.75)
    plt.show()

class Rang400Map:
    def __init__(self, range_pos=400, n_eve=4, n_gts=5, n_ubs=2, n_community=4):
        self.n_eve = n_eve
        self.n_gts = n_gts
        self.n_ubs = n_ubs
        self.pos_eve = np.empty((self.n_eve, 2), dtype=np.float32)
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
        self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)
        self.range_pos = range_pos
        self.area = self.range_pos / 2
        self.n_community = n_community
        self.gts_in_community = [[] for _ in range(self.n_community)]

    def set_eve(self):
        for i in range(2):
            range_eve_ly = self.area * i
            range_eve_ry = self.area * (i + 1)
            for j in range(2):
                range_eve_lx = self.area * j
                range_eve_rx = self.area * (j + 1)
                x = np.random.uniform(range_eve_lx, range_eve_rx)
                y = np.random.uniform(range_eve_ly, range_eve_ry)
                self.pos_eve[i * 2 + j] = x, y

    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]
        for i in range(self.n_gts - 1):
            x = np.random.uniform(250, 300)
            y = np.random.uniform(250, 300)
            self.pos_gts[i] = x, y
            num_community = (y // self.area) * 2 + (x // self.area)
            self.gts_in_community[int(num_community)].append(i)
        self.pos_gts[0] = [280, 320]
        self.pos_gts[1] = [280, 340]
        self.pos_gts[2] = [300, 320]
        self.pos_gts[3] = [300, 340]

        self.pos_gts[4] = [100, 100]
        # self.pos_gts[self.n_gts - 1] = np.random.uniform(100, 150), np.random.uniform(100, 150)
        self.gts_in_community[0].append(self.n_gts - 1)

    def set_ubs(self):
        self.pos_ubs[0] = 205, 195
        self.pos_ubs[1] = 195, 205

    def get_map(self):
        self.set_eve()
        self.set_gts()
        self.set_ubs()

        return dict(pos_gts=self.pos_gts,
                    pos_eve=self.pos_eve,
                    pos_ubs=self.pos_ubs,
                    area=self.area,
                    range_pos=self.range_pos,
                    gts_in_community=self.gts_in_community)

class Rang400MapSpecial:
    def __init__(self, range_pos=400, n_eve=16, n_gts=12, n_ubs=2, n_community=16):
        self.n_eve = n_eve
        self.n_gts = n_gts
        self.n_ubs = n_ubs
        self.pos_eve = np.empty((self.n_eve, 2), dtype=np.float32)
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
        self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)
        self.range_pos = range_pos
        self.fen = 4
        self.area = self.range_pos / self.fen
        self.n_community = n_community
        self.gts_in_community = [[] for _ in range(self.n_community)]

    def set_eve(self):
        # for i in range(self.fen):
        #     range_eve_ly = self.area * i
        #     range_eve_ry = self.area * (i + 1)
        #     for j in range(self.fen):
        #         range_eve_lx = self.area * j
        #         range_eve_rx = self.area * (j + 1)
        #         x = np.random.uniform(range_eve_lx, range_eve_rx)
        #         y = np.random.uniform(range_eve_ly, range_eve_ry)
        #         self.pos_eve[i * self.fen + j] = x, y
        self.pos_eve[0] = [-200, -200]
        self.pos_eve[1] = [-200, -200]
        self.pos_eve[2] = [-200, -200]
        self.pos_eve[3] = [360, 80]
        self.pos_eve[4] = [-200, -200]
        self.pos_eve[5] = [-200, -200]
        self.pos_eve[6] = [260, 130]
        self.pos_eve[7] = [310, 110]
        self.pos_eve[8] = [-200, -200]
        self.pos_eve[9] = [110, 210]
        self.pos_eve[10] = [-200, -200]
        self.pos_eve[11] = [-200, -200]
        self.pos_eve[12] = [60, 310]
        self.pos_eve[13] = [170, 320]
        self.pos_eve[14] = [-200, -200]
        self.pos_eve[15] = [-200, -200]


    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]
        self.pos_gts[0] = [240, 140]
        self.pos_gts[1] = [260, 160]
        self.pos_gts[2] = [340, 140]
        self.pos_gts[3] = [360, 160]
        self.pos_gts[4] = [340, 60]
        self.pos_gts[5] = [340, 40]
        self.pos_gts[6] = [150, 240]
        self.pos_gts[7] = [140, 260]
        self.pos_gts[8] = [140, 360]
        self.pos_gts[9] = [140, 340]
        self.pos_gts[10] = [40, 340]
        self.pos_gts[11] = [40, 360]
        for i in range(self.n_gts):
            x, y = self.pos_gts[i]
            num_community = (y // self.area) * self.fen + (x // self.area)
            self.gts_in_community[int(num_community)].append(i)

    def set_ubs(self):
        self.pos_ubs[0] = 250, 50
        self.pos_ubs[1] = 50, 250

    def get_map(self):
        self.set_eve()
        self.set_gts()
        self.set_ubs()

        return dict(pos_gts=self.pos_gts,
                    pos_eve=self.pos_eve,
                    pos_ubs=self.pos_ubs,
                    area=self.area,
                    range_pos=self.range_pos,
                    gts_in_community=self.gts_in_community)


if __name__ == '__main__':
    Map400Special = Rang400MapSpecial(range_pos=400, n_eve=16, n_gts=12, n_ubs=2, n_community=16)
    map400 = Map400Special.get_map()
    plot(map400)
    print(map400)
