import torch.nn as nn

import numpy as np

import torch

import pickle
import json

class PositionlEncoding(nn.Module):

    def __init__(self, d_hid, n_position=100):
        super(PositionlEncoding, self).__init__()

        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json(path):
    data = json.load(open(path, 'r'))
    return data

