import torch as T
from train_util import get_cuda
import numpy as np
import data_util.config as config

def _get_len_attn_table(n_position, d_hid, padding_idx=None):
    """
    LenAtten encoding table
    """

    def cal_angle(n_position, hid_idx):
        return (n_position) / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    sinusoid_table = np.array([cal_angle(n_position, hid_j) for hid_j in range(d_hid)])

    if n_position == 0:
        sinusoid_table[0::2] = 0 # dim 2i
        sinusoid_table[1::2] = 0  # dim 2i+1
    else:
        sinusoid_table[0::2] = np.sin(sinusoid_table[0::2])
        sinusoid_table[1::2] = np.cos(sinusoid_table[1::2])
        
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.
    return T.FloatTensor(sinusoid_table)

def create_len_attn_table():
    print('Creating Len attention table.')
    x = np.zeros((config.max_word_len, config.hidden_dim), dtype=np.float)
    x = T.FloatTensor(x)

    for i in range(0, config.max_word_len):
        position =_get_len_attn_table(i, config.hidden_dim)
        x[i]= T.add(x[i], position)
    return x
