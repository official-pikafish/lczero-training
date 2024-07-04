import numpy as np
import tensorflow as tf
from policy_index import policy_index

columns = 'abcdefghi'
rows = '0123456789'


def index_to_position(x):
    return columns[x % 9] + rows[x // 9]


def make_map():
    z = np.zeros((90 * 90, 2062), dtype=np.int32)
    apm_out = np.zeros((2062,), dtype=np.int32)
    apm_in = np.zeros((90 * 90), dtype=np.int32)
    # loop for moves (for i in 0:2062, stride by 1)
    for pickup_index in range(90):
        for putdown_index in range(90):
            move = index_to_position(pickup_index) + index_to_position(putdown_index)
            if policy_index.count(move) == 0:
                continue
            move = policy_index.index(move)
            du_idx = putdown_index + (90 * pickup_index)
            z[du_idx, move] = 1
            apm_out[move] = du_idx
            apm_in[du_idx] = move

    return z, apm_out, apm_in


apm_map, apm_out, apm_in = make_map()


def set_zero_sum(x):
    x = x + (1 - tf.reduce_sum(x, axis=1, keepdims=True)) * (
            1.0 / 90)
    return x


def get_up_down(moves):
    out = tf.matmul(moves, apm_map, transpose_b=True)
    out = out[..., :90 * 90]
    out = tf.reshape(out, [-1, 90, 90])
    pu = set_zero_sum(tf.reduce_sum(out, axis=-1))
    pd = set_zero_sum(tf.reduce_sum(out, axis=-2))
    return pu, pd
