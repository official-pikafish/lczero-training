import numpy as np
from policy_index import policy_index

columns = 'abcdefghi'
rows = '0123456789'


def index_to_position(x):
    return columns[x % 9] + rows[x // 9]


def make_map():
    z = np.zeros((90 * 90, 2062), dtype=np.int32)
    # loop for moves (for i in 0:2062, stride by 1)
    for pickup_index in range(90):
        for putdown_index in range(90):
            move = index_to_position(pickup_index) + index_to_position(putdown_index)
            if policy_index.count(move) == 0:
                continue
            move = policy_index.index(move)
            z[putdown_index + (90 * pickup_index), move] = 1

    return z


if __name__ == "__main__":
    header = \
"""/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2019 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace lczero {
"""
    line_length = 15
    maps = make_map()
    with open('attention_policy_map.h', 'w') as f:
        f.write(header + '\n')
        f.write('const short kAttnPolicyMap[] = {\n')
        for move_index in range(8100):
            legal_move_one_hot = maps[move_index]
            if legal_move_one_hot.sum() == 0:
                i = -1
            else:
                i = np.argmax(legal_move_one_hot)
            if move_index % line_length == 0 and move_index > 0:
                f.write('\n')
            f.write(str(i).rjust(5))
            f.write(',')
        f.write('};\n\n')
        f.write('}  // namespace lczero')
