#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
"""
General comments on how chunkparser works.

A "training record" or just "record" is a fixed-length packed byte array. Typically
records are generated during training and are stored together by game, one record for 
each position in the game, but this arrangement is not required.
Over dev time additional fields have been added to the training record, most of which 
just put additional information after the end of the byte array used in the previous 
version. Currently supported training record versions are V3, V4, V5, and V6.

shufflebuffer.ShuffleBuffer is a simple structure holding an array of training
records that are efficiently randomized and replaced as needed. All records in
ShuffleBuffer are adjusted to be the same number of bytes by appending unused 
bytes *before* being put in the shuffler. 
byte padding is done in chunkparser.ChunkParser.sample_record()
sample_record() also skips most training records to avoid sampling over-correlated
positions since they typically are from sequential positions in a game.

Current implementation of "diff focus" also is in sample_record() and works by
probabilistically skipping records according to how accurate the no-search
eval ("orig_q") is compared to eval after search ("best_q") as well as the
recorded policy_kld (a measure of difference between no search policy and the
final policy distribution). It does not use draw values at this point. Putting
diff focus here is efficient because it runs in parallel workers and peeks at
the records without requiring any unpacking.

The constructor for chunkparser.ChunkParser() sets a bunch of class constants
and creates a fixed number of parallel Python multiprocessing.Pipe objects,
which consist of a "reader" and a "writer". The writer(s) get data directly
from training data files and write them into the pipe using the writer.send_bytes()
method. The reader(s) get data out of the pipe using the reader.rev_bytes()
method and feed them to the ShuffleBuffer using its insert_or_replace() method,
which also handles the shuffling itself.

Records come back out of the ShuffleBuffer (already a fixed byte number
regardless of training version) using the multiplexed generators specified in
the ChunkParser.parse() method. They are first recovered as raw byte records
in the vX_gen() method (currently v6_gen), then converted to tuples of more
interpretable data in the convert_vX_to_tuple() method and finally sent on
to tensorflow in training batches by the batch_gen() method.
"""

import itertools
import multiprocessing as mp
import numpy as np
import random
import shufflebuffer as sb
import struct
import unittest
import gzip
from time import time, sleep
from select import select

n_future_probs = 2
n_future_boards = 16

V7B_VERSION = struct.pack("i", 170)
V7_VERSION = struct.pack("i", 7)
V6_VERSION = struct.pack("i", 6)
V5_VERSION = struct.pack("i", 5)
CLASSICAL_INPUT = struct.pack("i", 1)
V4_VERSION = struct.pack("i", 4)
V3_VERSION = struct.pack("i", 3)
V7B_STRUCT_STRING = "4si8248s1920sBBBbfffffffffffffffIHHfffHHffffffff" + "8248s" * n_future_probs + str(
    14 * 16 * n_future_boards) + "s"
V7_STRUCT_STRING = "4si8248s1920sBBBbfffffffffffffffIHHfffHHffffffff"
V6_STRUCT_STRING = '4si8248s1920sBBBbfffffffffffffffIHHQ'
V5_STRUCT_STRING = '4si8248s1920sBBBbfffffff'
V4_STRUCT_STRING = '4s8248s1920sBBBbffff'
V3_STRUCT_STRING = '4s8248s1920sBBBb'

v7b_struct = struct.Struct(V7B_STRUCT_STRING)
v7_struct = struct.Struct(V7_STRUCT_STRING)
v6_struct = struct.Struct(V6_STRUCT_STRING)
v5_struct = struct.Struct(V5_STRUCT_STRING)
v4_struct = struct.Struct(V4_STRUCT_STRING)
v3_struct = struct.Struct(V3_STRUCT_STRING)

struct_sizes = {V7B_VERSION: v7b_struct.size, V7_VERSION: v7_struct.size,
                V6_VERSION: v6_struct.size, V5_VERSION: v5_struct.size,
                V4_VERSION: v4_struct.size, V3_VERSION: v3_struct.size}


def reverse_expand_bits(plane):
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(
        np.float32).tobytes()


# Interface for a chunk data source.
class ChunkDataSrc:
    def __init__(self, items):
        self.items = items

    def next(self):
        if not self.items:
            return None
        return self.items.pop()


def reverse_board(planes):
    # planes is 14 * 16 = 224 bytes
    # The order of the squares is reversed when switching sides, so first reverse 
    # the squares within each plane, then switch the first 7 planes (player 1 pieces) with planes 7:14 (player 2 pieces)
    # the 15th plane is 1 for empty squares and 0 for occupied squares, so it doesn't need to be reversed
    planes = bytearray(planes)
    for i in range(len(planes) // 16):
        planes_ = np.unpackbits(np.frombuffer(planes[i * 16:(i + 1) * 16], dtype=np.uint8), bitorder='little')[
                  :90].reshape((10, 9))
        planes_ = np.flip(planes_, axis=0).reshape(90)
        # pad to 128 bits
        planes_ = np.pad(planes_, (0, 38), 'constant')
        planes[i * 16:(i + 1) * 16] = np.packbits(planes_, bitorder='little').tobytes()

    planes[:112], planes[112:] = planes[112:], planes[:112]
    return planes


def chunk_reader(chunk_filenames, chunk_filename_queue):
    """
    Reads chunk filenames from a list and writes them in shuffled
    order to output_pipes.
    """
    chunks = []
    done = chunk_filenames

    while True:
        if not chunks:
            chunks, done = done, chunks
            random.shuffle(chunks)
        if not chunks:
            print("chunk_reader didn't find any chunks.")
            return None
        while len(chunks):
            filename = chunks.pop()
            done.append(filename)
            chunk_filename_queue.put(filename)
    print("chunk_reader exiting.")
    return None


class ChunkParser:

    def __init__(self,
                 chunks,
                 expected_input_format,
                 shuffle_size=1,
                 sample=1,
                 buffer_size=1,
                 batch_size=256,
                 diff_focus_min=1,
                 diff_focus_slope=0,
                 diff_focus_q_weight=6.0,
                 diff_focus_pol_scale=3.5,
                 workers=None):
        self.inner = ChunkParserInner(self, chunks, expected_input_format,
                                      shuffle_size, sample, buffer_size,
                                      batch_size, diff_focus_min,
                                      diff_focus_slope, diff_focus_q_weight,
                                      diff_focus_pol_scale, workers)

    def shutdown(self):
        """
        Terminates all the workers
        """
        for i in range(len(self.processes)):
            self.processes[i].terminate()
            self.processes[i].join()
            self.inner.readers[i].close()
            self.inner.writers[i].close()
        self.chunk_process.terminate()
        self.chunk_process.join()

    def parse(self):
        return self.inner.parse()

    def sequential(self):
        return self.inner.sequential()


def convert_v7b_to_tuple(content):
    """
    Unpack a v6 binary record to 5-tuple (state, policy pi, result, q, m)

    v6 struct format is (10296 bytes total):
                              size         1st byte index
    uint32_t version;                               0
    uint32_t input_format;                          4
    float probabilities[2062];  8248 bytes          8
    __uint128_t planes[120];    1920 bytes       8256
    uint8_t side_to_move;                       10176
    uint8_t rule50_count;                       10177
    // Bitfield with the following allocation:
    //  bit 7: side to move (input type 3)
    //  bit 6: position marked for deletion by the rescorer (never set by lc0)
    //  bit 5: game adjudicated (v6)
    //  bit 4: max game length exceeded (v6)
    //  bit 3: best_q is for proven best move (v6)
    //  bit 2: not used
    //  bit 1: not used
    //  bit 0: flip transform (input type 3)
    uint8_t invariance_info;                    10178
    uint8_t dep_result;                         10179
    float root_q;                               10180
    float best_q;                               10184
    float root_d;                               10188
    float best_d;                               10192
    float root_m;      // In plies.             10196
    float best_m;      // In plies.             10200
    float plies_left;                           10204
    float result_q;                             10208
    float result_d;                             10212
    float played_q;                             10216
    float played_d;                             10220
    float played_m;                             10224
    // The folowing may be NaN if not found in cache.
    float orig_q;      // For value repair.     10228
    float orig_d;                               10232
    float orig_m;                               10236
    uint32_t visits;                            10240
    // Indices in the probabilities array.
    uint16_t played_idx;                        10244
    uint16_t best_idx;                          10246
    float pol_kld;                              10248
    float q_st;                                 10252
    float d_st;                                 10256
    uint16_t opp_played_idx;                    10260
    uint16_t next_played_idx;                   10262
    float extra[8]                              10264
    ...                                         10296
    """
    # unpack the V6 content from raw byte array, arbitrarily chose 4 2-byte values
    # for the 8 "reserved" bytes
    (ver, input_format, probs, planes, stm, rule50_count, invariance_info, dep_result, root_q, best_q, root_d, best_d,
     root_m, best_m, plies_left, result_q, result_d, played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
     played_idx, best_idx, pol_kld, st_q, st_d, opp_played_idx, next_played_idx, f1, f2, f3, f4, f5, f6, f7, f8,
     opp_probs, next_probs, fut) = v7b_struct.unpack(content)
    """
    v5 struct format was (8308 bytes total)
        int32 version (4 bytes)
        int32 input_format (4 bytes)
        1858 float32 probabilities (7432 bytes)
        104 (13*8) packed bit planes of 8 bytes each (832 bytes)
        uint8 castling us_ooo (1 byte)
        uint8 castling us_oo (1 byte)
        uint8 castling them_ooo (1 byte)
        uint8 castling them_oo (1 byte)
        uint8 side_to_move (1 byte)
        uint8 rule50_count (1 byte)
        uint8 dep_ply_count (1 byte) (unused)
        int8 result (1 byte)
        float32 root_q (4 bytes)
        float32 best_q (4 bytes)
        float32 root_d (4 bytes)
        float32 best_d (4 bytes)
        float32 root_m (4 bytes)
        float32 best_m (4 bytes)
        float32 plies_left (4 bytes)
    """
    # v3/4 data sometimes has a useful value in dep_ply_count (now invariance_info),
    # so copy that over if the new ply_count is not populated.

    flat_planes = []
    for i in range(2):
        flat_planes.append(
            (np.zeros(90, dtype=np.float32) + i).tobytes())

    if plies_left == 0:
        plies_left = invariance_info
    plies_left = struct.pack("f", plies_left)

    # Unpack bit planes and cast to 32 bit float
    planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8), bitorder='little').reshape((-1, 128))[:, :90].reshape(
        (-1, 9)).astype(np.float32)
    rule50_divisor = 119.0
    if input_format > 3:
        rule50_divisor = 120.0
    rule50_plane = struct.pack("f", rule50_count / rule50_divisor) * 90

    if input_format == 1 or input_format == 2:
        middle_planes = flat_planes[stm]
    elif input_format == 3 or input_format == 4 or input_format == 132 or input_format == 5 or input_format == 133:
        middle_planes = flat_planes[0] + flat_planes[0]

    # Concatenate all byteplanes. Make the last plane all 1"s so the NN can
    # detect edges of the board more easily
    aux_plus_6_plane = flat_planes[0]
    if (input_format == 132
        or input_format == 133) and invariance_info >= 128:
        aux_plus_6_plane = flat_planes[1]
    planes = planes.tobytes() + \
             middle_planes + \
             rule50_plane + \
             aux_plus_6_plane + \
             flat_planes[1]

    assert len(planes) == ((8 * 15 * 1 + 4 * 1 * 1) * 10 * 9 * 4)

    if ver == V6_VERSION or ver == V7_VERSION:
        winner = struct.pack("fff", 0.5 * (1.0 - result_d + result_q),
                             result_d, 0.5 * (1.0 - result_d - result_q))
    else:
        dep_result = float(dep_result)
        assert dep_result == 1.0 or dep_result == -1.0 or dep_result == 0.0
        winner = struct.pack("fff", dep_result == 1.0, dep_result == 0.0,
                             dep_result == -1.0)

    def clip(x, lo, hi):
        return min(max(x, lo), hi)

    def qd_to_wdl(q, d):
        e = 1e-2
        assert -1.0 - e <= q <= 1.0 + e and 0.0 - e <= d <= 1.0 + e
        q = clip(q, -1.0, 1.0)
        d = clip(d, 0.0, 1.0)
        w = 0.5 * (1.0 - d + q)
        l = 0.5 * (1.0 - d - q)
        return (w, d, l)

    root_wdl = struct.pack("fff", *(qd_to_wdl(root_q, root_d)))

    st_wdl = struct.pack("fff", *(qd_to_wdl(st_q, st_d)))

    fut = np.unpackbits(np.frombuffer(fut, dtype=np.uint8), bitorder='little').reshape((-1, 128))[:, :90].reshape(
        (-1, 9)).astype(np.float32)

    return (planes, probs, winner, root_wdl, plies_left, st_wdl, opp_probs, next_probs, fut)


class ChunkParserInner:
    def __init__(self, parent, chunks, expected_input_format, shuffle_size,
                 sample, buffer_size, batch_size, diff_focus_min,
                 diff_focus_slope, diff_focus_q_weight, diff_focus_pol_scale,
                 workers):
        """
        Read data and yield batches of raw tensors.

        "parent" the outer chunk parser to store processes. Must not be stored by self directly or indirectly.
        "chunks" list of chunk filenames.
        "shuffle_size" is the size of the shuffle buffer.
        "sample" is the rate to down-sample.
        "diff_focus_min", "diff_focus_slope", "diff_focus_q_weight" and "diff_focus_pol_scale" control diff focus
        "workers" is the number of child workers to use.

        The data is represented in a number of formats through this dataflow
        pipeline. In order, they are:

        chunk: The name of a file containing chunkdata

        chunkdata: type Bytes. Multiple records of v6 format where each record
        consists of (state, policy, result, q)

        raw: A byte string holding raw tensors contenated together. This is
        used to pass data from the workers to the parent. Exists because
        TensorFlow doesn"t have a fast way to unpack bit vectors. 7950 bytes
        long.
        """

        self.expected_input_format = expected_input_format

        # set the down-sampling rate
        self.sample = sample
        # set the details for diff focus, defaults accept all positions
        self.diff_focus_min = diff_focus_min
        self.diff_focus_slope = diff_focus_slope
        self.diff_focus_q_weight = diff_focus_q_weight
        self.diff_focus_pol_scale = diff_focus_pol_scale
        # set the mini-batch size
        self.batch_size = batch_size
        # set number of elements in the shuffle buffer.
        self.shuffle_size = shuffle_size
        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)

        if workers > 0:
            print("Using {} worker processes.".format(workers))

            # Start the child workers running
            self.readers = []
            self.writers = []
            parent.processes = []
            self.chunk_filename_queue = mp.Queue(maxsize=4096)
            for _ in range(workers):
                read, write = mp.Pipe(duplex=False)
                p = mp.Process(target=self.task,
                               args=(self.chunk_filename_queue, write))
                p.daemon = True
                parent.processes.append(p)
                p.start()
                self.readers.append(read)
                self.writers.append(write)

            parent.chunk_process = mp.Process(target=chunk_reader,
                                              args=(chunks,
                                                    self.chunk_filename_queue))
            parent.chunk_process.daemon = True
            parent.chunk_process.start()
        else:
            self.chunks = chunks

    def sample_record(self, chunkdata):
        """
        Randomly sample through the v3/4/5/6/7 chunk data and select records in v6 format
        Downsampling to avoid highly correlated positions skips most records, and
        diff focus may also skip some records.
        """
        version = chunkdata[0:4]
        record_size = struct_sizes.get(version, None)
        if record_size is None:
            return

        n_chunks = len(chunkdata) // record_size
        if n_chunks == 0:
            return

        probs = [chunkdata[i + 8:i + 8 + 2062 * 4] for i in range(0, len(chunkdata), record_size)]
        # if there is a single legal move then the loss will be 0, so pick an arbitrary move
        probs.extend(n_future_probs * [struct.pack("f", 1.0) + struct.pack("f", -1.0) * 2061])

        white_boards = b""
        black_boards = b""

        ppb = 14

        for i in range(n_chunks):
            start = i * record_size
            plane = chunkdata[start + 8256:start + 8256 + 16 * ppb]
            if i % 2 == 0:  # this board is from white's perspective
                white_boards += plane
                black_boards += reverse_board(plane)
            else:
                white_boards += reverse_board(plane)
                black_boards += plane
        white_boards += white_boards[-16 * ppb:] * n_future_boards  # history is the final position if game over
        black_boards += black_boards[-16 * ppb:] * n_future_boards

        for i in range(0, len(chunkdata), record_size):
            if self.sample > 1:
                # Downsample, using only 1/Nth of the items.
                if random.randint(0, self.sample - 1) != 0:
                    continue  # Skip this record.

            idx = i // record_size
            record = chunkdata[i:i + record_size]

            if version == V6_VERSION or version == V7_VERSION:
                # diff focus code, peek at best_q, orig_q and pol_kld from record (unpacks as tuple with one item)
                best_q = struct.unpack('f', record[10184:10188])[0]
                orig_q = struct.unpack('f', record[10228:10232])[0]
                pol_kld = struct.unpack('f', record[10248:10252])[0]

                # if orig_q is NaN or pol_kld is 0, accept, else accept based on diff focus
                if not np.isnan(orig_q) and pol_kld > 0:
                    diff_q = abs(best_q - orig_q)
                    q_weight = self.diff_focus_q_weight
                    pol_scale = self.diff_focus_pol_scale
                    total = (q_weight * diff_q + pol_kld) / (q_weight +
                                                             pol_scale)
                    thresh_p = self.diff_focus_min + self.diff_focus_slope * total
                    if thresh_p < 1.0 and random.random() > thresh_p:
                        continue

            record += b"".join(probs[idx + 1: idx + 1 + n_future_probs])
            boards = white_boards if idx % 2 == 0 else black_boards
            record += boards[16 * ppb * idx:16 * ppb * (idx + n_future_boards)]

            yield record

    def single_file_gen(self, filename):

        with gzip.open(filename, "rb") as chunk_file:
            version = chunk_file.read(4)
            chunk_file.seek(0)
            if version == b'':
                return
            record_size = struct_sizes.get(version, None)
            if record_size is None:
                print("Unknown version {} in file {}".format(
                    version, filename))
                return
            chunkdata = chunk_file.read()
            for item in self.sample_record(chunkdata):
                yield item

    def sequential_gen(self):
        for filename in self.chunks:
            for item in self.single_file_gen(filename):
                yield item

    def sequential(self):
        # read from all files in order in this process.
        gen = self.sequential_gen()
        gen = self.tuple_gen(gen)  # convert v7->tuple
        gen = self.batch_gen(gen, allow_partial=False)  # assemble into batches
        for b in gen:
            yield b

    def task(self, chunk_filename_queue, writer):
        """
        Run in fork"ed process, read data from chunkdatasrc, parsing, shuffling and
        sending v6 data through pipe back to main process.
        """
        while True:
            filename = chunk_filename_queue.get()
            for item in self.single_file_gen(filename):
                writer.send_bytes(item)

    def v7_gen(self):
        """
        Read v7 records from child workers, shuffle, and yield
        records.
        """
        sbuff = sb.ShuffleBuffer(v7b_struct.size, self.shuffle_size)
        while len(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    def tuple_gen(self, gen):
        """
        Take a generator producing v7 records and convert them to tuples.
        applying a random symmetry on the way.
        """
        for r in gen:
            yield convert_v7b_to_tuple(r)

    def batch_gen(self, gen, allow_partial=True):
        """
        Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s) or (not allow_partial and len(s) != self.batch_size):
                return
            n_entries = len(s[0])
            yield tuple([b"".join([x[i] for x in s]) for i in range(n_entries)])

    def parse(self):
        """
        Read data from child workers and yield batches of unpacked records
        """
        gen = self.v7_gen()  # read from workers
        gen = self.tuple_gen(gen)  # convert v7->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b


# # Tests to check that records parse correctly
# class ChunkParserTest(unittest.TestCase):
#     def setUp(self):
#         self.v4_struct = struct.Struct(V4_STRUCT_STRING)

#     def generate_fake_pos(self):
#         """
#         Generate a random game position.
#         Result is ([[64] * 104], [1]*5, [1858], [1], [1])
#         """
#         # 0. 104 binary planes of length 64
#         planes = [
#             np.random.randint(2, size=64).tolist() for plane in range(104)
#         ]

#         # 1. generate the other integer data
#         integer = np.zeros(7, dtype=np.int32)
#         for i in range(5):
#             integer[i] = np.random.randint(2)
#         integer[5] = np.random.randint(100)

#         # 2. 1858 probs
#         probs = np.random.randint(9, size=1858, dtype=np.int32)

#         # 3. And a winner: 1, 0, -1
#         winner = np.random.randint(3) - 1

#         # 4. evaluation after search
#         best_q = np.random.uniform(-1, 1)
#         best_d = np.random.uniform(0, 1 - np.abs(best_q))
#         return (planes, integer, probs, winner, best_q, best_d)

#     def v4_record(self, planes, i, probs, winner, best_q, best_d):
#         pl = []
#         for plane in planes:
#             pl.append(np.packbits(plane))
#         pl = np.array(pl).flatten().tobytes()
#         pi = probs.tobytes()
#         root_q, root_d = 0.0, 0.0
#         return self.v4_struct.pack(V4_VERSION, pi, pl, i[0], i[1], i[2], i[3],
#                                    i[4], i[5], i[6], winner, root_q, best_q,
#                                    root_d, best_d)

#     def test_structsize(self):
#         """
#         Test struct size
#         """
#         self.assertEqual(self.v4_struct.size, 8292)

#     def test_parsing(self):
#         """
#         Test game position decoding pipeline.
#         """
#         truth = self.generate_fake_pos()
#         batch_size = 4
#         records = []
#         for i in range(batch_size):
#             record = b""
#             for j in range(2):
#                 record += self.v4_record(*truth)
#             records.append(record)

#         parser = ChunkParser(ChunkDataSrc(records),
#                              shuffle_size=1,
#                              workers=1,
#                              batch_size=batch_size)
#         batchgen = parser.parse()
#         data = next(batchgen)

#         batch = (np.reshape(np.frombuffer(data[0], dtype=np.float32),
#                             (batch_size, 112, 64)),
#                  np.reshape(np.frombuffer(data[1], dtype=np.int32),
#                             (batch_size, 1858)),
#                  np.reshape(np.frombuffer(data[2], dtype=np.float32),
#                             (batch_size, 3)),
#                  np.reshape(np.frombuffer(data[3], dtype=np.float32),
#                             (batch_size, 3)))

#         fltplanes = truth[1].astype(np.float32)
#         fltplanes[5] /= 99
#         for i in range(batch_size):
#             data = (batch[0][i][:104],
#                     np.array([batch[0][i][j][0] for j in range(104, 111)]),
#                     batch[1][i], batch[2][i], batch[3][i])
#             self.assertTrue((data[0] == truth[0]).all())
#             self.assertTrue((data[1] == fltplanes).all())
#             self.assertTrue((data[2] == truth[2]).all())
#             scalar_win = data[3][0] - data[3][-1]
#             self.assertTrue(np.abs(scalar_win - truth[3]) < 1e-6)
#             scalar_q = data[4][0] - data[4][-1]
#             self.assertTrue(np.abs(scalar_q - truth[4]) < 1e-6)

#         parser.shutdown()


# if __name__ == "__main__":
#     unittest.main()


def apply_alpha(qs, alpha, alt_signs=True):
    if not isinstance(qs, np.ndarray):
        qs = np.array(qs)

    n = len(qs)
    signs = (-1) ** np.arange(n) if alt_signs else 1
    qs = qs * signs
    # Create an array with alpha^(i-j) at (i, j) if this is at most 1 and 0 otherwise.
    q_st = np.zeros(n)
    val = 0
    for i in range(n):
        if i == 0:
            val = qs[-1]
        else:
            val = alpha * val + qs[-i - 1] * (1 - alpha)
        q_st[-i - 1] = val

    q_st = q_st * signs

    return q_st


def rescore_file(filename, st_alpha=1 - 1 / 6, lt_alpha=1 - 1 / 24):
    v6_struct = struct.Struct(V6_STRUCT_STRING)
    v7_struct = struct.Struct(V7_STRUCT_STRING)

    record_size = v6_struct.size
    # C:/leeladata/train/training-run2-test77-20211214-1618/*.gz
    # apply ema with alpha
    cd_array = bytearray()

    try:
        with gzip.open(filename, "rb") as chunk_file:
            chunk_file.seek(0)
            chunkdata = chunk_file.read()
            if len(chunkdata) == 0:
                return
            version = chunkdata[0:4]
            assert version == V6_VERSION

            n_chunks = len(chunkdata) // record_size

            # Gather q bytes
            qs = []
            ds = []
            play_idx = []
            for i in range(n_chunks):
                qs.append(struct.unpack(
                    "f", chunkdata[i * record_size + 10180:i * record_size + 10184])[0])
                ds.append(struct.unpack(
                    "f", chunkdata[i * record_size + 10188:i * record_size + 10192])[0])
                play_idx.append(
                    chunkdata[i * record_size + 10244:i * record_size + 10246])
            # put max value if game has ended
            play_idx += [struct.pack("H", 65535)] * 2

            st_q = apply_alpha(qs, st_alpha)
            st_d = apply_alpha(ds, st_alpha, alt_signs=False)
            cd_array = b""
            for i in range(n_chunks):
                new_chunk = bytearray(
                    chunkdata[i * record_size:(i + 1) * record_size] + b"\x00" * (v7_struct.size - record_size))
                if abs(st_q[i]) > 1 + 1e-6:
                    print(f"Got {st_q[i]}")
                # root q
                new_chunk[10252:10256] = struct.pack("f", st_q[i])
                # root d
                new_chunk[10256:10260] = struct.pack("f", max(st_d[i], 0))
                new_chunk[0:4] = V7_VERSION
                new_chunk[10260:10262] = play_idx[i + 1]
                new_chunk[10262:10264] = play_idx[i + 2]
                assert len(new_chunk) == v7_struct.size
                cd_array += new_chunk

    except Exception as e:
        print(f"Could not read {filename}, got {e}")
    if cd_array == bytearray():
        return
    with gzip.open(filename, 'wb') as chunk_file:
        chunk_file.write(bytes(cd_array))


def check_v7_file(filename):
    v7_struct = struct.Struct(V7_STRUCT_STRING)
    record_size = v7_struct.size
    with gzip.open(filename, "rb") as chunk_file:
        chunk_file.seek(0)
        chunkdata = chunk_file.read()
        if len(chunkdata) == 0:
            return
        version = chunkdata[0:4]
        assert version == V7_VERSION
        assert len(chunkdata) % v7_struct.size == 0
        n_chunks = len(chunkdata) // v7_struct.size

        for i in range(n_chunks):
            chunk = chunkdata[i * record_size:(i + 1) * record_size]
            st_q = struct.unpack("f", chunk[10252:10256])[0]
            # root d
            st_d = struct.unpack("f", chunk[10256:10260])[0]

            opp_play = struct.unpack("H", chunk[10260:10262])[0]
            my_next_play = struct.unpack("H", chunk[10262:10264])[0]

            print(
                f"st_q: {st_q}, st_d: {st_d}, opp_play: {opp_play}, my_next_play: {my_next_play}")


def rescore_files(filenames, progress, task_id, **kwargs):
    i = 0
    for filename in filenames:
        rescore_file(filename, **kwargs)
        i += 1
        progress[task_id] = {"progress": i + 1, "total": len(filenames)}


def rescore_files_normal(filenames, **kwargs):
    n_chunks = 0
    i = 0
    for filename in filenames:
        rescore_file(filename, **kwargs)
        i += 1
        print(f"Processed {i} of {len(filenames)} chunks")


def rescore(filenames, n_workers=16, n_jobs=1000, **kwargs):
    from concurrent.futures import ProcessPoolExecutor
    from rich import progress
    import multiprocessing

    if isinstance(filenames, str):
        if not filenames.endswith(".gz"):
            filenames = filenames + "/*.gz"
        import glob
        filenames = glob.glob(filenames)

    print(
        f"Rescoring {len(filenames)} files with {n_workers} workers and {n_jobs} jobs each")

    with progress.Progress(
            "[progress.description]{task.description}",
            progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            progress.TimeRemainingColumn(),
            progress.TimeElapsedColumn(),
            refresh_per_second=1,  # bit slower updates
    ) as progress:
        futures = []  # keep track of the jobs
        with multiprocessing.Manager() as manager:
            # this is the key - we share some state between our
            # main process and our worker functions
            _progress = manager.dict()
            overall_progress_task = progress.add_task(
                "[green]All jobs progress:")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for n in range(0, n_jobs):  # iterate over the jobs we need to run
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = progress.add_task(f"task {n}", visible=False)
                    lo = n * len(filenames) // n_jobs
                    hi = min((n + 1) * len(filenames) //
                             n_jobs, len(filenames))
                    futures.append(executor.submit(
                        rescore_files, filenames[lo:hi], progress=_progress, task_id=task_id, **kwargs))

                # monitor the progress:
                while (n_finished := sum([future.done() for future in futures])) < len(
                        futures
                ):
                    progress.update(
                        overall_progress_task, completed=n_finished, total=len(
                            futures)
                    )
                    for task_id, update_data in _progress.items():
                        latest = update_data["progress"]
                        total = update_data["total"]
                        # update the progress bar for this task:
                        progress.update(
                            task_id,
                            completed=latest,
                            total=total,
                            visible=latest < total,
                        )

                # raise any errors:
                for future in futures:
                    future.result()


def ascii_board(bitboard, bn=0):
    planes = np.frombuffer(bitboard, dtype=np.float32)
    planes = np.reshape(planes, [124, 90])
    planes = planes[15 * bn:15 * (bn + 1)]
    planes = np.transpose(planes, [1, 0])
    for square in planes:
        if square[14] != 0:
            print(planes)
            break


def sample_record(chunkdata):
    """
    Randomly sample through the v3/4/5/6/7 chunk data and select records in v6 format
    Downsampling to avoid highly correlated positions skips most records, and
    diff focus may also skip some records.
    """
    version = chunkdata[0:4]
    record_size = struct_sizes.get(version, None)
    if record_size is None:
        return

    n_chunks = len(chunkdata) // record_size
    if n_chunks == 0:
        return

    probs = [chunkdata[i + 8:i + 8 + 2062 * 4] for i in range(0, len(chunkdata), record_size)]
    # if there is a single legal move then the loss will be 0, so pick an arbitrary move
    probs.extend(n_future_probs * [struct.pack("f", 1.0) + struct.pack("f", -1.0) * 2061])

    ppb = 14
    white_boards = b""
    black_boards = b""

    for i in range(n_chunks):
        start = i * record_size
        plane = chunkdata[start + 8256:start + 8256 + 16 * ppb]
        if i % 2 == 0:  # this board is from white's perspective
            white_boards += plane
            black_boards += reverse_board(plane)
        else:
            white_boards += reverse_board(plane)
            black_boards += plane
    white_boards += white_boards[-16 * ppb:] * n_future_boards  # history is the final position if game over
    black_boards += black_boards[-16 * ppb:] * n_future_boards

    for i in range(0, len(chunkdata), record_size):
        idx = i // record_size
        record = chunkdata[i:i + record_size]

        record += b"".join(probs[idx + 1: idx + 1 + n_future_probs])
        boards = white_boards if idx % 2 == 0 else black_boards
        record += boards[16 * ppb * (idx + 1):16 * ppb * (1 + idx + n_future_boards)]

        yield record


def gen(filename):
    with gzip.open(filename, "rb") as chunk_file:
        version = chunk_file.read(4)
        chunk_file.seek(0)
        if version == b'':
            return
        record_size = struct_sizes.get(version, None)
        if record_size is None:
            print("Unknown version {} in file {}".format(
                version, filename))
            return
        chunkdata = chunk_file.read()
        i = 0
        for item in sample_record(chunkdata):
            (planes, probs, winner, root_wdl, plies_left, st_wdl, opp_probs, next_probs, fut) = convert_v7b_to_tuple(
                item)
            ascii_board(planes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()
    rescore(args.folder)
