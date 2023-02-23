#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import czifile
import sys
import argparse
from skimage.transform import downscale_local_mean

parser = argparse.ArgumentParser(prog = 'background_subtract.py')
parser.add_argument('in_filename')
parser.add_argument('out_filename')
parser.add_argument('-d', '--downres', type=int, default=1)
args = parser.parse_args()

file = czifile.CziFile(args.in_filename)

#TODO get dimensions from file, don't hardcode. Not the same for all files

# Order subblocks into lists by channel
subblocks_by_channel = {}
for block in file.filtered_subblock_directory:
    s = block.start
    key = s[1] # Channel
    if not key in subblocks_by_channel.keys():
        subblocks_by_channel[key] = []

    subblocks_by_channel[key].append(block)

# Estimate background for each channel
def estimate_background(blocks):
    data = np.stack([b.data_segment().data() for b in blocks])
    # TODO blur, median filter or something
    return np.min(data, axis=0)
background = {k: estimate_background(blocks) for k, blocks in subblocks_by_channel.items()}

# (channel, subblock) for all subblocks
subblocks_with_channel = []
for k, v in subblocks_by_channel.items():
    [subblocks_with_channel.append((k, v)) for v in v]

def subtract_background(directory_entry, background):
    return directory_entry.data_segment().data() - background

def assemble_mosaic(blocks):
    out = np.zeros(file.shape, file.dtype)

    for channel, directory_entry in blocks:
        tile = subtract_background(directory_entry, background[channel])
        index = tuple(slice(i - j, i - j + k) for i, j, k in
                      zip(directory_entry.start, file.start, tile.shape))
        try:
            out[index] = tile
        except ValueError as e:
            warnings.warn(str(e))
    return out

assembled = assemble_mosaic(subblocks_with_channel)
print(assembled.shape)

downres_factor = args.downres
lowres = downscale_local_mean(assembled, (1, 1, downres_factor, downres_factor, 1))

# If we don't use exactly min as background estimation we should use a signed dtype for export
np.save(args.out_filename, lowres)
