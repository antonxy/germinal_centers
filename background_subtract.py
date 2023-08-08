#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import czifile
import sys
import argparse
from skimage.transform import downscale_local_mean
import xmltodict
import json
import pathlib
import tools
import traceback

def process_image(in_path, section, out_path, metadata_path, downres_factor, debug):

    file = czifile.CziFile(in_path)

    def parse_metadata(metadata):
        parsed = xmltodict.parse(metadata)

        pixel_size_str = parsed['ImageDocument']['Metadata']['ImageScaling']['ImagePixelSize']
        pixel_size_x, pixel_size_y = [float(x) for x in pixel_size_str.split(',')]
        assert pixel_size_x == pixel_size_y

        channels = parsed['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']
        acquisitionDate = parsed['ImageDocument']['Metadata']['Information']['Image']['AcquisitionDateAndTime']

        channels_recs = []
        for channel in channels:
            channel_id = channel['@Id']
            channel_name = channel['@Name']
            exp_time = channel['ExposureTime']
            light_intensity = channel['LightSourcesSettings']['LightSourceSettings']['Intensity']
            channels_recs.append({'id': channel_id, 'name': channel_name, 'exp_time': exp_time, 'light_intensity': light_intensity})
        return channels_recs, pixel_size_x, acquisitionDate 

    channels, pixel_size, acquisitionDate = parse_metadata(file.metadata())

    # Check that the order of dimensions in the file is as assumed
    print(f"{file.axes=}")
    assert file.axes == "SCYX0"


    # Order subblocks into lists by channel
    subblocks_by_channel = {}
    for block in file.filtered_subblock_directory:
        s = block.start
        if s[0] != section:
            continue
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

    if debug:
        for k, v in background.items():
            plt.figure()
            plt.imshow(v.squeeze())

    # (channel, subblock) for all subblocks
    subblocks_with_channel = []
    for k, v in subblocks_by_channel.items():
        [subblocks_with_channel.append((k, v)) for v in v]

    def subtract_background(directory_entry, background):
        return directory_entry.data_segment().data() - background

    def assemble_mosaic(blocks):
        start = tools.czi_section_start(file, section)
        shape = tools.czi_section_shape(file, section)
        out = np.zeros(shape[1:], file.dtype)

        for channel, directory_entry in blocks:
            tile = subtract_background(directory_entry, background[channel])
            assert tile.shape[0] == 1
            index = tuple(slice(i - j, i - j + k) for i, j, k in
                          zip(directory_entry.start[1:], start[1:], tile.shape[1:]))
            out[index] = tile[0]
        return out

    assembled = assemble_mosaic(subblocks_with_channel)
    print(assembled.shape)

    # Normalize the layout of the array
    # Dimensions should be (channel, y, x)
    assembled = assembled.squeeze()
    assert len(assembled.shape) == 3
    assert assembled.shape[0] == 4

    def reorder_channels(data, original_channels, desired_channels, remappings):
        remapped_original = [remappings.get(channel, channel) for channel in original_channels]
        ordering = [remapped_original.index(channel) for channel in desired_channels]
        if isinstance(data, np.ndarray):
            return data[ordering, :, :]
        else:
            return [data[o] for o in ordering]

    # Channels should be always in the same order (dapi, blue, red, green)
    original_channel_order = [channel['name'] for channel in channels]
    print(f"{original_channel_order=}")
    reorder = lambda d: reorder_channels(d, original_channel_order, ["DAPI", "AF488", "AF546", "AF647"], {"AF594": "AF546"})

    assembled = reorder(assembled)

    lowres = downscale_local_mean(assembled, (1, downres_factor, downres_factor))

    # If we don't use exactly min as background estimation we should use a signed dtype for export
    tools.create_dir_for_path(out_path)
    np.save(out_path, lowres)

    metadata = {
        'pixel_size_um': pixel_size * downres_factor,
        'pixel_size_original_um': pixel_size,
        'downres_factor': downres_factor,
        'channels': reorder(channels),
        'acquisition_date ': acquisitionDate 
    }
    tools.create_dir_for_path(metadata_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    if debug:
        plt.show()

def main():
    parser = argparse.ArgumentParser(prog = 'background_subtract.py')
    parser.add_argument('in_filename', nargs='*')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--only-new', action='store_true')
    parser.add_argument('--skip-fail', action='store_true')
    args = parser.parse_args()

    downres = 16

    for filenames in tools.get_section_files(args.in_filename, only_selected=True):
        if args.only_new and filenames.bg_sub.exists() and filenames.metadata.exists():
            continue
        print(f"Processing {filenames.czi} section {filenames.section_nr}")
        try:
            process_image(filenames.czi, filenames.section_nr, filenames.bg_sub, filenames.metadata, downres, args.debug)
        except Exception as e:
            traceback.print_exc()
            if not args.skip_fail:
                break


if __name__ == '__main__':
    main()
