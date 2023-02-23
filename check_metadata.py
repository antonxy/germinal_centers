#!/usr/bin/env python3

import czifile
import argparse
import xmltodict
import sys
import pandas as pd

parser = argparse.ArgumentParser(prog = 'check_metadata.py')
parser.add_argument('in_filename')
parser.add_argument('-o', '--xml-out', required=False)
parser.add_argument('-c', '--csv-out', required=False)
args = parser.parse_args()

try:
    file = czifile.CziFile(args.in_filename)
except ValueError as e:
    print(e)
    sys.exit(1)

metadata = file.metadata()

if args.xml_out is not None:
    with open(args.xml_out, 'w') as f:
        f.write(metadata)

parsed = xmltodict.parse(file.metadata())

#tracks = parsed['ImageDocument']['Metadata']\
#['Experiment']['ExperimentBlocks']\
#['AcquisitionBlock']['SubDimensionSetups']\
#['RegionsSetup']['SubDimensionSetups']\
#['TilesSetup']['SubDimensionSetups']\
#['MultiTrackSetup']['Track']
#
#channels = {}
#
#for track in tracks:
#    channel = track['Channels']['Channel'] 
#    active = bool(channel['@IsActivated'])
#    if active:
#        channel_name = channel['@Name']
#        exp_time = channel['DataGrabberSetup']['CameraFrameSetup']['ExposureTime']['#text']
#        channels[channel_name] = {'exp_time': exp_time}
#
#for channel_name, value in sorted(channels.items()):
#    exp_time = value['exp_time']
#    print(f"{channel_name=} {exp_time=}")

# TODO get pixel size

channels = parsed['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']

channels_recs = []
for channel in channels:
    channel_id = channel['@Id']
    channel_name = channel['@Name']
    exp_time = channel['ExposureTime']
    light_intensity = channel['LightSourcesSettings']['LightSourceSettings']['Intensity']
    channels_recs.append({'id': channel_id, 'name': channel_name, 'exp_time': exp_time, 'light_intensity': light_intensity})

df = pd.DataFrame.from_records(channels_recs).set_index('id')
print(df)

if args.csv_out is not None:
    df.to_csv(args.csv_out)
