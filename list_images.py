#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import argparse
import os
import ast
import pathlib
import json
import tools

def main():
    parser = argparse.ArgumentParser(prog = 'list_images.py')
    parser.add_argument('--output', '-o')
    parser.add_argument('in_filename', nargs='*')
    args = parser.parse_args()
    
    res = []
    undecided = 0
    undecided_names = []

    for filenames in tools.get_section_files(args.in_filename):
        print(f"Processing {filenames.czi} section {filenames.section_nr}")
        if filenames.user_metadata.exists():
            with open(filenames.user_metadata, 'r') as f:
                metadata = json.load(f)
            if metadata['keep']:
                metadata['has_mask'] = int(filenames.mask_polygon.exists())
                res.append(metadata)
        else:
            undecided += 1
            undecided_names.append(filenames.czi)
            
            
        
    table = pd.DataFrame(res)
    table['count'] = 1
    table = table.groupby(by=["mouse_line", "mouse_number"]).agg({"count": 'sum', "has_mask": "sum"})
    table.to_csv(args.output)
    
    print(f"Number of slides that we have not decided to keep or skip yet: {undecided}")
    
    for f in undecided_names:
        print(f)


if __name__ == '__main__':
    main()
