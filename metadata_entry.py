#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage
import czifile
import argparse
import json
import tools
import datetime

def ask_selection(question, options):
    while True:
        print(f'{question} ({"/".join(options)}):')
        x = input()
        if x in options.keys():
            return options[x]

def ask_value(question, default):
    print(f'{question} (default={default}, accept with enter)')
    x = input()
    if x == '':
        return default
    return x

def ask_date(question):
    while True:
        print(f'{question} (yyyy-mm-dd)')
        x = input()
        try:
            datetime.datetime.strptime(x, '%Y-%m-%d')
            return x
        except:
            print("Could not understand date")

def process_image(in_path, section, out_path, downres):
    file = czifile.CziFile(in_path)
    
    fullres = tools.czi_read_section(file, section)
    fullres = fullres.squeeze()

    downres_factor = downres
    lowres = skimage.transform.downscale_local_mean(fullres, (1, downres_factor, downres_factor))

    def auto_clim(img):
        return np.percentile(img, (2, 98))

    def onkey(event):
        if event.key == 'q':
            plt.close('all')

    ax1 = None
    for i in range(4):
        fig = plt.figure()
        ax = fig.add_subplot(111, sharex=ax1, sharey=ax1)
        cid = fig.canvas.mpl_connect('key_press_event', onkey)
        plt.imshow(lowres[i], clim=auto_clim(lowres[i]))
        if ax1 is None:
            ax1 = ax

    plt.show()

    keep = ask_selection("Keep this image?", {"y": True, "n": False, "s": None})
    if keep is None:
        return

    metadata = {
        'keep': keep,
    }

    if keep:
        # guess sample names
        guessed_mouse_line = None
        guessed_mouse_number = None
        try:
            print(in_path.parent.name)
            guessed_mouse_line = in_path.parent.name.split(' ')[0]
            guessed_mouse_number = in_path.parent.name.split(' ')[1]
        except:
            pass

        metadata['mouse_line'] = ask_value("Mouse line?", guessed_mouse_line)
        metadata['mouse_number'] = ask_value("Mouse number?", guessed_mouse_number)
        #metadata['measurement_date'] = ask_date("Measurement date?") # We can get it from czi metadata
        metadata['czifile'] = str(in_path)
        metadata['section'] = section

    tools.create_dir_for_path(out_path)
    with open(out_path, 'w') as f:
        json.dump(metadata, f)

def main():
    parser = argparse.ArgumentParser(prog = 'selection.py')
    parser.add_argument('in_filename', nargs='*')
    parser.add_argument('--only-new', action='store_true')
    parser.add_argument('-d', '--downres', type=int, default=16)
    args = parser.parse_args()


    for filenames in tools.get_section_files(args.in_filename):
        if args.only_new and filenames.user_metadata.exists():
            continue
        print(f"Processing {filenames.czi} section {filenames.section_nr}")
        process_image(filenames.czi, filenames.section_nr, filenames.user_metadata, args.downres)


if __name__ == '__main__':
    main()
