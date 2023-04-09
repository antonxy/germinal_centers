#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage
import czifile
import argparse
import tools

def process_image(in_path, out_path, downres):
    fullres = czifile.imread(in_path)
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

    keep = None
    while True:
        print('Keep this image? (y/n/s):')
        x = input()
        if x == 'y':
            keep = True
        elif x == 'n':
            keep = False
        elif x == 's':
            keep = None # Skip
        else:
            continue
        break

    if keep is not None:
        tools.create_dir_for_path(out_path)
        with open(out_path, 'w') as f:
            json.dump({'keep': keep}, f)

def main():
    parser = argparse.ArgumentParser(prog = 'selection.py')
    parser.add_argument('in_filename', nargs='*')
    parser.add_argument('--only-new', action='store_true')
    parser.add_argument('-d', '--downres', type=int, default=16)
    args = parser.parse_args()


    for filenames in tools.get_files(args.in_filename):
        if args.only_new and filenames.selection.exists():
            continue
        print(f"Processing {filenames.czi}")
        process_image(filenames.czi, filenames.selection, args.downres)


if __name__ == '__main__':
    main()
