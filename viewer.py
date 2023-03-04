#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage
import czifile
import argparse

filename = sys.argv[1]

parser = argparse.ArgumentParser(prog = 'blob_segmentation.py')
parser.add_argument('in_filename')
parser.add_argument('-d', '--downres', type=int, default=1)
parser.add_argument('-c', '--clim_auto', action='store_true')
args = parser.parse_args()

try:
    fullres = czifile.imread(args.in_filename)
except ValueError as e:
    print(e)
    print("Trying as npy")
    fullres = np.load(args.in_filename)

fullres = fullres.squeeze()

downres_factor = args.downres
lowres = skimage.transform.downscale_local_mean(fullres, (1, downres_factor, downres_factor))

def auto_clim(img):
    return np.percentile(img, (2, 98))

labels = []
scatters = []
figs = []

def onkey(event):
    if event.key == 'q':
        print(labels)
        sys.exit(0)
    if event.key == 'c':
        #fig.canvas.restore_region(bg) # should be done if we want to remove labels also I guess
        # Use this earlier: bg = fig.canvas.copy_from_bbox(fig.bbox)
        label = (event.xdata, event.ydata)
        labels.append(label)
        for ax, s in scatters:
            s.set_xdata([x for x, y in labels])
            s.set_ydata([y for x, y in labels])
            ax.draw_artist(s)
        for f in figs:
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()

ax1 = None
for i in range(4):
    fig = plt.figure()
    figs.append(fig)
    ax = fig.add_subplot(111, sharex=ax1, sharey=ax1)
    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    clim = auto_clim(lowres[i]) if args.clim_auto else None
    plt.imshow(lowres[i], clim=clim)
    scatters.append((ax, plt.plot([], [], c='r', marker='o',ls='', animated=True)[0]))
    if ax1 is None:
        ax1 = ax

plt.show()
