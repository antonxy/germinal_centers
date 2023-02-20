#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage
import scipy
import czifile
import sys

filename = sys.argv[1]

fullres = czifile.imread(filename)
fullres = fullres.squeeze()

#partial = fullres[:, :2000, :2000]
#np.save('partial.npy', partial)
#fullres = np.load('partial.npy')

downres_factor = 8

lowres = skimage.transform.downscale_local_mean(fullres, (1, downres_factor, downres_factor))

def find_contour(channel):
    # Blur the image to ignore small lines and stuff
    blur = skimage.filters.gaussian(channel, sigma=10)

    # Separate fore- and background using otsu threshold
    th = skimage.filters.threshold_otsu(blur)
    thresholded = blur > th
    
    # fill holes
    filled = scipy.ndimage.binary_fill_holes(thresholded)
    
    # reduce size of contour at edges so that we are for sure inside the cutting
    erosion = 100
    eroded = skimage.morphology.binary_erosion(filled, np.ones((erosion, erosion)))
    
    # convert binary image to contour line
    contours = skimage.measure.find_contours(eroded, 0.5)
    assert len(contours) == 1
    return contours[0], eroded

#contour, lowres_mask = find_contour(lowres[1])

background = 0

def find_blob(position, image):
    global background
    region_size = 500
    region = image[3, position[0] - region_size:position[0] + region_size, position[1] - region_size:position[1] + region_size] - background

    dr_fac = 2

    threshold = 200

    fp = skimage.morphology.disk(5)
    fp2 = skimage.morphology.disk(2)
    lr = skimage.transform.downscale_local_mean(region, (dr_fac, dr_fac))
    median = scipy.ndimage.maximum_filter(lr, footprint=fp)
    median = skimage.filters.gaussian(median, sigma=2)
    value_at_position = median[region_size // dr_fac, region_size // dr_fac]
    thr = median > threshold
    thr = skimage.morphology.erosion(thr, fp)
    thr = skimage.segmentation.flood(thr, (region_size // dr_fac, region_size // dr_fac))
    thr = skimage.morphology.dilation(thr, fp2)
    blobs_m = skimage.measure.find_contours(thr, .5)

    #threshold = 700
    #blur = skimage.filters.gaussian(region[3] > threshold, sigma=40)
    #blob = skimage.measure.find_contours(blur, 0.2)[0]

    #blur = skimage.filters.gaussian(region, sigma=40)
    #value_at_position = blur[region_size, region_size]
    #thr = blur > threshold
    #thr = skimage.segmentation.flood(thr, (region_size, region_size))
    #blobs = skimage.measure.find_contours(thr, .5)

    #s = np.linspace(0, 2*np.pi, 50)
    #r = region_size + 100*np.sin(s)
    #c = region_size + 100*np.cos(s)
    #init = np.array([r, c]).T
    #blur = skimage.filters.gaussian(region, 15, preserve_range=False)
    #snake = skimage.segmentation.active_contour(region,
    #                       init, alpha=0.000, beta=10, gamma=0.001, w_line=0, w_edge=1)
    #print(snake)

    fig, [ax1, ax2] = plt.subplots(1, 2)
    im1 = ax1.imshow(region)
    plt.colorbar(im1, ax=ax1)
    #ax1.plot(init[:, 1], init[:, 0], '--r', lw=3)
    #ax1.plot(snake[:, 1], snake[:, 0], '-g', lw=3)
#    if len(blobs) > 0:
#        print("Found blob")
#        blob = blobs[0]
#        ax1.plot(blob[:, 1], blob[:, 0], linewidth=2, c='r')
    if len(blobs_m) > 0:
        print("Found blob")
        blob = blobs_m[0] * dr_fac
        ax1.plot(blob[:, 1], blob[:, 0], linewidth=2, c='g')

    im2 = ax2.imshow(median)
    plt.colorbar(im2, ax=ax2)

    plt.show()
    

def onkey_bigplot(event):
    global background
    print(event.key, event.xdata, event.ydata)
    if event.key == 'b':
        p = (int(event.ydata), int(event.xdata))
        background = lowres[3, p[0] - 10: p[0] + 10, p[1] - 10: p[1] + 10].mean()
        print(f"Background: {background}")
    if event.key == 'a':
        find_blob((int(event.ydata * downres_factor), int(event.xdata * downres_factor)), fullres)


fig, ax = plt.subplots()
ax.imshow(lowres[3])
#ax.plot(contour[:, 1], contour[:, 0], linewidth=2, c='r');
cid = fig.canvas.mpl_connect('key_press_event', onkey_bigplot)
plt.show()

fig.canvas.mpl_disconnect(cid)
