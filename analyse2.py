#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage
import scipy
import scipy.ndimage as ndi
import czifile
import sys

from skimage.transform import downscale_local_mean

filename = sys.argv[1]

#fullres = czifile.imread(filename)
fullres = np.load(filename)
fullres = fullres.squeeze()

#partial = fullres[:, :2000, :2000]
#np.save('partial.npy', partial)
#fullres = np.load('partial.npy')

downres_factor = 16

def downscale_local_max(arr, factors):
    return skimage.measure.block_reduce(arr, block_size=factors, func=np.max, cval=0)

lowres = downscale_local_mean(fullres, (1, downres_factor, downres_factor))


def find_boundary(channel):
    # Blur the image to ignore small lines and stuff
    blur = skimage.filters.gaussian(channel, sigma=10)

    # Separate fore- and background using otsu threshold
    th = skimage.filters.threshold_otsu(blur)
    thresholded = blur > th
    
    # fill holes
    filled = scipy.ndimage.binary_fill_holes(thresholded)
    
    # reduce size of contour at edges so that we are for sure inside the cutting
    erosion = 40
    eroded = skimage.morphology.binary_erosion(filled, skimage.morphology.disk(erosion))

    labels, num_labels = ndi.label(eroded)

    assert num_labels >= 1

    areas = np.zeros(num_labels)
    for i in range(0, num_labels):
        areas[i] = (labels == (i + 1)).sum()

    max_area_label = np.argmax(areas) + 1
    
    largest = labels == max_area_label
    
    # convert binary image to contour line
    contours = skimage.measure.find_contours(largest, 0.5)
    return contours[0], largest 

boundary, mask = find_boundary(lowres[1])

plt.figure()
plt.imshow(lowres[1])
plt.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
plt.title("Step 1: Find boundary")

def histmax(data, bins, range=None):
    hist, bin_edges = np.histogram(data, bins=bins, range=range)
    max_idx = np.argmax(hist)
    return bin_edges[max_idx] + (bin_edges[max_idx + 1] - bin_edges[max_idx]) / 2


masked_area_background_intensity = histmax(lowres[3][mask], 200, range=(0, 1000))

masked_bg_sub = lowres[3] - masked_area_background_intensity

plt.figure()
plt.hist(masked_bg_sub[mask], bins=1000)
plt.axvline(x=0, c='r')
plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")

fig, ax = plt.subplots()
ax.imshow(masked_bg_sub, clim=(0, 400))
plt.plot(boundary[:, 1], boundary[:, 0], 'r')
plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")


fp = skimage.morphology.disk(3)
filtered = scipy.ndimage.median_filter(masked_bg_sub, footprint=fp)

fig, ax = plt.subplots()
ax.imshow(filtered, clim=(0, 400))
plt.title("Step 3: Median filter")


threshold = 100

plt.figure()
plt.hist(filtered[mask], bins=1000)
plt.axvline(x=0, c='r')
plt.axvline(x=threshold, c='g')
plt.title("Step 4: Threshold (user parameter)")

thr = filtered > threshold

fig, ax = plt.subplots()
ax.imshow(thr)
plt.title("Step 4: Threshold (user parameter)")

labels, num_labels = ndi.label(thr)

# Keep only blobs that are more than 50% in the boundary
labels_in_mask = labels.copy()
labels_in_mask[~mask] = 0

props = skimage.measure.regionprops(labels, intensity_image=masked_bg_sub)
props_in_mask = skimage.measure.regionprops(labels_in_mask)

def key_zipper(lst1, lst2, get_key):    
    dict1 = {get_key(e): e for e in lst1}
    dict2 = {get_key(e): e for e in lst2}

    intersectKeys = [k for k in dict1.keys() if k in dict2.keys()]

    output = []

    for key in intersectKeys:
        output.append((key, dict1[key], dict2[key]))

    return output

props_zipped = key_zipper(props, props_in_mask, lambda p: p.label)


def region_filter(prop, prop_in_mask):
    min_area = 20
    return prop.area > min_area and prop_in_mask.area > (prop.area / 2)

fig, ax = plt.subplots()
image_label_overlay = skimage.color.label2rgb(labels, image=masked_bg_sub / 400, bg_label=0)
ax.imshow(image_label_overlay)
ax.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
plt.title("Step 5: Select regions >50% inside boundary and >min_area")
for label, prop, prop_in_mask in props_zipped:
    # take regions with large enough areas
    if region_filter(prop, prop_in_mask):
        # draw rectangle around segmented area
        minr, minc, maxr, maxc = prop.bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

# Maybe interesting parameters:
# Area, circulatiry
# total flourescence inside blob - maybe do this on fullres image?

plt.show()
