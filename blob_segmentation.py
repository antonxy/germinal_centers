#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage
import scipy
import scipy.ndimage as ndi
import czifile
import sys
import argparse
import os
from skimage.transform import downscale_local_mean

parser = argparse.ArgumentParser(prog = 'blob_segmentation.py')
parser.add_argument('in_filename')
parser.add_argument('-o', '--out_filename', required=False)
parser.add_argument('-d', '--debug_folder', required=False)
args = parser.parse_args()

if args.debug_folder is not None:
    os.mkdir(args.debug_folder)

#fullres = czifile.imread(filename)
fullres = np.load(args.in_filename)
fullres = fullres.squeeze()

#partial = fullres[:, :2000, :2000]
#np.save('partial.npy', partial)
#fullres = np.load('partial.npy')

downres_factor = 16

def downscale_local_max(arr, factors):
    return skimage.measure.block_reduce(arr, block_size=factors, func=np.max, cval=0)

#lowres = downscale_local_mean(fullres, (1, downres_factor, downres_factor))
lowres = fullres


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
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '1_boundary.png'))

def histmax(data, bins, range=None):
    hist, bin_edges = np.histogram(data, bins=bins, range=range)
    max_idx = np.argmax(hist)
    return bin_edges[max_idx] + (bin_edges[max_idx + 1] - bin_edges[max_idx]) / 2


masked_area_background_intensity = np.array([histmax(lowres[i][mask], 200, range=(0, 1000)) for i in range(lowres.shape[0])])


masked_bg_sub = lowres - masked_area_background_intensity[:, np.newaxis, np.newaxis]

plt.figure()
plt.hist(masked_bg_sub[3][mask], bins=1000)
plt.axvline(x=0, c='r')
plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")

fig, ax = plt.subplots()
ax.imshow(masked_bg_sub[3], clim=(0, 400))
plt.plot(boundary[:, 1], boundary[:, 0], 'r')
plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")


fp = skimage.morphology.disk(3)
filtered = scipy.ndimage.median_filter(masked_bg_sub[3], footprint=fp)

fig, ax = plt.subplots()
ax.imshow(filtered, clim=(0, 400))
plt.title("Step 3: Median filter")


threshold = 100

plt.figure()
plt.hist(filtered[mask], bins=1000)
plt.axvline(x=0, c='r')
plt.axvline(x=threshold, c='g')
plt.title("Step 4: Threshold (user parameter)")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '4_threshold.png'))

thr = filtered > threshold

fig, ax = plt.subplots()
ax.imshow(thr)
plt.title("Step 4: Threshold (user parameter)")

# Give unique label to each blob
labels, num_labels = ndi.label(thr)


# Create table of properties for all blobs
extra_props = ('intensity_mean', 'eccentricity')
props = pd.DataFrame(skimage.measure.regionprops_table(labels, intensity_image=masked_bg_sub[3], properties=('label', 'area', 'bbox') + extra_props, separator='_')).set_index('label')

# Add areas inside the mask to each blob
labels_in_mask = labels.copy()
labels_in_mask[~mask] = 0
areas_inside_mask = pd.DataFrame(skimage.measure.regionprops_table(labels_in_mask, properties=('label', 'area'), separator='_')).set_index('label')
props = props.join(areas_inside_mask, rsuffix='_inside_mask')

# Add red channel info
areas_inside_mask = pd.DataFrame(skimage.measure.regionprops_table(labels_in_mask, intensity_image=masked_bg_sub[2], properties=('label', 'intensity_mean'), separator='_')).set_index('label')
props = props.join(areas_inside_mask, rsuffix='_red')

# Filter blobs
props_filtered = props[(props['area_inside_mask'] > (props['area'] / 2)) & (props['area'] > 20)]


def region_filter(prop, prop_in_mask):
    min_area = 10
    return prop.area > min_area and prop_in_mask.area > (prop.area / 2)

fig, ax = plt.subplots()
image_label_overlay = skimage.color.label2rgb(labels, image=masked_bg_sub[3] / 400, bg_label=0)
ax.imshow(image_label_overlay)
ax.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
plt.title("Step 5: Select regions >50% inside boundary and >min_area")
for row in props_filtered.itertuples():
    # draw rectangle around segmented area
    minr, minc, maxr, maxc = row.bbox_0, row.bbox_1, row.bbox_2, row.bbox_3
    rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '5_segmentation.png'))

plt.figure()
plt.imshow(masked_bg_sub[2], clim=(0, 400))
plt.title("Red channel")

# Maybe interesting parameters:
# Area, circulatiry
# total flourescence inside blob - maybe do this on fullres image?

props_filtered['intensity_sum'] = props_filtered.intensity_mean * props_filtered.area
props_filtered['intensity_sum_red'] = props_filtered.intensity_mean_red * props_filtered.area
print(props_filtered)

props_filtered.plot.scatter(x='area', y='intensity_mean', c='eccentricity', colormap='viridis')

if args.out_filename is not None:
    props_filtered.to_csv(args.out_filename)

if args.debug_folder is None:
    plt.show()
