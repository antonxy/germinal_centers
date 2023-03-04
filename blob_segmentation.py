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
import ast
from pathlib import Path
from skimage.transform import downscale_local_mean

parser = argparse.ArgumentParser(prog = 'blob_segmentation.py')
parser.add_argument('in_filename')
parser.add_argument('-o', '--out_filename', required=False)
parser.add_argument('-d', '--debug_folder', required=False)
parser.add_argument('--reference', required=False)
args = parser.parse_args()

if args.debug_folder is not None:
    try:
        os.mkdir(args.debug_folder)
    except FileExistsError:
        pass

input_image = np.load(args.in_filename)

print(f"Image resolution: {input_image.shape}")

reference = None
if args.reference is not None:
    reference = ast.literal_eval(Path(args.reference).read_text())

def auto_clim(img):
    return np.percentile(img, (2, 98))

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
    # padded so that contour is closed if it touched the image edge
    contours = skimage.measure.find_contours(np.pad(largest, [(1, 1), (1, 1)]), 0.5) - np.array([1.0, 1.0])
    return contours[0], largest 

boundary, mask = find_boundary(input_image[1])

plt.figure()
plt.imshow(input_image[1])
plt.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
plt.title("Step 1: Find boundary")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '1_boundary.png'))

def histmax(data, bins, range=None):
    hist, bin_edges = np.histogram(data, bins=bins, range=range)
    max_idx = np.argmax(hist)
    max_val = hist[max_idx]

    def x_of_idx(idx):
        return bin_edges[idx] + (bin_edges[idx + 1] - bin_edges[idx]) / 2

    x_of_maximum = x_of_idx(max_idx)

    return x_of_maximum

masked_area_background_intensity = np.array([histmax(input_image[i][mask], 200, range=(0, 1000)) for i in range(input_image.shape[0])])
masked_bg_sub = input_image - masked_area_background_intensity[:, np.newaxis, np.newaxis]

plt.figure()
plt.hist(masked_bg_sub[3][mask], bins=1000)
plt.axvline(x=0, c='r')
plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '2_sub_hist.png'))

fig, ax = plt.subplots()
ax.imshow(masked_bg_sub[3], clim=auto_clim(masked_bg_sub[3]))
plt.plot(boundary[:, 1], boundary[:, 0], 'r')
plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '2_sub_img.png'))


fp = skimage.morphology.disk(3)
filtered = scipy.ndimage.median_filter(masked_bg_sub[3], footprint=fp)

def threshold_mad(im, k=4):
    med = np.median(im)
    mad = np.median(np.abs(im.astype(np.float32) - med))
    return med + mad * k * 1.4826

fig, ax = plt.subplots()
ax.imshow(filtered, clim=auto_clim(filtered))
plt.title("Step 3: Median filter")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '3_median.png'))


threshold = threshold_mad(filtered[mask], k=4)
print(f"{threshold=}")

plt.figure()
plt.hist(filtered[mask], bins=1000)
plt.axvline(x=0, c='r')
plt.axvline(x=threshold, c='g')
plt.title("Step 4: Threshold (MAD method)")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '4_threshold.png'))

thr = filtered > threshold

fig, ax = plt.subplots()
ax.imshow(thr)
plt.title("Step 4: Thresholded image")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '4_threshold_img.png'))

# Give unique label to each blob
labels, num_labels = ndi.label(thr)


# Create table of properties for all blobs
extra_props = ('intensity_mean', 'eccentricity')
props = pd.DataFrame(skimage.measure.regionprops_table(labels, intensity_image=masked_bg_sub[3], properties=('label', 'area', 'bbox', 'image') + extra_props, separator='_')).set_index('label')

# Add areas inside the mask to each blob
labels_in_mask = labels.copy()
labels_in_mask[~mask] = 0
areas_inside_mask = pd.DataFrame(skimage.measure.regionprops_table(labels_in_mask, properties=('label', 'area'), separator='_')).set_index('label')
props = props.join(areas_inside_mask, rsuffix='_inside_mask')

# Add red channel info
areas_inside_mask = pd.DataFrame(skimage.measure.regionprops_table(labels_in_mask, intensity_image=masked_bg_sub[2], properties=('label', 'intensity_mean'), separator='_')).set_index('label')
props = props.join(areas_inside_mask, rsuffix='_red')
# TODO maybe count red channel at edge of the blob and red channel inside
# There should be some on the edge but less at the center

# Filter blobs
props_filtered = props[(props['area_inside_mask'] > (props['area'] / 2)) & (props['area'] > 50)]


# Keep only the selected labels in the label image
def select_labels(labels, labels_to_keep, regionprops):
    mask = np.zeros(labels.shape, dtype=bool)
    for label in labels_to_keep:
        row = regionprops.loc[label]
        minr, minc, maxr, maxc = row.bbox_0, row.bbox_1, row.bbox_2, row.bbox_3
        mask[minr:maxr,minc:maxc][row.image] = True
    return labels * mask

labels_selected = select_labels(labels, props_filtered.index, props)

fig, ax = plt.subplots()
image_label_overlay = skimage.color.label2rgb(labels_selected, image=masked_bg_sub[3] / threshold / 3, bg_label=0)
ax.imshow(image_label_overlay)
ax.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
if reference is not None:
    ax.scatter([x for x, y in reference], [y for x, y in reference], c='r')
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
plt.imshow(masked_bg_sub[2], clim=auto_clim(masked_bg_sub[2]))
plt.title("Red channel")
if args.debug_folder is not None:
    plt.savefig(os.path.join(args.debug_folder, '6_red.png'))

# Maybe interesting parameters:
# Area, circulatiry
# total flourescence inside blob - maybe do this on fullres image?

props_filtered['intensity_sum'] = props_filtered.intensity_mean * props_filtered.area
props_filtered['intensity_sum_red'] = props_filtered.intensity_mean_red * props_filtered.area

export_columns = ['area', 'eccentricity', 'intensity_mean', 'intensity_sum', 'intensity_mean_red', 'intensity_sum_red']
props_export = props_filtered[export_columns]

print(props_export)

if args.out_filename is not None:
    props_export.to_csv(args.out_filename)

if args.debug_folder is None:
    plt.show()
