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
import pathlib
from skimage.transform import downscale_local_mean
import json
import tools
import traceback

# disable SettingWithCopyWarning which is mostly false positives
pd.options.mode.chained_assignment = None

def process_image(input_path, metadata_path, boundary_path, blob_csv_path, show_debug = False, save_debug_path = None):
    tools.create_dir_for_path(save_debug_path)

    input_image = np.load(input_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    with np.load(boundary_path) as data:
        boundaries = list(data.values())


    pixel_size = metadata['pixel_size_um']

    print(f"Image resolution: {input_image.shape}")
    print(f"Image data type: {input_image.dtype}")

    reference = None
    #if args.reference is not None:
    #    reference = ast.literal_eval(args.reference.read_text())

    def auto_clim(img):
        return np.percentile(img, (2, 98))

    mask = np.any(np.stack([skimage.draw.polygon2mask(input_image[1].shape, boundary) for boundary in boundaries]), axis=0)

    print(mask.shape)

    plt.figure()
    plt.imshow(input_image[1])
    for boundary in boundaries:
        plt.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
    plt.title("Step 1: Find boundary")
    if save_debug_path is not None:
        plt.savefig(save_debug_path / '1_boundary.png')

    def histmax(data, bins, range=None):
        hist, bin_edges = np.histogram(data, bins=bins, range=range)
        max_idx = np.argmax(hist)
        max_val = hist[max_idx]

        def x_of_idx(idx):
            return bin_edges[idx] + (bin_edges[idx + 1] - bin_edges[idx]) / 2

        x_of_maximum = x_of_idx(max_idx)

        return x_of_maximum

    masked_area_background_intensity = np.array([histmax(input_image[i][mask], 500, range=(0, np.max(input_image[i][mask]) / 3)) for i in range(input_image.shape[0])])
    masked_bg_sub = input_image - masked_area_background_intensity[:, np.newaxis, np.newaxis]

    plt.figure()
    plt.hist(masked_bg_sub[3][mask], bins=1000)
    plt.axvline(x=0, c='r')
    plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")
    if save_debug_path is not None:
        plt.savefig(os.path.join(args.debug_folder, '2_sub_hist.png'))

    fig, ax = plt.subplots()
    ax.imshow(masked_bg_sub[3], clim=auto_clim(masked_bg_sub[3]))
    for boundary in boundaries:
        plt.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
    plt.title("Step 2: Subtract average background intensity of the pixels inside the boundary")
    if save_debug_path is not None:
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
    if save_debug_path is not None:
        plt.savefig(os.path.join(args.debug_folder, '3_median.png'))


    threshold = threshold_mad(filtered[mask], k=4)
    print(f"{threshold=}")

    plt.figure()
    plt.hist(filtered[mask], bins=1000)
    plt.axvline(x=0, c='r')
    plt.axvline(x=threshold, c='g')
    plt.title("Step 4: Threshold (MAD method)")
    if save_debug_path is not None:
        plt.savefig(os.path.join(args.debug_folder, '4_threshold.png'))

    thr = filtered > threshold

    fig, ax = plt.subplots()
    ax.imshow(thr)
    plt.title("Step 4: Thresholded image")
    if save_debug_path is not None:
        plt.savefig(os.path.join(args.debug_folder, '4_threshold_img.png'))

    # Give unique label to each blob
    labels, num_labels = ndi.label(thr)


    # Create table of properties for all blobs
    extra_props = ('intensity_mean', 'eccentricity', 'feret_diameter_max', 'equivalent_diameter_area', 'centroid')
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

    # Remove blobs outside boundary
    props = props[(props['area_inside_mask'] > (props['area'] / 2))]

    props['elongation'] = props.feret_diameter_max / props.equivalent_diameter_area
    props['total_area'] = np.count_nonzero(mask)

    # Filter blobs
    props_filtered = props[
        (props['area'] * pixel_size * pixel_size > 2000) &
        (props['elongation'] < 2)
    ]


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
    for boundary in boundaries:
        plt.plot(boundary[:, 1], boundary[:, 0], linewidth=2, c='r');
    if reference is not None:
        ax.scatter([x for x, y in reference], [y for x, y in reference], c='r')
    plt.title("Step 5: Select regions >50% inside boundary and >min_area")
    for row in props_filtered.itertuples():
        # draw rectangle around segmented area
        minr, minc, maxr, maxc = row.bbox_0, row.bbox_1, row.bbox_2, row.bbox_3
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    if save_debug_path is not None:
        plt.savefig(os.path.join(args.debug_folder, '5_segmentation.png'))

    plt.figure()
    plt.imshow(masked_bg_sub[2], clim=auto_clim(masked_bg_sub[2]))
    plt.title("Red channel")
    if save_debug_path is not None:
        plt.savefig(os.path.join(args.debug_folder, '6_red.png'))

    props_filtered['intensity_sum'] = props_filtered.intensity_mean * props_filtered.area
    props_filtered['intensity_sum_red'] = props_filtered.intensity_mean_red * props_filtered.area

    export_columns = ['area', 'eccentricity', 'elongation', 'intensity_mean', 'intensity_sum', 'intensity_mean_red', 'intensity_sum_red', 'total_area']
    props_export = props_filtered[export_columns]
    # Convert units to um
    props_export['area'] = props_export['area'] * (pixel_size * pixel_size)
    props_export['total_area'] = props_export['total_area'] * (pixel_size * pixel_size)

    print(props_export)

    props_export.to_csv(blob_csv_path)

    if show_debug:
        def on_key_press(event):
            if event.key == 'q':
                plt.close('all')
        plt.connect('key_press_event', on_key_press)
        plt.show()




def main():
    parser = argparse.ArgumentParser(prog = 'blob_segmentation.py')
    parser.add_argument('in_filename', nargs='*')
    parser.add_argument('--only-new', action='store_true')
    parser.add_argument('--skip-fail', action='store_true')
    parser.add_argument('--show-debug', action='store_true')
    args = parser.parse_args()

    for filenames in tools.get_files(args.in_filename):
        if args.only_new and filenames.blob_csv.exists():
            continue
        print(f"Processing {filenames.bg_sub}")
        try:
            process_image(filenames.bg_sub, filenames.metadata, filenames.mask_polygon, filenames.blob_csv, args.show_debug)
        except Exception as e:
            traceback.print_exc()
            if not args.skip_fail:
                break


if __name__ == '__main__':
    main()
