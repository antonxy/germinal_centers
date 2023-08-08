#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import Polygon
import sys
import argparse
import skimage
import scipy
import scipy.ndimage as ndi
import tools
import traceback
import signal

'''
Shortcuts:

a: add germinal center
scrollwheel: change threshold
d: delete germinal center
v: view germinal center

w: save changes
q: close without saving
'''

def auto_clim(img):
    return np.percentile(img, (2, 98))

class PolygonInteractor:
    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, axs, points, point_callback, view_callback):
        self.axs = axs
        canvas = axs[0].figure.canvas
        self.points = points
        self.point_callback = point_callback
        self.view_callback = view_callback

        x, y = self.points[:, 0], self.points[:, 1]
        self.lines = []
        for ax in axs:
            line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True, linestyle='')
            ax.add_line(line)
            self.lines.append(line)

        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas = canvas
        self.canvas.draw_idle()

    def on_draw(self, event):
        # self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for ax, line in zip(self.axs, self.lines):
            ax.draw_artist(line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        ax = event.inaxes
        if ax is None:
            return
        
        d = np.hypot(self.points[:, 0] - event.xdata, self.points[:, 1] - event.ydata)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.points = np.delete(self.points,
                                         ind, axis=0)
                self.update_lines()
        elif event.key == 'a':
            self.points = np.insert(
                self.points, self.points.shape[0] - 1 if self.points.shape[0] > 1 else 0, 
                [event.xdata, event.ydata],
                axis=0)
            self.update_lines()
            self.point_callback(event.xdata, event.ydata)
        elif event.key == 'v':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                x, y = self.points[ind]
                self.view_callback(x, y)
            
        for line in self.lines:
            if line.stale:
                self.canvas.draw_idle()
    
    def update_lines(self):
        for line in self.lines:
            line.set_data(self.points[:, 0], self.points[:, 1])
            
def range_around(center, size):
    return slice(max(0, center - size), center+size)
    
def crop_zeros(im):
    non_zero_0 = np.nonzero(np.sum(im != 0, axis=0))[0]
    non_zero_1 = np.nonzero(np.sum(im != 0, axis=1))[0]
    
    return (non_zero_1[0], non_zero_0[0]),im[non_zero_1[0]:non_zero_1[-1]+1, non_zero_0[0]:non_zero_0[-1]+1]
    
def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def process_image(bg_sub_path, mask_path, out_path):
    boundaries = []
    if mask_path.exists():
        with np.load(mask_path) as data:
            for v in data.values():
                boundaries.append(v)
    
    masks = {} # blob masks identified by xy coordinates of the point that created them
    if out_path.exists():
        print("Loading existing data")
        with np.load(out_path, allow_pickle=True) as data:
            npz = dict(data.items())
            
            # Put metadata and arrays back together
            masks_tmp = npz['metadata'].tolist()
            for i, el in enumerate(removekey(npz, 'metadata').values()):
                masks_tmp[i]['mask'] = el
                
            # Put in dict by key
            masks = {}
            for el in masks_tmp:
                masks[el['key']] = el
            
                
    input_image = np.load(bg_sub_path).squeeze()
    
    mask = np.any(np.stack([skimage.draw.polygon2mask(input_image[1].shape, boundary) for boundary in boundaries]), axis=0)
    
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
    
    med = np.median(masked_bg_sub)
        
    fp = skimage.morphology.disk(3)
    filtered = scipy.ndimage.median_filter(masked_bg_sub[3], footprint=fp)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax_detail1 = fig.add_subplot(2, 3, 4)
    ax_detail2 = fig.add_subplot(2, 3, 5, sharex=ax_detail1, sharey=ax_detail1)
    ax_detail3 = fig.add_subplot(2, 3, 6)
    
    threshold_fac = .5
    last_xy = (0, 0)
    
    def update_point(x, y):
        nonlocal last_xy
        x, y = int(x), int(y)
        last_xy = x, y
        
        size = 50
        xslice = range_around(x, size)
        yslice = range_around(y, size)
        x_local = x - xslice.start
        y_local = y - yslice.start
        detail_region = masked_bg_sub[:, yslice, xslice]
        filtered_region = filtered[yslice, xslice]
        
        value_at_marker = filtered[y, x]
        threshold = (value_at_marker - med) * threshold_fac
        detail_th = filtered_region > threshold
        flood = skimage.segmentation.flood(detail_th, (y_local, x_local))
        flood = ndi.binary_fill_holes(flood)
        (crop_offset_x, crop_offset_y), flood_crop = crop_zeros(flood)
        
        key = (x, y)
        masks[key] = {
            'key': key,
            'offset': (xslice.start + crop_offset_x, yslice.start + crop_offset_y),
            'mask': flood_crop,
            'threshold_fac': threshold_fac,
        }
        
        view_point(x, y)
        
    def view_point(x, y):
        nonlocal threshold_fac, last_xy
        x, y = int(x), int(y)
        last_xy = (x, y)
        el = masks[last_xy]
        offset = el['offset']
        mask = el['mask']
        threshold_fac = el['threshold_fac']
        
        size = 50
        xslice = range_around(x, size)
        yslice = range_around(y, size)
        x_local = x - xslice.start
        y_local = y - yslice.start
        detail_region = masked_bg_sub[:, yslice, xslice]
        filtered_region = filtered[yslice, xslice]
        
        local_offset_x = offset[0] - xslice.start
        local_offset_y = offset[1] - yslice.start
        
        # convert binary image to contour line
        # padded so that contour is closed if it touched the image edge
        contours = skimage.measure.find_contours(np.pad(mask, [(1, 1), (1, 1)]), 0.5)[0] - np.array([1.0, 1.0]) + np.array([local_offset_x, local_offset_y])
        # Simplify the polygon
        #contours = skimage.measure.approximate_polygon(contours, tolerance = 1)
        
        ax_detail1.clear()
        ax_detail1.imshow(filtered_region, clim=auto_clim(filtered_region))
        ax_detail1.scatter([x_local], [y_local], color='r')
        ax_detail2.clear()
        ax_detail2.imshow(detail_region[3], clim=auto_clim(detail_region[3]))
        ax_detail2.plot(contours[:, 1], contours[:, 0], 'r')
        

        ax_detail3.clear()
        ax_detail3.imshow(mask)
        ax_detail3.set_title(f"Thr: {threshold_fac:.2}")
        fig.canvas.draw_idle()
    
    ax1.imshow(input_image[2], clim=auto_clim(input_image[2]))
    for boundary in boundaries:
        ax1.plot(boundary[:, 1], boundary[:, 0], 'r')
    ax1.set_title("AF546")
    
    ax2.imshow(input_image[3], clim=auto_clim(input_image[3]))
    for boundary in boundaries:
        ax2.plot(boundary[:, 1], boundary[:, 0], 'r')
    ax2.set_title("AF647")
    
    
    # Load points
    points_to_load = []
    for k, _ in masks.items():
        x, y = k
        points_to_load.append([x, y])
    
    points = np.zeros((0, 2))
    if len(points_to_load) > 0:
        points = np.array(points_to_load)
    p = PolygonInteractor([ax1, ax2], points, update_point, view_point)
    
    def on_key_press(event):
        if event.key == 'q':
            plt.close('all')
        if event.key == 'w':
            print(f"Save {out_path}")
            tools.create_dir_for_path(out_path)
            
            # Only keep masks where the point has not been deleted
            masks_with_point = {}
            for x, y in p.points:
                masks_with_point[(x, y)] = masks[(x, y)]
                
            arrays = [el['mask'] for el in masks_with_point.values()]
            meta = np.array([removekey(el, 'mask') for el in masks_with_point.values()])
            np.savez(out_path, metadata=meta, *arrays)
            plt.close('all')
            
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    def on_scroll(event):
        nonlocal threshold_fac
        increment = 1 if event.button == 'up' else -1
        threshold_fac = np.clip(threshold_fac + increment * 0.01, 0, 1)
        update_point(*last_xy)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.show()

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    parser = argparse.ArgumentParser(prog = 'manual_blobs.py')
    parser.add_argument('in_filename', nargs='*')
    parser.add_argument('--only-new', action='store_true')
    parser.add_argument('--skip-fail', action='store_true')
    args = parser.parse_args()


    for filenames in tools.get_section_files(args.in_filename, only_selected=True):
        if not filenames.bg_sub.exists():
            print(f"Background subtraction file does not exist for {filenames.czi} section {filenames.section_nr}, skipping")
            continue
        if args.only_new and filenames.manual_blobs.exists():
            continue
        print(f"Processing {filenames.czi} section {filenames.section_nr}")
        try:
            process_image(filenames.bg_sub, filenames.mask_polygon, filenames.manual_blobs)
        except Exception as e:
            traceback.print_exc()
            if not args.skip_fail:
                break


if __name__ == '__main__':
    main()