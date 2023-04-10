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

    # Simplify the polygon
    contours_simple = skimage.measure.approximate_polygon(contours[0], tolerance = 2)

    return contours_simple, largest 



def dist_point_to_segment(p, s0, s1):
    """
    Get the distance from the point *p* to the segment (*s0*, *s1*), where
    *p*, *s0*, *s1* are ``[x, y]`` arrays.
    """
    s01 = s1 - s0
    s0p = p - s0
    if (s01 == 0).all():
        return np.hypot(*s0p)
    # Project onto segment, without going past segment ends.
    p1 = s0 + np.clip((s0p @ s01) / (s01 @ s01), 0, 1) * s01
    return np.hypot(*(p - p1))


class PolygonInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = self.poly.xy[:, 0], self.poly.xy[:, 1]
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.asarray(self.poly.xy)
        if xy.shape[0] == 0:
            return None
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'a':
            self.poly.xy = np.insert(
                self.poly.xy, self.poly.xy.shape[0] - 1 if self.poly.xy.shape[0] > 1 else 0, 
                [event.xdata, event.ydata],
                axis=0)
            self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
        elif event.key == 'c':
            self.poly.xy = np.zeros((0, 2))
            self.line.set_data([], [])
        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


def process_image(in_path, out_path):
    interactors = []
    def boundary_window(image, boundary):

        poly = Polygon(boundary[:, [1, 0]], animated=True, hatch='x', fill=False)

        fig, ax = plt.subplots()
        ax.imshow(input_image[1], clim=auto_clim(input_image[1]))
        ax.add_patch(poly)
        p = PolygonInteractor(ax, poly)
        interactors.append(p)

        def on_key_press(event):
            if event.key == 'q':
                plt.close('all')
            if event.key == 'w':
                print(f"Save {out_path}")
                tools.create_dir_for_path(out_path)
                boundaries = [p.poly.xy[:, [1, 0]] for p in interactors]
                np.savez(out_path, *boundaries)
                plt.close('all')
            if event.key == 'x':
                boundary, _ = find_boundary(input_image[1])
                poly.set_xy(boundary[:, [1, 0]])
                p.line.set_data(zip(*p.poly.xy))
                p.canvas.draw_idle()
            if event.key == '+':
                boundary_window(image, np.zeros((0, 2)))
                plt.show()
            if event.key == '-':
                interactors.remove(p)
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_key_press)


    input_image = np.load(in_path).squeeze()

    boundaries = []
    if out_path.exists():
        with np.load(out_path) as data:
            for v in data.values():
                boundaries.append(v)
    else:
        boundaries.append(np.zeros((0, 2)))

    for boundary in boundaries:
        boundary_window(input_image, boundary)

    plt.show()


def main():
    parser = argparse.ArgumentParser(prog = 'draw_mask.py')
    parser.add_argument('in_filename', nargs='*')
    parser.add_argument('--only-new', action='store_true')
    args = parser.parse_args()


    for filenames in tools.get_section_files(args.in_filename):
        if args.only_new and filenames.mask_polygon.exists():
            continue
        print(f"Processing {filenames.czi} section {filenames.section_nr}")
        process_image(filenames.bg_sub, filenames.mask_polygon)


if __name__ == '__main__':
    main()
