#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns

parser = argparse.ArgumentParser(prog = 'plot.py')
parser.add_argument('in_filenames', type=pathlib.Path, nargs='+')
args = parser.parse_args()

dfs = []

for fp in args.in_filenames:
    folder = fp.parent.name
    file = fp.name
    data = pd.read_csv(fp)
    data['mouse'] = folder
    data['cutting'] = file
    data['line'] = folder.split(' ')[0]
    dfs.append(data)

data = pd.concat(dfs)

plt.figure()
sns.stripplot(data=data, x='cutting', hue='mouse', y='area')

plt.figure()
sns.stripplot(data=data, x='mouse', hue='line', y='area')

plt.figure()
sns.stripplot(data=data, x='line', hue='line', y='area')

# Checks that all entries in the array are the same and returns the value
def same(data):
    val = data[0] if data.size > 0 else np.nan
    assert np.all(data == val)
    return val

# Add a count column that we can sum using groupby
data['count'] = 1
# First group by cutting since total_area should not be summed within one cutting (is there a more elegant way to do store total_area in the input table?)
grouped_by_file = data.groupby(by=["line", "mouse", "cutting"]).agg({"area": 'sum', "count": 'sum', "total_area": same})
# Then group by mouse. Now total_area should be summed over all cuttings for the mouse.
grouped_by_mouse = grouped_by_file.groupby(by=["line", "mouse"]).agg({"area": 'sum', "count": 'sum', "total_area": 'sum'})


# Plot portion of area covered by germinal centers for each mouse
grouped_by_mouse['area_covered'] = grouped_by_mouse['area'] / grouped_by_mouse['total_area']
plt.figure()
sns.stripplot(data=grouped_by_mouse, x='mouse', hue='line', y='area_covered', size=10)
plt.ylabel("Fraction of area covered")

# Plot number of germinal centers per total area for each mouse
grouped_by_mouse['count_per_area'] = grouped_by_mouse['count'] / grouped_by_mouse['total_area'] * (1000 * 1000)
plt.figure()
sns.stripplot(data=grouped_by_mouse, x='mouse', hue='line', y='count_per_area', size=10)
plt.ylabel("Count per square mm")



plt.show()
