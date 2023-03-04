#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns

parser = argparse.ArgumentParser(prog = 'plot.py')
parser.add_argument('in_filenames', nargs='+')
args = parser.parse_args()

dfs = []

for fp in map(Path, args.in_filenames):
    folder = fp.parent.name
    file = fp.name
    data = pd.read_csv(fp)
    data['folder'] = folder
    data['file'] = file
    dfs.append(data)

data = pd.concat(dfs)
data['folder'] = data.folder.astype('category')
data['file'] = data.file.astype('category')
print(data)

plt.figure()
sns.stripplot(data=data, x='file', hue='folder', y='area')

count = data.value_counts(["folder", "file"]).to_frame(name="count").reset_index()
plt.figure()
sns.stripplot(data=count, x='folder', hue='folder', y='count')
plt.show()
