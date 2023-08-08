from dataclasses import dataclass
from pathlib import Path
import numpy as np
import czifile
import json
import traceback

data_path = Path('/opt/daria_microscopy/')
source_directory = data_path / Path('input')
processed_directory = data_path / Path('processed')

# Calculates filenames for all files derived from a single czi file
class SectionFilenames:
    # source_file is a path relative to source_directory
    def __init__(self, source_file, section_nr):
        self.section_nr = section_nr

        source_file_with_section = source_file.with_name(source_file.with_suffix('').name + '_section_' + str(section_nr))
        source_subdir = source_directory / source_file_with_section
        processed_subdir = processed_directory / source_file_with_section

        self.czi = source_directory / source_file

        self.mask_polygon = source_subdir / Path('mask_polygons.npz')
        self.user_metadata = source_subdir / Path('user_metadata.json')
        self.manual_blobs = source_subdir / Path('manual_blobs.npz')

        self.bg_sub = processed_subdir / Path('background_subtracted.npy')
        self.metadata = processed_subdir / Path('metadata.json')
        self.blob_csv = processed_subdir / Path('blobs.csv')
        self.segmentation_debug = processed_subdir / Path('segmentation_debug')

def get_section_files(czifiles = None, only_selected=False):
    if czifiles == None or len(czifiles) == 0:
        source_files = [source_file.relative_to(source_directory) for source_file in source_directory.glob('**/*.czi') if not source_file.name.startswith('.')]
    else:
        source_files = [Path(f).resolve().relative_to(source_directory) for f in czifiles]

    for source_file in source_files:
        try:
            file = czifile.CziFile(source_directory / source_file)
            num_sections = czi_get_num_sections(file)
            for section in range(num_sections):
                filenames = SectionFilenames(source_file, section)
                if only_selected:
                    if filenames.user_metadata.exists():
                        with open(filenames.user_metadata, 'r') as f:
                            metadata = json.load(f)
                        if metadata['keep'] == True:
                            yield filenames
                        else:
                            print(f"File {source_file} not selected")
                    else:
                        print(f"File {source_file} has no metadata yet")
                else:
                    yield filenames
        except Exception as e:
            print(f"Failed to read file {source_file}, skipping")
            traceback.print_exc()


def create_dir_for_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)

def czi_section_shape(self, section):
    shape = [np.add(directory_entry.start, directory_entry.shape)
             for directory_entry in self.filtered_subblock_directory
             if directory_entry.start[0] == section ]
    shape = np.max(shape, axis=0)
    shape = tuple(np.subtract(shape, czi_section_start(self, section)))
    return shape

def czi_section_start(self, section):
    start = [ directory_entry.start
             for directory_entry in self.filtered_subblock_directory
             if directory_entry.start[0] == section ]
    start = tuple(np.min(start, axis=0))
    return start
    
def czi_get_num_sections(self):
    if self.axes == "SCYX0":
        return self.shape[0]
    elif self.axes == "CYX0":
        return 1
    else:
        raise RuntimeError(f"Unknown file layout: {self.axes=}")

def czi_read_section(file, section):
    if file.axes == "CYX0":
        return file.asarray()
    
    assert file.axes == "SCYX0"

    start = czi_section_start(file, section)
    shape = czi_section_shape(file, section)
    out = np.zeros(shape[1:], file.dtype)

    for directory_entry in file.filtered_subblock_directory:
        if directory_entry.start[0] == section:
            subblock = directory_entry.data_segment()
            tile = subblock.data()
            assert tile.shape[0] == 1
            index = tuple(slice(i - j, i - j + k) for i, j, k in
                          zip(directory_entry.start[1:], start[1:], tile.shape[1:]))
            try:
                out[index] = tile[0]
            except ValueError as e:
                warnings.warn(str(e))

    return out
