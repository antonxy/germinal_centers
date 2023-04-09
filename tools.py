from dataclasses import dataclass
from pathlib import Path

data_path = Path('/opt/daria_microscopy/')
source_directory = data_path / Path('Emile')
processed_directory = data_path / Path('processed')

# Calculates filenames for all files derived from a single czi file
class Filenames:
    # source_file is a path relative to source_directory
    def __init__(self, source_file):
        source_subdir = source_directory / source_file.with_suffix('')
        processed_subdir = processed_directory / source_file.with_suffix('')

        self.czi = source_directory / source_file
        self.bg_sub = processed_subdir / Path('background_subtracted.npy')
        self.metadata = processed_subdir / Path('metadata.json')
        self.mask_polygon = processed_subdir / Path('mask_polygons.npz')
        self.selection = processed_subdir / Path('selection.json')
        self.blob_csv = processed_subdir / Path('blobs.csv')
        self.segmentation_debug = processed_subdir / Path('segmentation_debug')

def get_files(czifiles = None):
    if czifiles == None or len(czifiles) == 0:
        source_files = [source_file.relative_to(source_directory) for source_file in source_directory.glob('**/*.czi') if not source_file.name.startswith('.')]
    else:
        source_files = [Path(f).relative_to(source_directory) for f in czifiles]

    for source_file in source_files:
        yield Filenames(source_file)


def create_dir_for_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
