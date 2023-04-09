from dataclasses import dataclass
from pathlib import Path

@dataclass
class Filenames:
    czi : Path
    bg_sub : Path
    metadata : Path
    mask_polygon : Path
    selection : Path
    blob_csv: Path

def get_files(czifiles = None):
    data_path = Path('/opt/daria_microscopy/')
    source_directory = data_path / Path('Emile')
    processed_directory = data_path / Path('processed')

    if czifiles == None or len(czifiles) == 0:
        source_files = [source_file.relative_to(source_directory) for source_file in source_directory.glob('**/*.czi') if not source_file.name.startswith('.')]
    else:
        source_files = [Path(f).relative_to(source_directory) for f in czifiles]

    for source_file in source_files:
        czifile = source_directory / source_file
        source_subdir = source_directory / source_file.with_suffix('')
        processed_subdir = processed_directory / source_file.with_suffix('')

        mask_polygon = processed_subdir / Path('mask_polygons.npz')
        selection = processed_subdir / Path('selection.json')

        bg_sub = processed_subdir / Path('background_subtracted.npy')
        metadata = processed_subdir / Path('metadata.json')
        blob_csv = processed_subdir / Path('blobs.csv')
        yield Filenames(czi = czifile, bg_sub = bg_sub, metadata = metadata, mask_polygon = mask_polygon, selection = selection, blob_csv = blob_csv)

def create_dir_for_path(path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
