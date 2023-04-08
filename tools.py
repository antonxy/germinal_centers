from dataclasses import dataclass
from pathlib import Path

@dataclass
class Filenames:
    czi : Path
    bg_sub : Path
    mask_polygon : Path

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
        processed_folder = processed_directory / source_file.with_suffix('')
        #bg_sub = processed_folder / Path('background_subtracted.npy')
        bg_sub = data_path / Path('background_subtracted') / source_file.with_suffix('.npy')
        mask_polygon = processed_folder / Path('mask_polygon.npy')
        yield Filenames(czi = czifile, bg_sub = bg_sub, mask_polygon = mask_polygon)

def create_dir_for_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
