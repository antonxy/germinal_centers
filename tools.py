from dataclasses import dataclass
from pathlib import Path

@dataclass
class Filenames:
    czi : Path
    bg_sub : Path
    metadata : Path
    mask_polygon : Path
    selection : Path

def get_files(czifiles = None):
    data_path = Path('/opt/daria_microscopy/')
    source_directory = data_path / Path('Emile')
    processed_directory = data_path / Path('processed')

    if czifiles == None or len(czifiles) == 0:
        #source_files = [source_file.relative_to(source_directory) for source_file in source_directory.glob('**/*.czi') if not source_file.name.startswith('.')]
        with open(data_path / Path("good_images.txt"), "r") as f:
            source_files = [Path(l.strip()) for l in f.readlines()]

        with open(data_path / Path("doable_images.txt"), "r") as f:
            source_files += [Path(l.strip()) for l in f.readlines()]
    else:
        source_files = [Path(f).relative_to(source_directory) for f in czifiles]

    for source_file in source_files:
        czifile = source_directory / source_file
        mask_polygon = source_directory / Path('mask_polygon.npy')
        selection = source_directory / Path('selection.json')

        processed_subdir = processed_directory / source_file.with_suffix('')
        bg_sub = processed_subdir / Path('background_subtracted.npy')
        metadata = processed_subdir / Path('metadata.json')
        yield Filenames(czi = czifile, bg_sub = bg_sub, metadata = metadata, mask_polygon = mask_polygon, selection = selection)

def create_dir_for_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
