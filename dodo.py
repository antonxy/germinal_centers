from pathlib import Path
from doit.tools import config_changed

data_path = Path('/opt/daria_microscopy/')
source_directory = data_path / Path('Emile')
source_files = [source_file.relative_to(source_directory) for source_file in source_directory.glob('**/*.czi') if not source_file.name.startswith('.')]

DOIT_CONFIG = {'default_tasks': ['segmentation']}

def dir_creator(path):
    def create_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    return create_dir

bg_sub_config = {
    'options': [
        '-d', '16'
    ]
}

def task_bg_sub_and_downscale():
    output_directory = data_path / Path('background_subtracted')
    py_file = Path('./mosaic_background_subtract.py')

    for source_file in source_files:
        input_file = source_directory / source_file
        encoded_file = output_directory / source_file.with_suffix('.npy')
        metadata_file = output_directory / source_file.with_suffix('.json')
        yield {
            'name': source_file,
            'actions': [
                dir_creator(encoded_file),
                ['python3', py_file, input_file, encoded_file] + bg_sub_config['options']
            ],
            'file_dep': [py_file, input_file],
            'targets': [encoded_file, metadata_file],
            'uptodate': [config_changed(bg_sub_config)],
        }

segmentation_config = {
}

def task_segmentation():
    interm_directory = data_path / Path('background_subtracted')
    output_directory = data_path / Path('segmentation')
    py_file = Path('./blob_segmentation.py')

    for source_file in source_files:
        interm_file = interm_directory / source_file.with_suffix('.npy')
        metadata_file = interm_directory / source_file.with_suffix('.json')
        csv_file = output_directory / source_file.with_suffix('.csv')
        debug_folder = output_directory / source_file.with_suffix('')
        yield {
            'name': source_file,
            'actions': [
                dir_creator(csv_file),
                ['python3', py_file, interm_file, '-o', csv_file, '-d', debug_folder]
            ],
            'file_dep': [py_file, interm_file, metadata_file],
            'targets': [csv_file],
            'uptodate': [config_changed(segmentation_config)],
        }
