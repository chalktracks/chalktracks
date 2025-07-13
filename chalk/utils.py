from pathlib import Path
from shutil import copy


def put_files_into_dir(files:list[Path], dest_dir:Path, symlink:bool=True):
    """
    Put given files into the destination directory.
    If symlink = True, symlinks are made, otherwise files are copied
    """

    def copy_file(file_path:Path):
        copy(file_path, dest_dir)
    def link_file(file_path:Path):
        target_path = dest_dir/file_path.name
        file_relative_path = file_path.absolute().relative_to(dest_dir.absolute(), walk_up=True)
        target_path.symlink_to(file_relative_path)
    
    put_file = link_file if symlink else copy_file

    for file_path in files:
        put_file(file_path)