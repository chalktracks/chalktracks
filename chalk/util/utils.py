from pathlib import Path
from shutil import copy


def put_files_into_dir(files:list[Path], dest_dir:Path, symlink:bool=True, name_prefix:str=""):
    """
    Put given files into the destination directory.
    If symlink = True, symlinks are made, otherwise files are copied
    If name_prefix is provided, it will be prepended to the filename
    """

    def copy_file(file_path:Path, target_name:str):
        copy(file_path, dest_dir / target_name)
    def link_file(file_path:Path, target_name:str):
        target_path = dest_dir / target_name
        file_relative_path = file_path.absolute().relative_to(dest_dir.absolute(), walk_up=True)
        target_path.symlink_to(file_relative_path)
    
    put_file = link_file if symlink else copy_file

    for file_path in files:
        target_name = f"{name_prefix}_{file_path.name}" if name_prefix else file_path.name
        put_file(file_path, target_name)