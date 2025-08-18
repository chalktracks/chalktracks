import argparse
import os
from pathlib import Path

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the remove_empty_files command."""
    parser.description = "Find and remove all zero-size files in a directory"
    parser.add_argument("directory", help="Directory to search for empty files")

def main(args):
    """Find and remove all zero-size files in the specified directory."""
    directory = Path(args.directory)
    
    # Validate directory exists
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return 1
    
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        return 1
    
    # Find all files in the directory
    all_files = [f for f in directory.iterdir() if f.is_file()]
    
    if not all_files:
        print(f"No files found in '{directory}'")
        return 0
    
    # Find empty files (zero size)
    empty_files = [f for f in all_files if f.stat().st_size == 0]
    
    if not empty_files:
        print(f"No empty files found in '{directory}'")
        return 0
    
    print(f"Found {len(empty_files)} empty files in '{directory}':")
    for empty_file in empty_files:
        print(f"  {empty_file.name}")
    
    # Delete empty files
    deleted_count = 0
    for empty_file in empty_files:
        try:
            empty_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {empty_file.name}: {e}")
    
    print(f"Successfully deleted {deleted_count} empty files")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
