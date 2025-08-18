import argparse
from pathlib import Path
from chalk.utils import put_files_into_dir

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the symlink_images command."""
    parser.description = "Create symlinks to images from one directory in another directory"
    parser.add_argument("--from-dir", required=True, help="Source directory containing images")
    parser.add_argument("--to-dir", required=True, help="Destination directory for symlinks")

def main(args):
    """Create symlinks to images from source to destination directory."""
    from_dir = Path(args.from_dir)
    to_dir = Path(args.to_dir)
    
    # Validate source directory
    if not from_dir.exists():
        print(f"Error: Source directory '{from_dir}' does not exist")
        return 1
    
    # Create destination directory if it doesn't exist
    to_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files (common formats)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(from_dir.glob(f"*{ext}"))
        image_files.extend(from_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"Warning: No image files found in '{from_dir}'")
        return 0
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    print(f"Creating symlinks for {len(image_files)} images from '{from_dir}' to '{to_dir}'...")
    
    try:
        # Use put_files_into_dir with symlink=True
        put_files_into_dir(image_files, to_dir, symlink=True)
        print(f"Successfully created {len(image_files)} symlinks in '{to_dir}'")
        return 0
    except Exception as e:
        print(f"Error creating symlinks: {e}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
