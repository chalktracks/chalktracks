import argparse
import shutil
from pathlib import Path

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the add_sequence command."""
    parser.description = """Copy raw captured images (e.g., from camera) into a dataset sequence directory 
    as the first step in creating and preprocessing an image sequence. Creates a new subdirectory 
    under the sequence directory to organize the images by sequence name."""
    parser.add_argument("--image-dir", required=True, help="Source directory containing raw images to copy")
    parser.add_argument("--sequence-dir", required=True, help="Base directory where image sequences are stored")
    parser.add_argument("--seq-name", required=True, help="Name for the new sequence (creates subdirectory)")

def main(args):
    """Copy raw images into a new sequence directory for dataset preprocessing."""
    image_dir = Path(args.image_dir)
    sequence_dir = Path(args.sequence_dir)
    seq_name = args.seq_name
    
    # Validate input directory
    if not image_dir.exists():
        print(f"Error: Image directory '{image_dir}' does not exist")
        return 1
    
    # Validate sequence directory
    if not sequence_dir.exists():
        print(f"Error: Sequence directory '{sequence_dir}' does not exist")
        return 1
    
    # Create sequence directory structure
    seq_output_dir = sequence_dir / seq_name / "0_raw_images"
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files (common formats)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"Warning: No image files found in '{image_dir}'")
        return 0
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    print(f"Copying {len(image_files)} images from '{image_dir}' to '{seq_output_dir}'...")
    
    # Copy images to the new sequence directory
    copied_count = 0
    for image_file in image_files:
        try:
            dest_file = seq_output_dir / image_file.name
            shutil.copy2(image_file, dest_file)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {image_file.name}: {e}")
    
    print(f"Successfully copied {copied_count} images to sequence '{seq_name}'")
    print(f"Sequence created at: {seq_output_dir.parent}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
