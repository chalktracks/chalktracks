"""Label tool module for interactive image annotation."""

import argparse
from pathlib import Path
from .label_tool import DirectoryConfig, app

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the label_tool command."""
    parser.description = "Launch the interactive labeling tool for annotating images with chalk line labels"
    parser.add_argument("source_image_dir", help="Directory containing source images to label")
    parser.add_argument("output_dir", help="Output directory where images (symlinks), masks, and labels will be stored")

def main(args):
    """Launch the interactive labeling tool for annotating images."""
    
    # Parse arguments
    source_image_dir = Path(args.source_image_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    
    # Validate source directory
    if not source_image_dir.exists():
        print(f"Error: Source image directory '{source_image_dir}' does not exist")
        return 1
    
    if not source_image_dir.is_dir():
        print(f"Error: '{source_image_dir}' is not a directory")
        return 1
    
    # Set up output directory structure
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    labels_dir = output_dir / 'labels'
    
    # Validate that required output directories exist
    for dir_name, dir_path in [('images', images_dir), ('masks', masks_dir), ('labels', labels_dir)]:
        if not dir_path.exists():
            print(f"Error: Required output directory '{dir_path}' does not exist")
            print(f"Please create the {dir_name} directory before running the label tool")
            return 1
        if not dir_path.is_dir():
            print(f"Error: '{dir_path}' exists but is not a directory")
            return 1
    
    directory_config = DirectoryConfig(
        input_images_dir=source_image_dir,
        output_images_dir=images_dir,
        output_masks_dir=masks_dir,
        output_labels_dir=labels_dir,
    )

    # Configure Flask app and run
    app.config["directory_config"] = directory_config
    print(f"Starting label tool for images in: {source_image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images (symlinks): {images_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Labels: {labels_dir}")
    print("Open your browser to http://localhost:5000 to start labeling")
    
    try:
        app.run(debug=True, host='0.0.0.0')
        return 0
    except KeyboardInterrupt:
        print("\nLabel tool stopped by user")
        return 0
    except Exception as e:
        print(f"Error running label tool: {e}")
        return 1