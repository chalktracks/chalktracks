import argparse
from pathlib import Path

try:
    import fiftyone as fo
    FIFTYONE_AVAILABLE = True
    FiftyOneDataset = fo.Dataset
except ImportError:
    FIFTYONE_AVAILABLE = False
    FiftyOneDataset = None


def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the view_images command."""
    parser.add_argument("image_dir", help="Directory containing images to view. Can be a simple image directory or contain images/labels/masks subdirectories.")
    parser.add_argument("--name", help="Dataset name for FiftyOne (defaults to directory name)")


def create_chalk_color_scheme():
    """
    Create a FiftyOne ColorScheme using the chalk project's segmentation class colors.
    """
    if not FIFTYONE_AVAILABLE:
        return None
    
    # Import here to avoid circular imports
    from chalk import segmentation_classes
    
    # Convert RGB tuples to hex strings
    def rgb_to_hex(rgb):
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"
    
    # Create mask target colors for each segmentation class
    mask_targets_colors = []
    for seg_class in segmentation_classes:
        mask_targets_colors.append({
            "intTarget": seg_class.index,
            "color": rgb_to_hex(seg_class.render_color)
        })
    
    # Create the color scheme
    color_scheme = fo.ColorScheme(
        color_by="value",  # Color by pixel values (indices)
        default_mask_targets_colors=mask_targets_colors,
        opacity=0.5,  # Slightly transparent for better visualization
    )
    
    return color_scheme


def detect_dataset_structure(image_dir: Path) -> dict:
    """
    Detect the structure of the dataset directory.
    Returns a dict with the dataset type and relevant paths.
    """
    image_dir = Path(image_dir)
    
    # Check if it has the structured format: images/, labels/, masks/
    images_subdir = image_dir / "images"
    labels_subdir = image_dir / "labels" 
    masks_subdir = image_dir / "masks"
    
    if images_subdir.exists() and images_subdir.is_dir():
        # Structured format
        result = {
            "type": "structured",
            "images_path": images_subdir,
            "has_labels": labels_subdir.exists() and labels_subdir.is_dir(),
            "has_masks": masks_subdir.exists() and masks_subdir.is_dir(),
        }
        
        if result["has_masks"]:
            result["masks_path"] = masks_subdir
        if result["has_labels"]:
            result["labels_path"] = labels_subdir
            
        return result
    else:
        # Simple image directory
        # Check if it contains image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if image_files:
            return {
                "type": "simple",
                "images_path": image_dir,
                "has_labels": False,
                "has_masks": False,
            }
        else:
            return {
                "type": "empty",
                "images_path": image_dir,
                "has_labels": False,
                "has_masks": False,
            }


def create_fiftyone_dataset(structure: dict, dataset_name: str):
    """
    Create a FiftyOne dataset based on the detected structure.
    """
    
    if structure["type"] == "empty":
        raise ValueError(f"No images found in {structure['images_path']}")
    
    elif structure["type"] == "simple":
        # Simple image directory - just images, no labels
        print(f"Loading simple image dataset from {structure['images_path']}")
        dataset = fo.Dataset.from_images_dir(
            images_dir=structure["images_path"],
            name=dataset_name,
        )
        
    elif structure["type"] == "structured":
        if structure["has_masks"]:
            # Image segmentation dataset with masks
            print(f"Loading image segmentation dataset from {structure['images_path']} with masks from {structure['masks_path']}")
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.ImageSegmentationDirectory,
                data_path=structure["images_path"],
                labels_path=structure["masks_path"],
                name=dataset_name,
            )
        elif structure["has_labels"]:
            # Object detection dataset with YOLO labels
            print(f"Loading YOLO dataset from {structure['images_path']} with labels from {structure['labels_path']}")
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.YOLOv5Dataset,
                data_path=structure["images_path"],
                labels_path=structure["labels_path"],
                name=dataset_name,
            )
        else:
            # Structured but no labels/masks - treat as simple
            print(f"Loading simple image dataset from {structure['images_path']} (no labels found)")
            dataset = fo.Dataset.from_images_dir(
                images_dir=structure["images_path"],
                name=dataset_name,
            )
    
    return dataset


def main(args):
    """Display images from a directory for visual inspection using FiftyOne."""
    
    if not FIFTYONE_AVAILABLE:
        print("Error: FiftyOne is not installed.")
        print("Please install it with: pip install fiftyone")
        print("Or install all requirements with: pip install -r requirements.txt")
        return
    
    image_dir = Path(args.image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {image_dir}")
    
    if not image_dir.is_dir():
        raise ValueError(f"Path is not a directory: {image_dir}")
    
    # Determine dataset name
    dataset_name = args.name if args.name else image_dir.name
    
    # Delete existing dataset if it exists
    if dataset_name in fo.list_datasets():
        print(f"Deleting existing dataset: {dataset_name}")
        fo.load_dataset(dataset_name).delete()
    
    # Detect the structure of the dataset
    structure = detect_dataset_structure(image_dir)
    print(f"Detected dataset structure: {structure['type']}")
    
    if structure["has_labels"]:
        print("  - Found labels directory")
    if structure["has_masks"]:
        print("  - Found masks directory")
    
    # Create FiftyOne dataset
    try:
        dataset = create_fiftyone_dataset(structure, dataset_name)
        print(f"Created dataset '{dataset_name}' with {len(dataset)} samples")
        
        # Apply custom color scheme if this is a structured dataset with masks
        if structure["has_masks"]:
            color_scheme = create_chalk_color_scheme()
            if color_scheme:
                dataset.app_config.color_scheme = color_scheme
                print("Applied custom chalk color scheme for segmentation masks")
        
        # Launch the FiftyOne app
        print("Launching FiftyOne app...")
        session = fo.launch_app(dataset)
        
        print("\nFiftyOne app is running!")
        print("- View your dataset in the web browser")
        print("- Press Ctrl+C in the terminal to stop the app")
        
        # Wait for user to stop the session
        session.wait()
        
    except Exception as e:
        print(f"Error creating or viewing dataset: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
