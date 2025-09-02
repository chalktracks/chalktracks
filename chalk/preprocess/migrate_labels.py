import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import re

from chalk.preprocess.label_tool.label_tool import save_mask_file

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the migrate_labels command."""
    parser.add_argument("data_dir", help="Directory containing data to migrate labels.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes.")

def migrate_mask_file(mask_path: Path, dry_run: bool = False):
    """
    Convert RGB mask to integer mask where:
    - Background (black/near-black) -> 0  
    - Chalk (white/near-white) -> 1
    
    Optimized version using vectorized NumPy operations instead of 
    pixel-by-pixel processing for 100-1000x speed improvement.
    """
    print(f"Processing mask: {mask_path}")
    
    # Read the RGB mask
    rgb_mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if rgb_mask is None:
        print(f"  Warning: Could not read {mask_path}")
        return False
    
    # Convert BGR to RGB (OpenCV uses BGR)
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
    
    if not dry_run:
        # Optimized conversion: create integer mask directly using vectorized operations
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
        
        # Create integer mask: pixels above threshold get value 1 (chalk), below get 0 (background)
        class_img = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        class_img[gray > (255./3/2)] = 1  # Any pixel above threshold is assigned int value "1"
        
        # Save the integer mask directly
        cv2.imwrite(str(mask_path), class_img)
        print(f"  ✓ Converted RGB mask to integer mask")
    else:
        print(f"  [DRY RUN] Would convert RGB mask to integer mask")
    
    return True

def migrate_label_file(label_path: Path, dry_run: bool = False):
    """
    Convert YOLO labels from class index 0 to class index 1 for chalk.
    
    This updates the class indices in YOLO segmentation format where
    each line contains: class_id x1 y1 x2 y2 x3 y3 ...
    """
    print(f"Processing label: {label_path}")
    
    if not label_path.exists():
        print(f"  Warning: Label file does not exist: {label_path}")
        return False
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"  Warning: Empty label file: {label_path}")
            return False
        
        # Convert class index from 0 to 1
        updated_lines = []
        changes_made = False
        
        for line in lines:
            line = line.strip()
            if not line:
                updated_lines.append(line)
                continue
                
            # YOLO format: class_id x1 y1 x2 y2 ... (for segmentation)
            parts = line.split()
            if len(parts) < 3:  # Need at least class_id and one coordinate pair
                updated_lines.append(line)
                continue
            
            class_id = parts[0]
            if class_id == "0":
                # Change class index from 0 to 1 for chalk
                parts[0] = "1"
                changes_made = True
                print(f"  ✓ Changed class index 0 -> 1")
            
            updated_lines.append(" ".join(parts))
        
        if changes_made:
            if not dry_run:
                with open(label_path, 'w') as f:
                    f.write("\n".join(updated_lines))
                print(f"  ✓ Updated label file")
            else:
                print(f"  [DRY RUN] Would update label file")
        else:
            print(f"  No changes needed (no class 0 found)")
            
    except Exception as e:
        print(f"  Error processing {label_path}: {e}")
        return False
    
    return True

def find_labelled_sequences(data_dir: Path):
    """
    Find all directories that contain labelled data.
    Look for directories with masks/ and labels/ subdirectories.
    """
    labelled_dirs = []
    
    for item in data_dir.rglob("*"):
        if item.is_dir():
            masks_dir = item / "masks"
            labels_dir = item / "labels"
            
            if masks_dir.exists() and labels_dir.exists():
                # Check if there are actually files in these directories
                mask_files = list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.jpeg"))
                label_files = list(labels_dir.glob("*.txt"))
                
                if mask_files or label_files:
                    labelled_dirs.append(item)
    
    return labelled_dirs

def main(args):
    """
    Migrate labelled data from old format to new format.
    
    This script performs two main conversions:
    1. Convert RGB mask images to integer masks (background=0, chalk=1)
    2. Update YOLO label files to use class index 1 instead of 0 for chalk
    
    The script searches for directories containing 'masks/' and 'labels/' 
    subdirectories and processes all mask and label files within them.
    """
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    print(f"Migrating labels in {data_dir}...")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    # Find all directories with labelled data
    labelled_dirs = find_labelled_sequences(data_dir)
    
    if not labelled_dirs:
        print("No labelled sequences found!")
        print("Looking for directories containing both 'masks/' and 'labels/' subdirectories...")
        return
    
    print(f"Found {len(labelled_dirs)} labelled sequence(s):")
    for seq_dir in labelled_dirs:
        print(f"  - {seq_dir}")
    
    print()
    
    total_masks_processed = 0
    total_labels_processed = 0
    
    # Process each labelled sequence
    for seq_dir in labelled_dirs:
        print(f"Processing sequence: {seq_dir}")
        
        masks_dir = seq_dir / "masks"
        labels_dir = seq_dir / "labels"
        
        # Process mask files
        if masks_dir.exists():
            mask_files = list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.jpeg"))
            for mask_file in mask_files:
                if migrate_mask_file(mask_file, args.dry_run):
                    total_masks_processed += 1
        
        # Process label files
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            for label_file in label_files:
                if migrate_label_file(label_file, args.dry_run):
                    total_labels_processed += 1
        
        print()
    
    print(f"Migration complete!")
    print(f"  Masks processed: {total_masks_processed}")
    print(f"  Labels processed: {total_labels_processed}")
    if args.dry_run:
        print("  (No files were actually modified - this was a dry run)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate chalk labelling data from old format to new format")
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
