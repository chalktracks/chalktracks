import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import random
import yaml

from chalk.util.utils import put_files_into_dir
from chalk import segmentation_classes

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the combine_sequences command."""
    parser.add_argument("--sequence-dir", required=True, help="Directory containing sequences.")
    parser.add_argument("--combined-dir", required=True, help="Directory to store combined sequences.")


def move_images_and_labels(image_paths:list[Path], target_dir:Path, sequence_name:str):
    """Move images to the target directory with sequence name prefix, also moving their corresponding labels."""
    label_paths = [p.parent / ".." / "labels" / p.name.replace(".png", ".txt") for p in image_paths]
    mask_paths = [p.parent / ".." / "masks" / p.name for p in image_paths]

    put_files_into_dir(image_paths, target_dir / "images", symlink=True, name_prefix=sequence_name)
    put_files_into_dir(label_paths, target_dir / "labels", symlink=True, name_prefix=sequence_name)
    put_files_into_dir(mask_paths, target_dir / "masks", symlink=True, name_prefix=sequence_name)

def main(args):
    """Combine multiple image sequences into a single dataset."""

    combined_dir = Path(args.combined_dir)

    # confirm combined_dir is empty
    if any(combined_dir.iterdir()):
        print(f"Error: Combined directory '{combined_dir}' is not empty. Please delete existing contents or choose another directory.")
        return
    
    # get sequence directories to combine
    # should be subdirectories of sequence_dir, which in turn have subdirectory "2_labelled"
    sequence_dirs = [d for d in Path(args.sequence_dir).iterdir() if d.is_dir() and (d / "2_labelled").exists()]
    if not sequence_dirs:
        print("No valid sequence directories found.")
        return
    
    print(f"Combining {len(sequence_dirs)} sequence(s):")
    for seq_dir in sequence_dirs:
        print(f"  {seq_dir.name}")
    
    # get list of all images with their sequence names
    all_images_with_sequences = []
    for seq_dir in sequence_dirs:
        image_dir = seq_dir / "2_labelled" / "images"
        images = list(image_dir.glob("*.png"))
        # Store tuples of (image_path, sequence_name)
        for image in images:
            all_images_with_sequences.append((image, seq_dir.name))
    
    # split into train, validation, and test sets
    # for now, hardcode train,test,val portions
    split_ratios = (0.7, 0.15, 0.15) # train, test, val
    assert math.isclose(sum(split_ratios), 1.0), "split ratios must add to 1.0"
    p_train, p_test, p_val = split_ratios

    random.seed(123)
    random.shuffle(all_images_with_sequences) # just to be sure we're randomly sampling

    images_train, images_test_and_val = train_test_split(all_images_with_sequences, train_size=p_train)
    images_test, images_val = train_test_split(images_test_and_val, train_size=p_test/(p_test+p_val))

    n_images = len(all_images_with_sequences)
    n_train = len(images_train)
    n_test = len(images_test)
    n_val = len(images_val)

    train_dir = combined_dir / "train"
    test_dir = combined_dir / "test"
    val_dir = combined_dir / "val"

    # Group images by sequence for each split
    def group_by_sequence(image_sequence_tuples):
        sequence_groups = {}
        for image_path, sequence_name in image_sequence_tuples:
            if sequence_name not in sequence_groups:
                sequence_groups[sequence_name] = []
            sequence_groups[sequence_name].append(image_path)
        return sequence_groups

    for directory, images_with_sequences in [
        (train_dir, images_train),
        (test_dir, images_test),
        (val_dir, images_val)]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "images").mkdir(exist_ok=True)
        (directory / "labels").mkdir(exist_ok=True)
        (directory / "masks").mkdir(exist_ok=True)
        
        # Group by sequence and process each sequence separately
        sequence_groups = group_by_sequence(images_with_sequences)
        for sequence_name, image_paths in sequence_groups.items():
            move_images_and_labels(image_paths, directory, sequence_name)

    # create data.yaml
    description_dict = {
        "path" : str(combined_dir.absolute()),
        "train" : "./train/images",
        "val" : "./val/images",
        "test" : "./test/images",
        "names" : {cls.index : cls.name for cls in segmentation_classes}
    }
    description_path = combined_dir / "data.yaml"
    with open(description_path, 'w') as f:
        yaml.dump(description_dict, f)
    
    print("Finished combining sequences")
    print(f"""
dataset split ({n_images} images)
train:\t{n_train}\t({n_train/n_images:.1%})
test: \t{n_test}\t({n_test/n_images:.1%})
val:  \t{n_val}\t({n_val/n_images:.1%})
""")
    print(f"Combined dataset saved to {combined_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
