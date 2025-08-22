import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import random
from chalk.util.utils import put_files_into_dir

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the combine_sequences command."""
    parser.add_argument("--sequence-dir", required=True, help="Directory containing sequences.")
    parser.add_argument("--combined-dir", required=True, help="Directory to store combined sequences.")


def move_images_and_labels(image_paths:list[Path], target_dir:Path):
    """Move images to the target directory, also moving their corresponding labels."""
    label_paths = [p.parent / ".." / "labels" / p.name.replace(".png", ".txt") for p in image_paths]
    mask_paths = [p.parent / ".." / "masks" / p.name for p in image_paths]

    put_files_into_dir(image_paths, target_dir / "images", symlink=True)
    put_files_into_dir(label_paths, target_dir / "labels", symlink=True)
    put_files_into_dir(mask_paths, target_dir / "masks", symlink=True)

def main(args):
    """Combine multiple image sequences into a single dataset."""

    print("Begin")

    
    # get sequence directories to combine
    # should be subdirectories of sequence_dir, which in turn have subdirectory "2_labelled"
    sequence_dirs = [d for d in Path(args.sequence_dir).iterdir() if d.is_dir() and (d / "2_labelled").exists()]
    if not sequence_dirs:
        print("No valid sequence directories found.")
        return
    
    print(f"Combining {len(sequence_dirs)} sequence(s):")
    for seq_dir in sequence_dirs:
        print(f"  {seq_dir.name}")
    
    # get list of all images
    all_images = []
    for seq_dir in sequence_dirs:
        image_dir = seq_dir / "2_labelled" / "images"
        images = list(image_dir.glob("*.png"))
        all_images.extend(images)
    
    # split into train, validation, and test sets
    # for now, hardcode train,test,val portions
    split_ratios = (0.7, 0.15, 0.15) # train, test, val
    assert math.isclose(sum(split_ratios), 1.0), "split ratios must add to 1.0"
    p_train, p_test, p_val = split_ratios

    random.seed(123)
    random.shuffle(all_images) # just to be sure we're randomly sampling

    images_train, images_test_and_val = train_test_split(images, train_size=p_train)
    images_test, images_val = train_test_split(images_test_and_val, train_size=p_test/(p_test+p_val))

    n_images = len(all_images)
    n_train = len(images_train)
    n_test = len(images_test)
    n_val = len(images_val)

    train_dir = Path(args.combined_dir) / "train"
    test_dir = Path(args.combined_dir) / "test"
    val_dir = Path(args.combined_dir) / "val"

    for directory, images in [
        (train_dir, images_train),
        (test_dir, images_test),
        (val_dir, images_val)]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "images").mkdir(exist_ok=True)
        (directory / "labels").mkdir(exist_ok=True)
        (directory / "masks").mkdir(exist_ok=True)
        move_images_and_labels(images, directory)


    print("Finished combining sequences")
    print(f"""
dataset split ({n_images} images)
train:\t{n_train}\t({n_train/n_images:.1%})
test: \t{n_test}\t({n_test/n_images:.1%})
val:  \t{n_val}\t({n_val/n_images:.1%})
""")
    print(f"Combined dataset saved to {args.combined_dir}")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
