"""
Assemble a dataset from multiple sequences
"""
from pathlib import Path
from sequence import Sequence, LabelledImage
import random
import math
import itertools
import yaml
import attrs
from sklearn.model_selection import train_test_split
from chalk import segmentation_classes
import fiftyone as fo

@attrs.define
class DatasetSplit:
    train : list[LabelledImage]
    test : list[LabelledImage]
    val : list[LabelledImage]

    def __str__(self):
        n_train = len(self.train)
        n_test = len(self.test)
        n_val = len(self.val)
        n_images = sum([n_train, n_test, n_val])

        return f"""
dataset split ({n_images} images)
train:\t{n_train}\t({n_train/n_images:.1%})
test: \t{n_test}\t({n_test/n_images:.1%})
val:  \t{n_val}\t({n_val/n_images:.1%})
"""


def combine_sequences(sequences:list[Sequence], sample_mode:str="all") -> list[LabelledImage]:
    """
    Given some input sequences, get a list of labelled images.
    Intended to support a few options for mixing sequences - use all, sample evenly, user specified ratios etc
    """
    sequence_images = [s.get_labelled_images() for s in sequences]

    if sample_mode == "even":

        # even sampling, same number of images from all sequences
        shortest_sequence_len = min([len(s) for s in sequence_images])
        n_to_sample = shortest_sequence_len

        selected_images = []

        for images in sequence_images:
            selected_images.append(random.sample(images, n_to_sample))
    
    elif sample_mode == "all":
        selected_images = sequence_images
    else:
        raise ValueError(f"unrecognised {sample_mode=}")

    # flatten to single list with all images
    selected_images = list(itertools.chain.from_iterable(selected_images))

    # return shuffled
    random.shuffle(selected_images)
    
    return selected_images

def split_images(images:list[LabelledImage]) -> DatasetSplit:
    
    # for now, hardcode train,test,val portions
    split_ratios = (0.7, 0.15, 0.15) # train, test, val
    assert math.isclose(sum(split_ratios), 1.0), "split ratios must add to 1.0"
    p_train, p_test, p_val = split_ratios

    random.shuffle(images) # just to be sure

    images_train, images_test_and_val = train_test_split(images, train_size=p_train)
    images_test, images_val = train_test_split(images_test_and_val, train_size=p_test/(p_test+p_val))

    return DatasetSplit(
        train=images_train,
        test=images_test,
        val=images_val,
    )

def process_sequences_into_combined_dataset(sequences:list[Sequence]):

    all_images = combine_sequences(sequences)

    print(f"combined dataset contains {len(all_images)} images")

    split = split_images(all_images)
    print(split)

    # move images into output dir
    dataset_dir = Path("data/combined_dataset")
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    val_dir = dataset_dir / "val"

    for path in [
        train_dir,
        test_dir,
        val_dir
    ]:
        path.mkdir(exist_ok=True, parents=True)

    for images, dest in [
        (split.train, train_dir),
        (split.test, test_dir),
        (split.val, val_dir)
    ]:

        for image in images:
            image.put_in_dir(dest)
        
    # create data.yaml
    description_dict = {
        "path" : str(dataset_dir.absolute()),
        "train" : "./train/images",
        "val" : "./val/images",
        "test" : "./test/images",
        "names" : {cls.index : cls.name for cls in segmentation_classes}
    }
    description_path = dataset_dir / "data.yaml"
    with open(description_path, 'w') as f:
        yaml.dump(description_dict, f)
    
    return dataset_dir

def view_train_set(dataset_dir:Path):


    name = dataset_dir.name

    # delete existing dataset with same name
    if name in fo.list_datasets():
        fo.load_dataset(name).delete()

    # Import dataset by explicitly providing paths to the source media and masks

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageSegmentationDirectory,
        data_path=dataset_dir   / "train" / "images",
        labels_path=dataset_dir / "train" / "masks",
        name=name,
    )

    # Launch the FiftyOne app
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":

    # list of input sequences  
    sequences = [
        # Sequence(Path("data/sequences/sequence_chippenham_20240707")),
        # Sequence(Path("data/sequences/sequence_quick_test_dataset_20250712/")),
        # Sequence(Path("data/sequences/sequence_reagan_rd_driveway_bright_sun_20250617/")),
        # Sequence(Path("data/sequences/sequence_try_get_blue_working_better_20250717/")),
        # Sequence(Path("data/sequences/sequence_brain_picnic_failed_red_detections_20250719/")),
        # Sequence(Path("data/sequences/sequence_round_the_house_20250720/")),
        # Sequence(Path("data/sequences/sequence_more_brick_20250722/")),
        Sequence(Path("data/sequences/sequence_driveway_stop_turn_signs_20250803/")),

    ]
    dataset_dir = process_sequences_into_combined_dataset(sequences)

    view_train_set(dataset_dir)


    
