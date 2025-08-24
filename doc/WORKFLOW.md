# Training Workflow

Process for using chalktracks tooling to label images and train chalk detection model.

General idea:
* keep data and training outputs within a workspace
* data is tracked by dvc/git, can be versioned with git and be backed up using dvc 
* another dir usesd for training models
* chalktracks repo used as workflow tooling, does not need to be cloned, just pip-installed

Expected workspace top-level structure:
```
chalk_workspace/
├── data       # training dataset tracked with dvc
└── training   # workspace for running and tracking model training runs
```

The rest of this document provides the steps to build a dataset and train a model.

## Prepare workspace

Setup a workspace for training workflow and install the chalktracks cli tool `chalk` (using [uv](https://docs.astral.sh/uv/) for python env):

```
mkdir chalk_workspace
cd chalk_workspace/
mkdir data
mkdir training
uv venv
uv pip install git+https://github.com/chalktracks/chalktracks.git
. .venv/bin/activate
```

## Build the dataset

### Initialise data directory with dvc tracking

```
cd data
git init
dvc init
dvc config cache.type symlink
git commit -m "dvc init"
mkdir sequences
mkdir combined_dataset
dvc add combined_dataset
git add combined_dataset.dvc .gitignore
git commit -m "dvc add empty data dir"
```

Note on `dvc config cache.type symlink` - before using DVC, I set up the workflow to symlink images across process stage directories to eliminate copies. Then I was confused when DVC got rid of my symlinks after `dvc commit`. AFAIU, it was replacing them with hardlinks. This should be fine, but confused me, so I set cache type to symlink as above, so I could still see my links. Probably I could get rid of all this symlinking and let dvc manage/avoid duplication, will leave it as-is for now. 

## Processing steps: add new sequence

The following commands are run from the `workspace/data` directory.

Images are imported from `$IMPORT_DIR` and stored in `$SEQUENCE_DIR`. 

E.g. to create a new sequence "my_new_sequence" using images from a camera, set:
```
IMPORT_DIR=/media/my_camera
SEQUENCE_DIR=./sequences/my_new_sequence
```

### Import

```
chalk add_sequence --image-dir ${IMPORT_DIR} --sequence-dir ${SEQUENCE_DIR}
```

Stores input images under `${SEQUENCE_DIR}/0_raw_images`. Creates sequence directory structure (eg for name `sequence_0`):
```
sequence_0
├── 0_raw_images
├── 1_keyframes
└── 2_labelled
    ├── images
    ├── labels
    └── masks
```

Add new sequence to dvc
```
dvc add ${SEQUENCE_DIR}
git add -u
git commit -m "Add ${SEQUENCE_DIR} raw images"
```


### Filter

Note the images are stored in `0_raw_images` and symlinked into `1_keyframes` and `2_labelled/images` as required, to avoid copies.


Manual/ad-hoc filtering:
First link files into keyframe dir:
```
chalk symlink_images --from-dir ${SEQUENCE_DIR}/0_raw_images --to-dir ${SEQUENCE_DIR}/1_keyframes
```

Then run filtering scripts as appropriate, potentially including:
```
chalk remove_empty_files ${SEQUENCE_DIR}/1_keyframes/
```

```
chalk similarity_filter --ssim-threshold 0.5 --image-dir ${SEQUENCE_DIR}/1_keyframes/ --dry-run --visualise
```
(remove `--dry-run` flag and rerun when you are happy with the filter results)

And/or, manually view images (eg in gthumb or file browser) and delete non-useful files

When filtering complete, add to dvc
```
dvc add ${SEQUENCE_DIR}
git add -u
git commit -m "completed keyframing for ${SEQUENCE_DIR}"
```

### Label

Run label tool
```
chalk label_tool ${SEQUENCE_DIR}/1_keyframes/  ${SEQUENCE_DIR}/2_labelled/
```
Label tool runs a UI to assist labelling of images. Will load images from given input directory, and save images (as symlinks), masks and labels to given output directory.

To inspect the labelled sequence;
```
chalk view_images ${SEQUENCE_DIR}/2_labelled/
```


Save labels 
```
dvc add ${SEQUENCE_DIR}
git add -u
git commit -m "completed labelling for ${SEQUENCE_DIR}"
```


## Processing steps: assemble combined dataset

Combine all sequences into a final dataset with test/train/val splits:
```
chalk combine_sequences --sequence-dir ./sequences/ --combined-dir ./combined_dataset/
```

Save combined dataset 
```
dvc commit
git add -u
git commit -m "built dataset from existing sequences"
```

TODO
* generate default config
* train
 * check how to track input dataset from dvc in mlfow
* script to run mlflow?


## Other scripts

```
python -m chalk.migrations update_for_sign_labels data_dir
```
Recursively finds and updates:
    * masks from RGB to int image
    * label txt from chalk = 0 to current label convention.

TODO:
* implement the scripts above
* combine into single cli?
* Test


Scratch space - thinking about chalk repo, what scripts will exist?


preprocess:
    add_sequence --image-dir /path/to/images --sequence-dir /path/to/sequence/storage --seq-name 20250817_my_new_sequence
    symlink_images --from-dir sequence_dir/sequence_name/0_raw_images --to-dir sequence_dir/sequence_name/1_keyframes
    similarity_filter --ssim-threshold 0.5 --image-dir sequence_dir/sequence_name/1_keyframes
    label_tool sequence_dir/sequence_name/2_labelled
    combine_sequences --sequence-dir /path/to/sequence/storage --combined-dir /path/to/combined/storage
    check_labels /path/to/labelled/images
    migrate_labels /path/to/data/dir

model:
    train_model /path/to/data /path/to/params.yaml
    convert_model /path/to/input/dir /path/to/output/dir

util:
    view_images /path/to/image/dir
