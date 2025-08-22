# Training Workflow

Process for using chalktracks tooling to label images and train chalk detection model.

General idea:
* data lives in a repo tracked by dvc
* use another dir for training models
* use chalktracks tool to preprocess and label data, as well as training model


Setup workspace for training workflow (uses [uv](https://docs.astral.sh/uv/) for python env):

```
mkdir chalk_workspace
cd chalk_workspace/
mkdir data
mkdir training
uv venv
uv pip install git+https://github.com/chalktracks/chalktracks.git
. .venv/bin/activate
```

## Processing steps: initialise data directory with dvc tracking

```
cd data
git init
dvc init
git commit -m "dvc init"
mkdir -p data/sequences
mkdir data/combined_dataset
dvc add data
git add data.dvc .gitignore
git commit -m "dvc add empty data dir"
```

## Processing steps: add new sequence

The following commands import images from `$IMPORT_DIR` and store in `$SEQUENCE_DIR`. 

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

Commit to dvc
```
dvc commit
git add data.dvc
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
dvc commit
git add data.dvc
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
dvc commit
git add data.dvc
git commit -m "completed labelling for ${SEQUENCE_DIR}"
```


## Processing steps: assemble combined dataset

Combine all sequences into a final dataset with test/train/val splits:
```
chalk combine_sequences --sequence-dir ./data/sequences/ --combined-dir ./data/combined_dataset/```

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
