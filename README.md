# chalktracks

![Header image - children playing with dump trucks](doc/header_img.webp)

Building a toy dump truck that can follow lines drawn in chalk.

## About

🚂 Toy trains are fun, but your imagination is limited by how much track you have.

🖍️ What if you could draw the tracks with chalk? 

Introducing 🎉 Chalk Tracks! 🎉

🚚 This project aims to build a toy dump truck that follows tracks drawn in chalk. It will drive where you draw!

We're building this in order to:
* Explore a concept for a new toy
* Develop an ML-based portfolio project
* Keep the kids entertained!

The evisaged system consists of a motorised toy dump truck with a forward-facing camera fitted, running a segmentation model for chalk line detection. The bulk of the work will involve building the platform, and developing the chalk line segmentation model.

<img src="doc/20240627_172255.jpg" width="400"/> <img src="doc/20240707_152512.jpg" width="400"/> 

**Figures:** First prototype truck, with camera taped on. 

<br/>

https://github.com/user-attachments/assets/9ad01d09-88ea-4e09-aedb-314ebd87270b

**Video:** First prototype chalk segmentation, video captured on a hand-held smartphone, segmentation model trained on [roboflow](https://roboflow.com/).

### Tech Choices

#### Camera

This project will use the [Sipeed MaixCAM edge AI camera](https://wiki.sipeed.com/hardware/en/maixcam/index.html) for detecting chalk lines.

<img src="doc/Sipeed-MaixCAM-02.jpg" width="300"/>

This was chosen based on:
* **Cost** - this is a very low cost device for running segmentation models onboard. 
* **Capability** - despite the cost, the device appears sufficiently performant to run segmentation at a sufficient rate (1TOPS INT8 NPU)
* **Support** - The device appears sufficiently supported with docs, examples and a community forum, to get up and running without too much difficulty. 

#### Segmentation Model

The choice of model to use for chalk segmentation was primarily based on the choice of camera. The MaixCam docs clearly outline the process for deploying [**YoloV8 instance segmentation**](https://docs.ultralytics.com/tasks/segment/), and further Ultralytics provide extensive tooling and documentation supporting the training of these models, hence this is chosen as the model to use for the task.

<img src="doc/yolo-seg-demo.png" width="500"/>

#### MLOps

The project makes use of 

<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" width="32" height="32" /> [Fiftyone](https://docs.voxel51.com/) (for dataset visualization) </br>
<img src="https://upload.wikimedia.org/wikipedia/commons/a/af/Data_Version_Control._Official_Logo_by_Iterative.ai.png" width="32" height="32" />  [DVC](https://dvc.org/) (for dataset versioning) </br>
<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="32" height="32" />  [MLflow](https://mlflow.org/) (for experiment tracking) </br>


## Segmentation Model Training Workflow

**WIP**

This section describes the workflow for training the segmentation model. 

Code will be added and steps will be documented as the project is built out.

1) Hardware setup

    Prepare the truck with motors and camera fitted.

    Wiki todo: add a few photos of building first prototype truck, and notes about the choice of camera

1) Data collection

    Drive the truck over a variety of chalk lines, while recording from the camera.
    1) Copy data collection script to camera, set to autostart
    1) draw chalk lines in the test environment
    1) drive truck around on top of the lines while recording camera images
    1) remove SD card from camera, copy images to dataset directory

1) Save sample videos

    For later demonstration of the segmentation, set aside some video sequences as desired

    TODO: ffmpeg command

1) [Key framing](https://github.com/chalktracks/chalktracks/wiki/Keyframing)

    To reduce labelling workload, filter to a keyframe sequence where images are removed if they are too similar to the previous keyframe.

    `chalk similarity_filter --ssim-threshold 0.8 --image-dir data/sequences/sequence_0/1_keyframes/`

    ![keyframe example](doc/keyframes.png)

1) Anonymisation 

    Where the dataset contains images of people, blur their faces. Note, no imagery of people will be captured/shared without consent.
    1) Run script to read raw images and blur faces
    2) *manual step*: review output images, manually delete any that failed to blur.

1) [Labelling](https://github.com/chalktracks/chalktracks/wiki/Label-Tool)

    Manually annotate the dataset with chalk line lables. Here a simple [labelling tool](https://github.com/chalktracks/chalktracks/wiki/Label-Tool) has been developed to annotate the chalk lines with a touch screen interface. After annotation, labels must be converted to appropriate format for training.

    https://github.com/user-attachments/assets/9c7b821c-f266-4acc-9cbd-2b938dd8f299

    **Video:** Demo showing the process of labelling chalk lines with the label tool.

    <img src="doc/labelled_images.webp"/>

    **Figure:** a subset of images labelled with the labelling tool.


1) Split into train/test/validation sets

1) Merge with existing datasets

1) Train the model

1) Convert model for target architecture

1) Deploy/test

1) Visualise?
    51 scripts to view labels and model outputs?

The following directory structure is created through the process of collecting raw images up to training the model:
```
dataset_name/
├── 0_raw_images
├── 1_keyframes
├── 2_anonymised
├── 3_labelled
│   ├── images
│   ├── labels
│   └── masks
└── 4_split
    ├── test
    │   ├── images
    │   ├── labels
    │   └── masks
    ├── train
    │   ├── images
    │   └── labels
    │   └── masks
    └── val
        ├── images
        └── labels
    │   └── masks

```



## CLI Auto-Discovery

The project includes a dynamic CLI that automatically discovers commands from any module with `main(args)` and `add_arg_parser(parser)` functions. Commands are organized by module location (preprocess, model, util) and can be run via:

```bash
chalk <command> [args]          # After pip install
python -m chalk.cli <command>   # Direct module usage
```

To add a new command, simply create a module in the appropriate category with the required functions - it will automatically appear in `chalk --help`.
See [chalk/cli.py](./chalk/cli.py) for implementation.


## TODO


* script to run model on a directory of images and make a video
---

**Note:** As of August 2025 I've started using GitHub Copilot for this project. Code added after this date may be heavily AI-generated. 🤖