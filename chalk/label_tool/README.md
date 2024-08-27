# Label tool

A web interface for labelling images containing chalk lines.

https://github.com/user-attachments/assets/525ca4f9-dd9f-43bc-a460-a5646528fd3e

**Video:** demo of the workflow for manually labelling chalk lines with label tool.

Consists of an html page [label_tool.html](./static/label_tool.html) served by a simple flask app [label_tool.py](./label_tool.py).

The user can quickly draw chalk line labels directly on top of the source image with a touch screen interface.

## Workflow

1) Copy the images to be labelled into a source directory. I use `~/tmp/label_tool/source_images` as a working directory for this task.

1) Create a directory to use as the output directory. Here I'll use `~/tmp/label_tool/outputs`.

1) Run label tool specifying input and output directories:

    ```python -m chalk.label_tool --source_image_dir ~/tmp/label_tool/source_images --output_dir ~/tmp/label_tool/outputs```

    Note that images are removed from the input directory as they are processed. The labelling is complete when there are no images remaining in the input directory.

1) Navigate to the app in your browser.

    The script will print the url used for the webserver (e.g. `http://192.168.1.251:5000`). Navigate to this url on the device you want to use for labelling. I use a touch screen device with a stylus. 

1) Perform labelling. 

    The app will display a random image from the dataset. Draw over the chalk line using the mouse, or touch if using a touchscreen, then click "next".

    If a mistake is made drawing over the line, it can be undone with the "undo" button.

    If you want to exclude an image from the training dataset, click "skip" to move to the next image.

    Continue labelling images until all have been labelled, or you have enough images to train on. 

    <ctrl-c> to kill the app.

1) Save outputs. The labelling results will be saved in the following directory structure:

    ```
    label_tool/
    └── outputs
        ├── images
        ├── masks
        └── labels
    ```
    
    Here the `images` directory contains the original source images which were labelled, the `masks` directory contains chalk segmentation masks for each image in `images`, and `labels` contains the same mask represented in the [ultralytics yolo format](https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format).



