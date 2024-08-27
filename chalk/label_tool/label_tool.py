import argparse
from flask import Flask, send_from_directory, jsonify, request
import cv2
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import random
import base64


@dataclass
class DirectoryConfig:
    input_images_dir: Path
    output_images_dir: Path
    output_masks_dir: Path
    output_labels_dir: Path

app = Flask(__name__)

def mask_to_yolo_label(maskfile:Path, labelfile:Path):
    """
    Given the path to an image mask, convert it to yolo-format labels, and save to labelpath
    Refer: 
        https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format
        https://github.com/orgs/ultralytics/discussions/8528#discussioncomment-8868637
    """

    img = cv2.imread(maskfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    _, img = cv2.threshold(img, 1, 255, 0)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ## TODO
    # consider simplifying the contour
    # - not sure if this has an effect on model training (speedup?)

    def contour_to_str(contour):
        contour = contour.squeeze()
        contour = contour / [width, height] # normalise
        class_index = 0 # currently only one class: chalk
        return f"{class_index} " + " ".join([f"{x} {y}" for x,y in contour])
        
    with open(labelfile, "w") as f:
        f.write(
            "\n".join([contour_to_str(contour) for contour in contours])
        )


@app.route('/')
def index():
    return send_from_directory('static', 'label_tool.html')

@app.route('/next_image')
def next_image():
    input_images_dir:Path = app.config["directory_config"].input_images_dir
    images = [f for f in input_images_dir.iterdir() if f.is_file()]
    if not images:
        return jsonify({"error": "No images left"}), 404
    image = random.choice(images)
    
    return jsonify({"filename": image.name})

@app.route('/images/<filename>')
def get_image(filename):
    input_images_dir = app.config["directory_config"].input_images_dir
    return send_from_directory(input_images_dir, filename)

@app.route('/save_segmentation', methods=['POST'])
def save_segmentation():
    data = request.json
    image_name = Path(data['image_name'])
    print(image_name)
    segmentation_data = data['segmentation_data']
    input_image_file:Path = app.config["directory_config"].input_images_dir/image_name
    output_image_file:Path = app.config["directory_config"].output_images_dir/image_name
    output_mask_file:Path = app.config["directory_config"].output_masks_dir/ image_name
    output_label_file = app.config["directory_config"].output_labels_dir/ f"{image_name.stem}.txt"    
    
    # Decode the base64 string to binary data
    segmentation_data = base64.b64decode(segmentation_data)
    
    with open(output_mask_file, 'wb') as f:
        f.write(segmentation_data)

    # Save yolo-format label
    mask_to_yolo_label(output_mask_file, output_label_file)
    
    shutil.move(input_image_file, output_image_file)

    return jsonify({"success": True})


@app.route('/skip_image', methods=['POST'])
def skip_image():
    data = request.json
    image_name = data['image_name']
    image_path:Path = app.config["directory_config"].input_images_dir / image_name

    # skipped images are simply deleted from the input directory
    image_path.unlink()

    return jsonify({"success": True})

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    source_image_dir:Path = args.source_image_dir.expanduser()
    output_dir:Path = args.output_dir.expanduser()

    assert source_image_dir.exists() and source_image_dir.is_dir()
    
    directory_config = DirectoryConfig(
        input_images_dir=source_image_dir,
        output_images_dir=output_dir / 'images',
        output_masks_dir=output_dir / 'masks',
        output_labels_dir=output_dir / 'labels',
    )

    for dir in [
            directory_config.output_images_dir,
            directory_config.output_masks_dir,
            directory_config.output_labels_dir,
        ]:
        dir.mkdir(exist_ok=True, parents=True)

    app.config["directory_config"] = directory_config
    app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    main()
