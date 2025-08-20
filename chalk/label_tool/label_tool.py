import argparse
from flask import Flask, send_from_directory, jsonify, request, render_template
import cv2
from dataclasses import dataclass
import os
from pathlib import Path
import random
import base64
import io
from collections import defaultdict
from PIL import Image
import numpy as np

from chalk.util.utils import put_files_into_dir
from chalk import segmentation_classes

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# must be in agreement with classColors from label_tool.html
# class_colors = {
#     'chalk': (255, 255, 255),
#     'sign_stop': (255, 0, 0),
#     'sign_turn': (0, 0, 255),
#     'edge': (0, 255, 0)
# }
assert max(cls.index for cls in segmentation_classes) < 255, "error - only up to 255 classes supported due to storing of masks in uint8 image"


class_colors = {
    cls.name: f"rgb({cls.render_color[0]}, {cls.render_color[1]}, {cls.render_color[2]})"
    for cls in segmentation_classes
}

rgb_to_int_dict = defaultdict(
    lambda : 0, 
    {cls.render_color : cls.index for cls in segmentation_classes}
)

# def rbg_to_int(mask_rgb):
#     key = tuple(int(x) for x in mask_rgb)
#     return rgb_to_int_dict[key]

def rbg_to_int(mask_rgb):
    """
    Translates an RGB tuple to a class index by finding the closest
    matching class color within a tolerance.

    Gemini generated.

    Checks within a threshold tollerance - couldn't figure out how to stop html ui from changing the drawn colors very slightly
    """
    # The tolerance value. You can adjust this as needed.
    tolerance = 10
    
    # Ensure the input is an integer tuple for comparison
    r, g, b = (int(x) for x in mask_rgb)
    
    # Iterate through our known class colors
    for cls in segmentation_classes:
        known_r, known_g, known_b = cls.render_color
        
        # Check if each component is within the tolerance
        if (abs(r - known_r) <= tolerance and
            abs(g - known_g) <= tolerance and
            abs(b - known_b) <= tolerance):
            
            # If a match is found, return the corresponding class index
            return cls.index
            
    # If no matching color is found within the tolerance, return the default value (0)
    return 0

@dataclass
class DirectoryConfig:
    input_images_dir: Path
    output_images_dir: Path
    output_masks_dir: Path
    output_labels_dir: Path


def save_mask_file(image:np.array, maskfile:Path):
    """
    Images are sent as RGB, as rendered to user on UI
    Here we convert to a uint8 mask file (one value per class)
    and save to disk
    """
    class_img = np.apply_along_axis(rbg_to_int, axis=2, arr=image)
    cv2.imwrite(str(maskfile), class_img)

def mask_to_yolo_label(maskfile:Path, labelfile:Path):
    """
    Given the path to an image mask, convert it to yolo-format labels, and save to labelpath
    Refer: 
        https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format
        https://github.com/orgs/ultralytics/discussions/8528#discussioncomment-8868637
    """

    img = cv2.imread(str(maskfile), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    def contour_to_str(contour, class_index):
        contour = contour.squeeze()
        contour = contour / [width, height] # normalise
        return f"{class_index} " + " ".join([f"{x} {y}" for x,y in contour])
    
    contour_strings = []
    for segmentation_class in segmentation_classes:
        class_mask = (img == segmentation_class.index).astype(np.uint8)
        class_contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_strings.extend([contour_to_str(contour, segmentation_class.index) for contour in class_contours])

    ## TODO
    # consider simplifying the contour
    # - not sure if this has an effect on model training (speedup?)

    with open(labelfile, "w") as f:
        f.write("\n".join(contour_strings))


@app.route('/')
def index():
    return render_template('label_tool.html', class_colors=class_colors)


@app.route('/next_image')
def next_image():
    input_images_dir:Path = app.config["directory_config"].input_images_dir
    output_images_dir:Path = app.config["directory_config"].output_images_dir
    processed_image_names = [f.name for f in output_images_dir.iterdir() if f.is_file()]
    images_to_process = [f for f in input_images_dir.iterdir() if f.is_file() and f.name not in processed_image_names]
    if not images_to_process:
        return jsonify({"error": "No images left"}), 404
    image = random.choice(images_to_process)
    
    return jsonify({"filename": image.name})

@app.route('/images/<filename>')
def get_image(filename):
    input_images_dir = app.config["directory_config"].input_images_dir
    return send_from_directory(input_images_dir.absolute(), filename)

@app.route('/save_segmentation', methods=['POST'])
def save_segmentation():
    data = request.json
    image_name = Path(data['image_name'])
    segmentation_data = data['segmentation_data']
    input_image_file:Path = app.config["directory_config"].input_images_dir/image_name
    output_image_dir:Path = app.config["directory_config"].output_images_dir
    output_mask_file:Path = app.config["directory_config"].output_masks_dir/ image_name
    output_label_file = app.config["directory_config"].output_labels_dir/ f"{image_name.stem}.txt"    
    
    # Decode the base64 string to binary data
    segmentation_data = base64.b64decode(segmentation_data)


    image_stream = io.BytesIO(segmentation_data)

    # Open the image using PIL
    pil_image = Image.open(image_stream).convert('RGB')

    # Convert the PIL image to a NumPy array
    image = np.array(pil_image)
    
    save_mask_file(image, output_mask_file)

    # Save yolo-format label
    mask_to_yolo_label(output_mask_file, output_label_file)
    
    put_files_into_dir([input_image_file],output_image_dir, symlink=True)

    return jsonify({"success": True})


@app.route('/remove_image', methods=['POST'])
def remove_image():
    data = request.json
    image_name = data['image_name']
    image_path:Path = app.config["directory_config"].input_images_dir / image_name

    # remove image from the input directory
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
