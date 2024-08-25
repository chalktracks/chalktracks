"""
Remove approximate duplicates from dataset based on structural similarity of consecutive images
"""
import argparse
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import copy
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def main(source_image_dir, output_image_dir, visualise):

    if visualise:
        plt.ion()

    if not output_image_dir.exists():
        output_image_dir.mkdir(parents=True)

    source_images = source_image_dir.glob("*.png")

    # algorithm:
    #  given last key frame (or first frame in sequence, to initialise)
    #  iterate through images in sequence until ssim from keyframe to current image is below a threshold value.
    #  at this point, set the current image as the latest keyframe

    keyframe_files = []
    ssim_vals = []
    ssim_history_len = 50
    current_keyframe = None

    ssim_threshold = 0.5

    print("Begin filtering keyframes")
    for source_image_file in tqdm(sorted(source_images)):
        # print(source_image_file.name)
        img_rgb = cv2.imread(source_image_file, cv2.IMREAD_ANYCOLOR)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        

        if current_keyframe is None:
            # initialisation:
            # take current frame as first keyframe
            # also setup plotting
            current_keyframe = img
            keyframe_files.append(source_image_file)

            if visualise:
                plt.subplots(3,1)
                plt.subplot(3,1,1)
                img_ax = plt.imshow(img_bgr)
                plt.title("image sequence")
                plt.axis("off")
                plt.subplot(3,1,2)
                kf_ax = plt.imshow(img_bgr, cmap='gray')
                plt.title("key frames")
                plt.axis("off")
                ssim_plt_ax = plt.subplot(3,1,3)
                plt.title("ssim score (current frame vs latest keyframe)")
                ssim_plt_ax.set_ylim(bottom=0, top=1)
                ssim_plt_ax.set_xlim(left=0, right=ssim_history_len)
                plt.tight_layout() 
        else:
            ssim_to_keyframe = ssim(img, current_keyframe)
            if ssim_to_keyframe < ssim_threshold:
                current_keyframe = img
                keyframe_files.append(source_image_file)
                copy(source_image_file, output_image_dir)

                if visualise:
                    kf_ax.set_data(img_bgr)

            else:
                if visualise:
                    kf_ax.set_data(current_keyframe/2)
                    kf_ax.set_cmap("gray")

            # maintain rolling window of ssim vals for plotting
            ssim_vals.append(ssim_to_keyframe)
            ssim_vals = ssim_vals[-ssim_history_len:]

            if visualise:
                img_ax.set_data(img_bgr)
                ssim_plt_ax.clear()
                ssim_plt_ax.plot(list(range(-len(ssim_vals),0)), ssim_vals)
                plt.title("ssim score (current frame vs latest keyframe)")
                ssim_plt_ax.set_ylim(bottom=0, top=1)
                ssim_plt_ax.set_xlim(left=-ssim_history_len, right=0)
                plt.show()
                plt.pause(0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image_dir", type=Path, required=True)
    parser.add_argument("--output_image_dir", type=Path, required=True)
    parser.add_argument("--visualise", action="store_true", help="Plot keyframes and ssim metric while processing")
    args = parser.parse_args()
    main(args.source_image_dir, args.output_image_dir, args.visualise)
