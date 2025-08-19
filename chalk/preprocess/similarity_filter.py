import argparse
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the similarity_filter command."""
    parser.description = """Filter images based on structural similarity to reduce dataset size by removing
    near-duplicate frames. Uses SSIM to compare consecutive images and removes non-keyframes in place."""
    parser.add_argument("--ssim-threshold", type=float, required=True, help="Reject frames with SSIM value higher than this threshold (0.0-1.0)")
    parser.add_argument("--image-dir", required=True, help="Directory containing images to filter")
    parser.add_argument("--visualise", action="store_true", help="Plot keyframes and SSIM metric while processing")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually deleting files")

def main(args):
    """Filter images based on structural similarity threshold."""
    source_image_dir = Path(args.image_dir)
    ssim_threshold = args.ssim_threshold
    visualise = args.visualise
    dry_run = args.dry_run

    if visualise:
        plt.ion()

    if not source_image_dir.exists():
        print(f"Error: Source directory '{source_image_dir}' does not exist")
        return 1

    # Find image files (common formats)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    source_images = []
    for ext in image_extensions:
        source_images.extend(source_image_dir.glob(f"*{ext}"))
        source_images.extend(source_image_dir.glob(f"*{ext.upper()}"))

    if not source_images:
        print(f"Warning: No image files found in '{source_image_dir}'")
        return 0

    # Algorithm:
    # Given last key frame (or first frame in sequence, to initialise)
    # iterate through images in sequence until ssim from keyframe to current image is below a threshold value.
    # At this point, set the current image as the latest keyframe

    keyframe_paths = []
    ssim_vals = []
    ssim_history_len = 50
    current_keyframe = None

    print(f"Begin filtering keyframes with SSIM threshold {ssim_threshold}")
    for source_image_file in tqdm(sorted(source_images)):
        try:
            img_rgb = cv2.imread(str(source_image_file), cv2.IMREAD_COLOR)
            if img_rgb is None:
                print(f"Warning: Could not read image {source_image_file}")
                continue
                
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            print(f"Error processing {source_image_file}: {e}")
            continue

        if current_keyframe is None:
            # Initialisation: take current frame as first keyframe
            current_keyframe = img
            keyframe_paths.append(source_image_file)

            if visualise:
                plt.subplots(3, 1)
                plt.subplot(3, 1, 1)
                img_ax = plt.imshow(img_bgr)
                plt.title("image sequence")
                plt.axis("off")
                plt.subplot(3, 1, 2)
                kf_ax = plt.imshow(img_bgr, cmap='gray')
                plt.title("key frames")
                plt.axis("off")
                ssim_plt_ax = plt.subplot(3, 1, 3)
                plt.title("ssim score (current frame vs latest keyframe)")
                ssim_plt_ax.set_ylim(bottom=0, top=1)
                ssim_plt_ax.set_xlim(left=0, right=ssim_history_len)
                plt.tight_layout()
        else:
            ssim_to_keyframe = ssim(img, current_keyframe)
            if ssim_to_keyframe < ssim_threshold:
                current_keyframe = img
                keyframe_paths.append(source_image_file)

                if visualise:
                    kf_ax.set_data(img_bgr)
            else:
                if visualise:
                    kf_ax.set_data(current_keyframe/2)
                    kf_ax.set_cmap("gray")

            # Maintain rolling window of ssim vals for plotting
            ssim_vals.append(ssim_to_keyframe)
            ssim_vals = ssim_vals[-ssim_history_len:]

            if visualise:
                img_ax.set_data(img_bgr)
                ssim_plt_ax.clear()
                ssim_plt_ax.plot(list(range(-len(ssim_vals), 0)), ssim_vals)
                plt.title("ssim score (current frame vs latest keyframe)")
                ssim_plt_ax.set_ylim(bottom=0, top=1)
                ssim_plt_ax.set_xlim(left=-ssim_history_len, right=0)
                plt.show()
                plt.pause(0.001)

    print(f"Filtered {len(source_images)} images down to {len(keyframe_paths)} keyframes")
    
    # Delete non-keyframe images (or show what would be deleted in dry-run mode)
    images_to_remove = [img for img in source_images if img not in keyframe_paths]
    
    if dry_run:
        print(f"DRY RUN: Would remove {len(images_to_remove)} images:")
        for img_path in images_to_remove:
            print(f"  {img_path.name}")
        print(f"Would keep {len(keyframe_paths)} keyframe images")
    else:
        if images_to_remove:
            print(f"Removing {len(images_to_remove)} non-keyframe images...")
            removed_count = 0
            for img_path in images_to_remove:
                try:
                    img_path.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {img_path.name}: {e}")
            print(f"Successfully removed {removed_count} images, kept {len(keyframe_paths)} keyframes")
        else:
            print("No images to remove - all images are keyframes")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
