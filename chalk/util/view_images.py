import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the view_images command."""
    parser.add_argument("image_dir", help="Directory containing images to view.")

def main(args):
    """Display images from a directory for visual inspection."""
    print(f"Viewing images in {args.image_dir}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
