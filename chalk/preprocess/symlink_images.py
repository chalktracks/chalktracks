import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the symlink_images command."""
    parser.add_argument("--from-dir", required=True, help="Source directory for images.")
    parser.add_argument("--to-dir", required=True, help="Destination directory for symlinks.")

def main(args):
    print(f"Symlinking images from {args.from_dir} to {args.to_dir}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
