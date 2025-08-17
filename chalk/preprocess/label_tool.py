import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the label_tool command."""
    parser.add_argument("image_dir", help="Directory containing images to label.")

def main(args):
    print(f"Launching label tool for {args.image_dir}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
