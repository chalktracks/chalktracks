import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the check_labels command."""
    parser.add_argument("image_dir", help="Directory containing labelled images.")

def main(args):
    print(f"Checking labels in {args.image_dir}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
