import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the combine_sequences command."""
    parser.add_argument("--sequence-dir", required=True, help="Directory containing sequences.")
    parser.add_argument("--combined-dir", required=True, help="Directory to store combined sequences.")

def main(args):
    """Combine multiple image sequences into a single dataset."""
    print(f"Combining sequences from {args.sequence_dir} into {args.combined_dir}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
