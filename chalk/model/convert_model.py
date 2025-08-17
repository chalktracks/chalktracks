import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the convert_model command."""
    parser.add_argument("input_dir", help="Directory containing model to convert.")
    parser.add_argument("output_dir", help="Directory to save converted model.")

def main(args):
    print(f"Converting model from {args.input_dir} to {args.output_dir}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
