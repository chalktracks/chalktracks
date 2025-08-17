import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the train_model command."""
    parser.add_argument("data_dir", help="Directory containing training data.")
    parser.add_argument("params_yaml", help="Path to params.yaml file.")

def main(args):
    print(f"Training model with data in {args.data_dir} and params {args.params_yaml}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
