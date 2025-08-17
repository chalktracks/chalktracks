import argparse

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the migrate_labels command."""
    parser.add_argument("data_dir", help="Directory containing data to migrate labels.")

def main(args):
    print(f"Migrating labels in {args.data_dir}...")
    # Your processing logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
