import argparse
from . import add_arg_parser, main

# __main__.py provided to make label tool module-runnable
# (note it's also runnable via the auto-generated CLI)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    exit(main(args))