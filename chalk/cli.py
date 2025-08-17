import argparse
import importlib
import pkgutil
from pathlib import Path
import chalk


def discover_commands():
    """Recursively discover all modules with 'main' and 'add_arg_parser' functions."""
    commands = {}
    
    # Get the chalk package path
    chalk_path = Path(chalk.__file__).parent
    
    # Walk through all modules in the chalk package
    for importer, modname, ispkg in pkgutil.walk_packages([str(chalk_path)], prefix="chalk."):
        try:
            # Skip __init__, cli, and __main__ modules
            if (modname.endswith(".__init__") or 
                modname.endswith(".cli") or 
                modname.endswith(".__main__")):
                continue
                
            # Import the module
            module = importlib.import_module(modname)
            
            # Check if it has both required functions
            if hasattr(module, 'main') and hasattr(module, 'add_arg_parser'):
                # Extract command name from module path
                # e.g., "chalk.preprocess.add_sequence" -> "add_sequence"
                command_name = modname.split('.')[-1]
                
                # Get the module category (preprocess, model, util)
                parts = modname.split('.')
                if len(parts) >= 3:
                    category = parts[1]  # preprocess, model, util
                    commands[command_name] = {
                        'module': module,
                        'category': category,
                        'full_name': modname
                    }
                    
        except ImportError as e:
            # Skip modules that can't be imported
            continue
        except Exception as e:
            # Skip modules that have other errors
            continue
    
    return commands


def main():
    """Main CLI entry point with automatic command discovery."""
    
    # Discover all available commands
    commands = discover_commands()
    
    if not commands:
        print("No commands found!")
        return 1
    
    parser = argparse.ArgumentParser(
        description="Chalk: A command-line tool for chalk line following dataset preparation and model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands (organized by category)",
        required=True,
        metavar="COMMAND"
    )
    
    # Group commands by category for better organization
    categories = {}
    for cmd_name, cmd_info in commands.items():
        category = cmd_info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((cmd_name, cmd_info))
    
    # Create subparsers for each command
    for category in sorted(categories.keys()):
        for cmd_name, cmd_info in sorted(categories[category]):
            module = cmd_info['module']
            
            # Get help text from module docstring if available
            help_text = f"[{category}] "
            if hasattr(module, '__doc__') and module.__doc__:
                help_text += module.__doc__.strip().split('\n')[0]
            else:
                help_text += f"{cmd_name} command"
            
            # Create subparser for this command
            cmd_parser = subparsers.add_parser(
                cmd_name,
                help=help_text,
                formatter_class=argparse.RawDescriptionHelpFormatter
            )
            
            # Use the module's add_arg_parser function to configure arguments
            module.add_arg_parser(cmd_parser)
            
            # Set the function to call when this command is selected
            cmd_parser.set_defaults(func=module.main)
    
    # Parse arguments and execute the selected command
    args = parser.parse_args()
    
    # Call the selected command's main function
    try:
        result = args.func(args)
        return result if result is not None else 0
    except Exception as e:
        print(f"Error executing {args.command}: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
