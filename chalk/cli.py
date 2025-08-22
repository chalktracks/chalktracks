import argparse
import ast
import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
import chalk


def has_required_functions(module_path):
    """
    Check if a Python file has 'main' and 'add_arg_parser' functions without importing it.
    Uses AST parsing which is much faster than importing.
    """
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file into an AST
        tree = ast.parse(content)
        
        # Look for function definitions
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.add(node.name)
        
        # Check if both required functions are present
        return 'main' in functions and 'add_arg_parser' in functions
        
    except Exception:
        # If we can't parse the file, assume it doesn't have the functions
        return False


def get_function_docstring(module_path, function_name):
    """
    Extract a function's docstring without importing the module.
    """
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef) and 
                node.name == function_name and 
                ast.get_docstring(node)):
                docstring = ast.get_docstring(node)
                return docstring.strip().split('\n')[0]  # First line only
                
    except Exception:
        pass
    
    return None


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
            
            # Get the actual file path
            spec = importlib.util.find_spec(modname)
            if spec is None or spec.origin is None:
                continue
                
            module_path = Path(spec.origin)
            
            # Skip if it's not a .py file
            if module_path.suffix != '.py':
                continue
            
            # Check if it has the required functions (without importing)
            if has_required_functions(module_path):
                # Extract command name from module path
                command_name = modname.split('.')[-1]
                
                # Get the module category (preprocess, model, util)
                parts = modname.split('.')
                if len(parts) >= 3:
                    category = parts[1]  # preprocess, model, util
                    
                    # Get docstring without importing
                    docstring = get_function_docstring(module_path, 'main')
                    
                    commands[command_name] = {
                        'module_name': modname,
                        'module_path': module_path,
                        'category': category,
                        'docstring': docstring or f"{command_name} command"
                    }
                    
        except Exception as e:
            # Skip modules that have errors
            continue
    
    return commands


def lazy_import_module(module_name):
    """Import a module only when needed."""
    return importlib.import_module(module_name)


def print_custom_help(commands):
    """Print custom help with category grouping."""
    print("usage: chalk [-h] COMMAND ...")
    print()
    print("Chalk: A command-line tool for chalk line following dataset preparation and model training.")
    print()
    print("positional arguments:")
    print("  COMMAND             Available commands (organized by category)")
    
    # Group commands by category
    categories = {}
    for cmd_name, cmd_info in commands.items():
        category = cmd_info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((cmd_name, cmd_info))
    
    # Print each category
    for category in sorted(categories.keys()):
        print(f"\n    [{category}]")
        for cmd_name, cmd_info in sorted(categories[category]):
            description = cmd_info['docstring']
            print(f"        {cmd_name:<18} {description}")
    
    print()
    print("options:")
    print("  -h, --help          show this help message and exit")
    print()
    print("Use 'chalk COMMAND -h' for detailed help on a specific command.")


def main():
    """Main CLI entry point with automatic command discovery."""
    
    # Discover all available commands (fast, no imports)
    commands = discover_commands()
    
    if not commands:
        print("No commands found!")
        return 1

    # Check if help is requested
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h']):
            print_custom_help(commands)
            return 0

    parser = argparse.ArgumentParser(
        description="Chalk: A command-line tool for chalk line following dataset preparation and model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # We'll handle help ourselves
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
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
    
    # Create subparsers for each command (lazy import only when needed)
    for category in sorted(categories.keys()):
        for cmd_name, cmd_info in sorted(categories[category]):
            
            # Create subparser for this command
            cmd_parser = subparsers.add_parser(
                cmd_name,
                help=cmd_info['docstring'],
                formatter_class=argparse.RawDescriptionHelpFormatter
            )
            
            # Store the module name for lazy loading
            cmd_parser.set_defaults(
                module_name=cmd_info['module_name'],
                command_name=cmd_name
            )
    
    # Parse arguments (first pass to get the command)
    args, remaining = parser.parse_known_args()
    
    # Now lazy import only the module we actually need
    try:
        module = lazy_import_module(args.module_name)
        
        # Create a new parser for this specific command
        specific_parser = argparse.ArgumentParser(
            prog=f"chalk {args.command_name}",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Configure the parser for this specific command
        module.add_arg_parser(specific_parser)
        
        # Parse all arguments with the command-specific parser
        final_args = specific_parser.parse_args(remaining)
        
        # Execute the command
        result = module.main(final_args)
        return result if result is not None else 0
        
    except Exception as e:
        print(f"Error executing {args.command_name}: {e}")
        return 1
if __name__ == "__main__":
    exit(main())
