import argparse
import yaml
from pathlib import Path


def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the get_default_training_config command."""
    parser.description = "Generate a default training configuration file (config.yaml) in the current directory"


def get_default_config():
    """Return the default training configuration."""
    return {
        # Model settings
        'model': 'yolo11n-seg.pt',  # Base model (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
        
        # Training hyperparameters
        # see https://docs.ultralytics.com/modes/train/#train-settings
        'epochs': 100,
        'time': None, 
        'imgsz': 320,
        'batch': 16,
        'lr0': 0.01,
        'patience': 100,
        'save_period': -1,
        'workers': 8,
        'device': None,  # auto-detect GPU/CPU
        'amp': True,   # Automatic Mixed Precision
        
        # Data augmentation
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'fliplr': 0.0,  # messes up left/right sign classification
    }


def main(args):
    """Generate a default training configuration file."""
    config_path = Path.cwd() / "config.yaml"
    
    # Check if file already exists
    if config_path.exists():
        print(f"Error: config.yaml already exists in {config_path.parent}")
        print("Please remove or rename the existing file first.")
        return 1
    
    # Get default configuration
    config = get_default_config()
    
    # Write config to file
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        print(f"✅ Created default training configuration: {config_path}")
        print()
        print("You can now:")
        print("1. Edit config.yaml to customize training parameters")
        print("2. Use it with: chalk train_model path/to/data.yaml config.yaml")
        
        return 0
        
    except Exception as e:
        print(f"Error writing config file: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
