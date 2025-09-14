import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
import random
import configparser
from typing import Dict
from chalk import segmentation_classes


def generate_mud_file_info(model_filename: str) -> Dict[str, Dict[str, str]]:
    """Generate mud file information as a dictionary structure."""
    # Get all class labels in order
    labels = [cls.name for cls in sorted(segmentation_classes, key=lambda x: x.index)]
    
    mud_info = {
        "basic": {
            "type": "cvimodel",
            "model": model_filename
        },
        "extra": {
            "model_type": "yolo11",
            "input_type": "rgb", 
            "type": "seg",
            "mean": "0, 0, 0",
            "scale": "0.00392156862745098, 0.00392156862745098, 0.00392156862745098",  # 1/255
            "labels": ", ".join(labels)
        }
    }
    
    return mud_info


def write_mud_file(mud_file_path: Path, mud_info: Dict[str, Dict[str, str]]) -> None:
    """Write mud file information to an INI format file."""
    config = configparser.ConfigParser()
    
    for section_name, section_data in mud_info.items():
        config.add_section(section_name)
        for key, value in section_data.items():
            config.set(section_name, key, value)
    
    with open(mud_file_path, 'w') as f:
        config.write(f)


def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the convert_model command."""
    parser.description = "Convert trained YOLO model to MaixCAM format using TPU-MLIR"
    parser.add_argument("--model-path", required=True, help="Path to the ONNX model file to convert")
    parser.add_argument("--train-data", required=True, help="Path to training data directory (used for calibration)")


def get_resource_path(filename):
    """Get path to a resource file bundled with the package."""
    # Look relative to this file for the resources
    current_dir = Path(__file__).parent
    resource_path = current_dir / "resources" / filename
    
    if resource_path.exists():
        return str(resource_path)
    else:
        raise FileNotFoundError(f"Resource file {filename} not found at {resource_path}")


def setup_workspace(model_path, train_data_dir):
    """Set up workspace directory with required files."""
    # Set random seed for consistency in test image selection
    random.seed(123)
    
    # Create temporary workspace
    workspace = Path(tempfile.mkdtemp(prefix="maixcam_convert_"))
    
    print(f"Setting up workspace at: {workspace}")
    
    # Create required directories
    data_dir = workspace / "data"
    train_images_dir = data_dir / "train_images"
    data_dir.mkdir()
    train_images_dir.mkdir()
    
    # Copy model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    shutil.copy2(model_path, data_dir / "model.onnx")
    print(f"Copied model: {model_path} -> {data_dir / 'model.onnx'}")
    
    # Copy training images for calibration
    train_data_dir = Path(train_data_dir)
    if not train_data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {train_data_dir}")
    
    # Get image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(train_data_dir.glob(f"*{ext}"))
        image_files.extend(train_data_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No image files found in training data directory: {train_data_dir}")
    
    # Use all available images for calibration
    for img_file in image_files:
        shutil.copy2(img_file, train_images_dir / img_file.name)
    
    print(f"Copied {len(image_files)} training images for calibration")
    
    # Use a random training image as test image (with consistent seed)
    test_image_path = random.choice(image_files)
    print(f"Using random training image as test image: {test_image_path.name}")
    
    shutil.copy2(test_image_path, data_dir / "test_image.png")
    print(f"Copied test image: {test_image_path} -> {data_dir / 'test_image.png'}")
    
    # Copy conversion script
    script_path = get_resource_path("tpuc_dev_convert.sh")
    shutil.copy2(script_path, workspace / "tpuc_dev_convert.sh")
    
    # Make script executable
    (workspace / "tpuc_dev_convert.sh").chmod(0o755)
    
    return workspace


def run_subprocess_with_live_output(cmd, cwd=None):
    """Run subprocess with live output streaming to terminal."""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd, 
            cwd=cwd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output line by line
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())  # Print without extra newline
                output_lines.append(line)
        
        # Wait for process to complete and get final return code
        return_code = process.wait()
        
    except Exception as e:
        raise RuntimeError(f"Failed to run command {' '.join(cmd)}: {e}")
    
    # Return a result-like object for compatibility
    class ProcessResult:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""  # We combined stderr with stdout
    
    return ProcessResult(return_code, ''.join(output_lines))


def build_docker_image(workspace):
    """Build the Docker image for TPU conversion."""
    print("Building Docker image...")
    
    # Copy Dockerfile to workspace
    dockerfile_path = get_resource_path("Dockerfile")
    shutil.copy2(dockerfile_path, workspace / "Dockerfile")
    
    # Build Docker image
    cmd = ["docker", "build", ".", "-t", "tpuc_dev_workspace"]
    result = run_subprocess_with_live_output(cmd, cwd=workspace)
    
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed with return code {result.returncode}")
    
    print("Docker image built successfully")


def run_conversion(workspace):
    """Run the conversion process in Docker."""
    print("Running model conversion...")
    
    # Get current user ID and group ID to avoid permission issues
    import os
    uid = os.getuid()
    gid = os.getgid()
    
    # Run Docker container with current user to avoid permission issues
    cmd = [
        "docker", "run", "--rm", 
        "--user", f"{uid}:{gid}",
        "-v", f"{workspace}:/workspace/",
        "tpuc_dev_workspace", 
        "/workspace/tpuc_dev_convert.sh"
    ]
    
    result = run_subprocess_with_live_output(cmd)
    
    if result.returncode != 0:
        raise RuntimeError(f"Model conversion failed with return code {result.returncode}")
    
    print("Model conversion completed successfully")


def copy_results(workspace, model_path, model_name=None):
    """Copy conversion results to output directory and generate mud file."""
    # Use same directory as the input model
    output_dir = Path(model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for generated model files
    model_file = list(workspace.glob("*.cvimodel")) 
    
    if not model_file:
        raise RuntimeError(f"No converted model files found matching *.cvimodel")
    model_file = model_file[0]
    
    # Determine the final model filename
    if model_name:
        final_model_name = f"{model_name}.cvimodel"
    else:
        final_model_name = model_file.name
    
    dest_file = output_dir / final_model_name
    shutil.copy2(model_file, dest_file)
    print(f"Copied: {dest_file}")
    
    # Generate the mud file for the model
    mud_file_path = dest_file.with_suffix('.mud')
    mud_info = generate_mud_file_info(final_model_name)
    write_mud_file(mud_file_path, mud_info)
    print(f"Generated mud file: {mud_file_path}")
    
    return dest_file, mud_file_path


def convert_model_to_maixcam(model_path, train_data_dir, model_name=None):
    """
    Convert trained model to MaixCAM format.
    
    Args:
        model_path: Path to the ONNX model file to convert
        train_data_dir: Path to training data directory (used for calibration)
        model_name: Optional name for the output model files (without extension).
                   If not provided, uses the original filename from the conversion.
        
    Returns:
        Tuple of (Path to the converted model file, Path to the mud file).
        
    Raises:
        FileNotFoundError: If model or training data not found
        RuntimeError: If conversion process fails
    """
    model_path = Path(model_path)
    train_data_dir = Path(train_data_dir)
    
    print(f"Converting model: {model_path}")
    print(f"Using training data: {train_data_dir}")
    
    workspace = None
    try:
        # Set up workspace
        workspace = setup_workspace(
            model_path, 
            train_data_dir
        )
        
        # Build Docker image
        build_docker_image(workspace)
        
        # Run conversion
        run_conversion(workspace)
        
        # Copy results
        converted_model_file, mud_file = copy_results(workspace, model_path, model_name)
        
        print("\n✅ Model conversion completed successfully!")
        print("Generated files:")
        print(f"  - {converted_model_file}")
        print(f"  - {mud_file}")

        return converted_model_file, mud_file

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise
        
    finally:
        # Clean up workspace
        if workspace and workspace.exists():
            try:
                shutil.rmtree(workspace)
                print(f"Cleaned up workspace: {workspace}")
            except PermissionError as e:
                print(f"Warning: Permission error during cleanup: {e}")
                print(f"You may need to manually remove: sudo rm -rf {workspace}")
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
                print(f"You may need to manually remove: sudo rm -rf {workspace}")


def main(args):
    """CLI entry point for convert_model command."""
    try:
        output_files = convert_model_to_maixcam(args.model_path, args.train_data)
        return 0
    except Exception as e:
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    exit(main(args))
