import argparse
from datetime import datetime

import mlflow.onnx
from ultralytics import YOLO
import mlflow
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
import onnx
import yaml
from pathlib import Path
from subprocess import run
from typing import Any, Dict
from coolname import generate_slug
from chalk import segmentation_classes
from chalk.model.convert_model import convert_model_to_maixcam


def generate_run_name() -> str:
    """Generate a human-readable run name with format: word-word-YYYYMMDD"""
    date_str = datetime.now().strftime("%Y%m%d")
    slug = generate_slug(2)  # 2 words
    return f"{slug}-{date_str}"


# define custom dataset source for local images
# Seems like a lot of work just to point to a local directory
# But seems the only way https://github.com/mlflow/mlflow/discussions/12578
class LocalFileSystemDatasetSource(mlflow.data.filesystem_dataset_source.FileSystemDatasetSource):
    def __init__(self, path: Path):
        self.path = path
    def uri(self) -> str:
        return str(self.path)
    def _get_source_type(self) -> str:
        return "local"
    def load(self) -> str:
        return self.uri()
    def _can_resolve(self) -> bool:
        return self.path.exists()
    def _resolve(self) -> str:
        return self.uri()
    def to_dict(self) -> dict:
        return {"path": self.uri()}
    def from_from_dict(cls, source_dict: Dict[Any, Any]) -> "LocalFileSystemDatasetSource":
        return LocalFileSystemDatasetSource(Path(source_dict["path"]))

def get_dataset_version(data_yaml_path: Path) -> str:
    """Get the current dataset version using git tags."""
    return run(["git", "describe", "--tags", "--dirty"], capture_output=True, text=True, cwd=data_yaml_path.parent).stdout.strip()

# see https://github.com/ultralytics/ultralytics/blob/da9d8730f9f8a99fe6cd980fc927fd1005c15676/ultralytics%2Futils%2Fcallbacks%2Fmlflow.py#L116 
# for default mflow integration 
# # Add mlfow model registration as per https://github.com/ultralytics/ultralytics/issues/8214
def on_train_end(trainer):
    """Callback to run at end of training to log model and dataset to MLflow."""
    
    # Log dataset
    data_path = trainer.data["path"] # should log just train path?
    dataset_version = get_dataset_version(data_path)
    dataset = MetaDataset(
        source=LocalFileSystemDatasetSource(data_path), 
        name=f"chalk_dataset_{dataset_version}",                              
    )
    mlflow.log_input(dataset, context="training")

    # Export to ONNX
    model_path = f"{trainer.save_dir}/weights/best.pt"
    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=[224, 320], simplify=True) 
    
    # Log ONNX model
    onnx_model = onnx.load(onnx_path)
    mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path="model", registered_model_name="chalk_detect")

    # Convert to MaixCAM format if enabled
    convert_enabled = getattr(trainer, 'convert_to_maixcam', True)  # Default to True for backward compatibility
    if convert_enabled:
        try:
            print("\n🔄 Converting model to MaixCAM format...")
            
            # Use training data directory for calibration
            train_data_dir = Path(data_path) / "train" / "images"
            if not train_data_dir.exists():
                # Fallback to other common training data locations
                alternatives = [
                    Path(data_path) / "images" / "train",
                    Path(data_path) / "train",
                    Path(data_path) / "images"
                ]
                for alt in alternatives:
                    if alt.exists() and any(alt.glob("*.jpg")) or any(alt.glob("*.png")):
                        train_data_dir = alt
                        break
                else:
                    print("⚠️  Could not find training images for MaixCAM conversion. Skipping conversion.")
                    return
            
            # Convert the ONNX model to MaixCAM format
            run_name = Path(trainer.save_dir).name
            maixcam_model_path, mud_file_path = convert_model_to_maixcam(onnx_path, train_data_dir, run_name)

            print("✅ MaixCAM conversion completed!")
            print("Generated MaixCAM files:")
            print(f"  - {maixcam_model_path}")
            print(f"  - {mud_file_path}")
            
            # Log each MaixCAM file as an artifact to MLflow
            mlflow.log_artifact(str(maixcam_model_path), "maixcam_model")
            mlflow.log_artifact(str(mud_file_path), "maixcam_model")

        except Exception as e:
            print(f"❌ MaixCAM conversion failed: {e}")
            print("Training completed successfully, but model conversion to MaixCAM format failed.")
    else:
        print("ℹ️  MaixCAM conversion skipped (disabled via --no-convert flag)")


def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the train_model command."""
    parser.add_argument("data_yaml", help="Path to data.yaml dataset description file.")
    parser.add_argument("params_yaml", help="Path to params.yaml training params file.")
    parser.add_argument("--no-convert", action="store_true", help="Skip automatic conversion to MaixCAM format after training")

def main(args):
    """Train a segmentation model using the provided dataset."""
    print(f"Training model with data in {args.data_yaml} and params {args.params_yaml}...")

    # Generate a human-readable run name
    run_name = generate_run_name()
    print(f"🏃 Starting training run: {run_name}")
    

    model = YOLO("yolo11n-seg.pt")  # from pretrained
    
    # Store the conversion flag so the callback can access it
    def on_train_end_with_convert_flag(trainer):
        # Set the conversion flag on the trainer object
        trainer.convert_to_maixcam = not args.no_convert
        return on_train_end(trainer)
    
    model.add_callback("on_train_end", on_train_end_with_convert_flag)

    with open(args.params_yaml) as f:
        training_params = yaml.safe_load(f)

    # ignore background class for training
    classes = [c.index for c in segmentation_classes if c.name != 'background']

    results = model.train(name=run_name,data=args.data_yaml, classes=classes, **training_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)





