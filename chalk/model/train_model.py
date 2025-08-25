import argparse

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
from chalk import segmentation_classes


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

def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the train_model command."""
    parser.add_argument("data_yaml", help="Path to data.yaml dataset description file.")
    parser.add_argument("params_yaml", help="Path to params.yaml training params file.")

def main(args):
    """Train a segmentation model using the provided dataset."""
    print(f"Training model with data in {args.data_yaml} and params {args.params_yaml}...")

    model = YOLO("yolo11n-seg.pt")  # from pretrained
    model.add_callback("on_train_end", on_train_end)

    with open(args.params_yaml) as f:
        training_params = yaml.safe_load(f)

    # ignore background class for training
    classes = [c.index for c in segmentation_classes if c.name != 'background']

    results = model.train(data=args.data_yaml, classes=classes, **training_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)





