"""
Convert trained YOLO models to MaixCAM format using TPU-MLIR.

This package provides functionality to convert ONNX models to the MaixCAM-compatible
format using Docker-based TPU-MLIR tools.
"""

from .convert_model import main, add_arg_parser, setup_workspace, get_resource_path, convert_model_to_maixcam

__all__ = ['main', 'add_arg_parser', 'setup_workspace', 'get_resource_path', 'convert_model_to_maixcam']
