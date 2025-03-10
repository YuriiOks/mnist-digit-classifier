# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/file/__init__.py
# Description: File utilities package for the MNIST Digit Classifier
# Created: 2024-05-01

# Import key functions to make them available at the package level
from utils.file.path_utils import get_project_root, resolve_path, get_asset_path, get_template_path
from utils.file.file_loader import load_text_file, load_binary_file, load_json_file, file_exists

__all__ = [
    "get_project_root",
    "resolve_path",
    "get_asset_path",
    "get_template_path",
    "load_text_file",
    "load_binary_file",
    "load_json_file",
    "file_exists"
] 