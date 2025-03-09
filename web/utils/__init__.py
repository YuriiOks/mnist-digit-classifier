# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/__init__.py
# Description: Init file for utils package
# Created: 2024-05-01

# This makes the utils directory a proper Python package

# Import key functions to make them available at the package level
from utils.file.path_utils import get_project_root, resolve_path, get_asset_path, get_template_path
from utils.file.file_loader import load_text_file, load_binary_file, load_json_file, file_exists