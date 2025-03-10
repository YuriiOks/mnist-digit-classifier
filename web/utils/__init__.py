# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/__init__.py
# Description: Utils package initialization
# Created: 2024-05-01

# This makes the utils directory a proper Python package

# Import key functions to make them available at the package level
from utils.file.path_utils import get_project_root, resolve_path, get_asset_path, get_template_path
from utils.file.file_loader import load_text_file, load_binary_file, load_json_file, file_exists
from utils.css.css_loader import load_css, load_theme_css, inject_css, load_css_file
from utils.html.html_loader import load_html_template
from utils.ui.layout import fix_layout_issues
from utils.ui.view_utils import apply_view_styling
from utils.template_loader import TemplateLoader
from utils.data_loader import DataLoader



__all__ = [
    "apply_view_styling",
    "get_project_root",
    "resolve_path",
    "get_asset_path",
    "get_template_path",
    "load_text_file",
    "load_binary_file",
    "load_json_file",
    "file_exists",
    "load_css",
    "load_theme_css",
    "inject_css",
    "load_css_file",
    "load_html_template",
    "fix_layout_issues",
    "TemplateLoader",
    "DataLoader"
]
