# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/ui/__init__.py
# Description: UI utilities package for the MNIST Digit Classifier
# Created: 2024-05-01

"""UI utilities for the MNIST Digit Classifier application."""

# Import key functions to make them available at the package level
from utils.ui.layout import fix_layout_issues

# Import only what exists in view_utils.py
from utils.ui.view_utils import apply_view_styling, create_card, create_section_container, close_section_container

__all__ = [
    "fix_layout_issues", 
    'apply_view_styling', 
    'create_card',
    'create_section_container',
    'close_section_container'
]