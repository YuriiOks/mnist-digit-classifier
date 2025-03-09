# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/__init__.py
# Description: Init file for views package
# Created: 2024-05-01

# Import views for easier access
from ui.views.home import HomeView
from ui.views.drawing import DrawingView
from ui.views.history import HistoryView

# Don't import subpackages here to avoid circular imports
# These will be imported where needed directly

__all__ = [
    'HomeView',
    'DrawingView',
    'HistoryView'
] 