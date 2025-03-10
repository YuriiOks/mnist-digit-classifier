# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/__init__.py
# Description: Init file for views package
# Created: 2024-05-01

# Import views for easier access
from ui.views.home.home_view import HomeView
from ui.views.draw.draw_view import DrawView
from ui.views.history.history_view import HistoryView
from ui.views.settings.settings_view import SettingsView

# Don't import subpackages here to avoid circular imports
# These will be imported where needed directly

__all__ = [
    'HomeView',
    'DrawView',
    'HistoryView',
    'SettingsView'
] 