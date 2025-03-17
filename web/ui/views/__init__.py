# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/__init__.py
# Description: Views package initialization
# Created: 2025-03-17

"""View components for the MNIST Digit Classifier."""

from ui.views.base_view import View
from ui.views.home_view import HomeView
from ui.views.draw_view import DrawView
from ui.views.history_view import HistoryView
from ui.views.settings_view import SettingsView

# Create instances of each view for easy access
home_view = HomeView()
draw_view = DrawView()
history_view = HistoryView()
settings_view = SettingsView()

# Dictionary mapping view names to view instances
views = {
    "home": home_view,
    "draw": draw_view,
    "history": history_view,
    "settings": settings_view
}

__all__ = [
    "View",
    "HomeView",
    "DrawView",
    "HistoryView",
    "SettingsView",
    "home_view",
    "draw_view",
    "history_view",
    "settings_view",
    "views"
]