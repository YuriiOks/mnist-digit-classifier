# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/__init__.py
# Description: Components package initialization
# Created: 2024-05-01

# Import components for easier access
from ui.components.base_component import BaseComponent
from ui.components import navigation
from ui.components import cards
from ui.components import controls

# Make components available at the package level
__all__ = [
    "BaseComponent",
    "navigation",
    "cards",
    "controls"
]