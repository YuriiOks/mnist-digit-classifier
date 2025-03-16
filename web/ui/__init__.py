# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/__init__.py
# Description: UI package initialization
# Created: 2025-03-16

"""UI components, layouts, and views for the MNIST Digit Classifier."""

from ui.layout.layout_components import Layout, Header, Footer, Sidebar
from ui.theme.theme_manager import ThemeManager, theme_manager
from ui.components.base.component import Component
from ui.components.cards.card import Card, FeatureCard, WelcomeCard
from ui.components.controls.buttons import Button, PrimaryButton, SecondaryButton, IconButton
from ui.components.controls.bb8_toggle import BB8Toggle

__all__ = [
    # Layout components
    "Layout",
    "Header", 
    "Footer",
    "Sidebar",
    
    # Theme management
    "ThemeManager",
    "theme_manager",
    
    # Base components
    "Component",
    
    # UI Components
    "Card",
    "FeatureCard",
    "WelcomeCard",
    "Button",
    "PrimaryButton",
    "SecondaryButton",
    "IconButton",
    "BB8Toggle"
]