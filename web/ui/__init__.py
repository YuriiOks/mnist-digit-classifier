# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/__init__.py
# Description: UI package initialization
# Created: 2025-03-17
# Updated: 2025-03-30

"""UI components, layouts, and views for the MNIST Digit Classifier."""

from ui.layout import Layout, Header, Footer, Sidebar
from ui.theme.theme_manager import ThemeManager, theme_manager
from ui.theme.css_manager import CSSManager, css_manager
from ui.components.base.component import Component
from ui.components.cards.card import Card, FeatureCard, WelcomeCard
from ui.components.controls.buttons import (
    Button,
    PrimaryButton,
    SecondaryButton,
    IconButton,
)
from ui.components.controls.bb8_toggle import BB8Toggle
from ui.views import HomeView, DrawView, HistoryView, SettingsView

__all__ = [
    # Layout components
    "Layout",
    "Header",
    "Footer",
    "Sidebar",
    # Theme management
    "ThemeManager",
    "theme_manager",
    "CSSManager",
    "css_manager",
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
    # Views
    "HomeView",
    "DrawView",
    "HistoryView",
    "SettingsView",
]
