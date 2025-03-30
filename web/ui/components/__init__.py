# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/components/__init__.py
# Description: Components package initialization
# Created: 2025-03-17
# Updated: 2025-03-30

"""UI components for the MNIST Digit Classifier."""

from ui.components.base.component import Component
from ui.components.cards.card import Card, FeatureCard, WelcomeCard
from ui.components.controls.buttons import (
    Button,
    PrimaryButton,
    SecondaryButton,
    IconButton,
)
from ui.components.controls.bb8_toggle import BB8Toggle
from ui.components import navigation
from ui.components import feedback
from ui.components import inputs

__all__ = [
    # Base components
    "Component",
    # Card components
    "Card",
    "FeatureCard",
    "WelcomeCard",
    # Control components
    "Button",
    "PrimaryButton",
    "SecondaryButton",
    "IconButton",
    "BB8Toggle",
    # Component packages
    "navigation",
    "feedback",
    "inputs",
]
