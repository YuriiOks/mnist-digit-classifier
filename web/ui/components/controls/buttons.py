# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/buttons.py
# Description: Custom button components
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Callable, Optional, List, Union

from ui.components.base.component import Component

logger = logging.getLogger(__name__)

class Button:
    """Base button component."""
    
    def __init__(
        self,
        label: str,
        key: Optional[str] = None,
        on_click: Optional[Callable] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
        help: Optional[str] = None,
        type: str = "default"
    ):
        """Initialize a button.
        
        Args:
            label: Button text
            key: Unique key for the button
            on_click: Function to call when button is clicked
            args: Arguments to pass to on_click
            kwargs: Keyword arguments to pass to on_click
            disabled: Whether the button is disabled
            help: Tooltip text
            type: Button type (default, primary, secondary, etc.)
        """
        self.label = label
        self.key = key
        self.on_click = on_click
        self.args = args
        self.kwargs = kwargs or {}
        self.disabled = disabled
        self.help = help
        self.type = type
    
    def render(self) -> bool:
        """Render the button.
        
        Returns:
            True if the button was clicked, False otherwise
        """
        try:
            return st.button(
                label=self.label,
                key=self.key,
                on_click=self.on_click,
                args=self.args,
                kwargs=self.kwargs,
                disabled=self.disabled,
                help=self.help,
                type=self.type
            )
        except Exception as e:
            logger.error(f"Error rendering button '{self.label}': {str(e)}")
            # Fallback to basic button
            return st.button(self.label, key=self.key, disabled=True, 
                            help="Error: Could not render button properly")


class PrimaryButton(Button):
    """Primary action button with prominent styling."""
    
    def __init__(
        self,
        label: str,
        key: Optional[str] = None,
        on_click: Optional[Callable] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
        help: Optional[str] = None
    ):
        """Initialize a primary button."""
        super().__init__(
            label=label,
            key=key,
            on_click=on_click,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            help=help,
            type="primary"
        )


class SecondaryButton(Button):
    """Secondary action button with less prominent styling."""
    
    def __init__(
        self,
        label: str,
        key: Optional[str] = None,
        on_click: Optional[Callable] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
        help: Optional[str] = None
    ):
        """Initialize a secondary button."""
        super().__init__(
            label=label,
            key=key,
            on_click=on_click,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            help=help,
            type="secondary"
        )


class IconButton:
    """Button represented by an icon."""
    
    def __init__(
        self,
        icon: str,
        key: Optional[str] = None,
        on_click: Optional[Callable] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
        help: Optional[str] = None,
        label: Optional[str] = None
    ):
        """Initialize an icon button.
        
        Args:
            icon: Icon to display (emoji or icon class)
            key: Unique key for the button
            on_click: Function to call when button is clicked
            args: Arguments to pass to on_click
            kwargs: Keyword arguments to pass to on_click
            disabled: Whether the button is disabled
            help: Tooltip text
            label: Optional text label to display alongside the icon
        """
        self.icon = icon
        self.key = key
        self.on_click = on_click
        self.args = args
        self.kwargs = kwargs or {}
        self.disabled = disabled
        self.help = help
        self.label = label
    
    def render(self) -> bool:
        """Render the icon button.
        
        Returns:
            True if the button was clicked, False otherwise
        """
        try:
            # If we have a label, use it with the icon
            display_label = f"{self.icon} {self.label}" if self.label else self.icon
            
            return st.button(
                label=display_label,
                key=self.key,
                on_click=self.on_click,
                args=self.args,
                kwargs=self.kwargs,
                disabled=self.disabled,
                help=self.help
            )
        except Exception as e:
            logger.error(f"Error rendering icon button '{self.icon}': {str(e)}")
            # Fallback to basic button
            return st.button("⚠️", key=self.key, disabled=True, 
                            help="Error: Could not render button properly")