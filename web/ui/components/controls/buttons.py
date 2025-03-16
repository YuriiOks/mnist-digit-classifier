# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/buttons.py
# Description: Button components
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Dict, Any, Callable, Optional, Tuple

from utils.aspects import AspectUtils
from ui.components.base.component import Component

logger = logging.getLogger(__name__)


class Button(Component[bool]):
    """Base button component."""

    def __init__(
        self,
        label: str,
        *,
        key: Optional[str] = None,
        on_click: Optional[Callable] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
        help: Optional[str] = None,
        type: str = "default",
        id: Optional[str] = None,
        classes: Optional[list] = None,
        attributes: Optional[Dict[str, str]] = None,
        **extra_kwargs
    ):
        """
        Initialize a button.

        Args:
            label: Button text.
            key: Streamlit unique key.
            on_click: Callback function for click event.
            args: Arguments for callback.
            kwargs: Keyword arguments for callback.
            disabled: Disable state of button.
            help: Tooltip text.
            type: Button style (default, primary, secondary).
            id: HTML ID for the component.
            classes: List of CSS classes to apply.
            attributes: Dictionary of HTML attributes.
            **extra_kwargs: Additional keyword arguments.
        """
        super().__init__(
            component_type="controls",
            component_name="button",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **extra_kwargs
        )
        
        self.__label = label
        self.__key = key or f"button_{id}"
        self.__on_click = on_click
        self.__args = args
        self.__kwargs = kwargs or {}
        self.__disabled = disabled
        self.__help = help
        self.__type = type

    @property
    def label(self) -> str:
        """Get the button label."""
        return self.__label
    
    @property
    def type(self) -> str:
        """Get the button type."""
        return self.__type
    
    @property
    def disabled(self) -> bool:
        """Get the button disabled state."""
        return self.__disabled
    
    @disabled.setter
    def disabled(self, value: bool) -> None:
        """Set the button disabled state."""
        self.__disabled = value
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the button as HTML.
        
        Returns:
            HTML representation of the button.
        """
        # The button is primarily rendered by Streamlit, not as HTML
        return f"<button class='{' '.join(self.classes)}' data-testid='baseButton-{self.__type}'>{self.__label}</button>"

    @AspectUtils.catch_errors
    def display(self) -> bool:
        """
        Render the button using Streamlit.
        
        Returns:
            True if the button was clicked, False otherwise.
        """
        return st.button(
            label=self.__label,
            key=self.__key,
            on_click=self.__on_click,
            args=self.__args,
            kwargs=self.__kwargs,
            disabled=self.__disabled,
            help=self.__help,
            type=self.__type
        )


class PrimaryButton(Button):
    """Primary button with prominent styling."""

    def __init__(self, label: str, **kwargs):
        """
        Initialize a primary button.
        
        Args:
            label: Button text
            **kwargs: Additional button parameters
        """
        super().__init__(label, type="primary", **kwargs)


class SecondaryButton(Button):
    """Secondary button with less prominent styling."""

    def __init__(self, label: str, **kwargs):
        """
        Initialize a secondary button.
        
        Args:
            label: Button text
            **kwargs: Additional button parameters
        """
        super().__init__(label, type="secondary", **kwargs)


class IconButton(Button):
    """Button represented by an icon."""

    def __init__(
        self,
        icon: str,
        *,
        key: Optional[str] = None,
        on_click: Optional[Callable] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        disabled: bool = False,
        help: Optional[str] = None,
        label: Optional[str] = None,
        **extra_kwargs
    ):
        """
        Initialize an icon button.

        Args:
            icon: Icon (emoji or HTML).
            key: Streamlit unique key.
            on_click: Callback function for click event.
            args: Arguments for callback.
            kwargs: Keyword arguments for callback.
            disabled: Disable state of button.
            help: Tooltip text.
            label: Optional text label.
            **extra_kwargs: Additional keyword arguments.
        """
        display_label = f"{icon} {label}" if label else icon
        
        super().__init__(
            label=display_label,
            key=key,
            on_click=on_click,
            args=args,
            kwargs=kwargs or {},
            disabled=disabled,
            help=help,
            **extra_kwargs
        )