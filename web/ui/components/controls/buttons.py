# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/buttons.py
# Description: Button components for user interactions
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import uuid

from ui.components.base.component import Component

logger = logging.getLogger(__name__)


class Button(Component):
    """Base button component.
    
    This component provides a flexible button with styling and click handling.
    """
    
    def __init__(
        self,
        label: str,
        *,
        on_click: Optional[Callable[[], Any]] = None,
        type: str = "default",
        size: str = "medium",
        disabled: bool = False,
        full_width: bool = False,
        icon: Optional[str] = None,
        icon_position: str = "left",
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a button component.
        
        Args:
            label: Text to display on the button.
            on_click: Function to call when the button is clicked.
            type: Button type ('default', 'primary', 'secondary', 'text', 'icon').
            size: Button size ('small', 'medium', 'large').
            disabled: Whether the button is disabled.
            full_width: Whether the button should take up the full width.
            icon: Optional icon to display on the button.
            icon_position: Position of the icon ('left' or 'right').
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing Button component with label: {label}")
        # Generate a key for Streamlit button
        self.key = f"btn_{uuid.uuid4().hex[:8]}"
        
        # Prepare classes
        button_classes = ["btn", f"btn-{type}", f"btn-{size}"]
        if full_width:
            button_classes.append("btn-full-width")
        if disabled:
            button_classes.append("btn-disabled")
        if classes:
            button_classes.extend(classes)
        
        # Prepare attributes
        button_attributes = attributes or {}
        if disabled:
            button_attributes["disabled"] = "disabled"
            button_attributes["aria-disabled"] = "true"
        
        super().__init__(
            "controls",
            "button",
            id=id or self.key,
            classes=button_classes,
            attributes=button_attributes
        )
        
        self.label = label
        self.on_click = on_click
        self.type = type
        self.size = size
        self.disabled = disabled
        self.full_width = full_width
        self.icon = icon
        self.icon_position = icon_position if icon else "none"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Button initialized with type: {type}, disabled: {disabled}")
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering.
        
        Returns:
            Dict[str, Any]: Dictionary of variables for template rendering.
        """
        variables = super().get_template_variables()
        
        # Prepare icon HTML
        icon_html = f'<span class="btn-icon">{self.icon}</span>' if self.icon else ""
        
        variables.update({
            "BUTTON_LABEL": self.label,
            "BUTTON_ICON": icon_html,
            "ICON_POSITION": self.icon_position
        })
        
        return variables
    
    def display(self) -> bool:
        """Display the button and handle clicks.
        
        This method renders the visual button using our template system
        and also creates a Streamlit button to handle the click event.
        
        Returns:
            bool: True if the button was clicked, False otherwise.
        """
        self.logger.debug("Displaying button component")
        try:
            # Render the HTML button
            html = self.safe_render()
            
            # Create a simple button without nested columns
            clicked = st.button(
                self.label,
                key=self.key,
                disabled=self.disabled,
                help=self.label,
                type=self.type if self.type in ['primary', 'secondary'] else None,
                use_container_width=self.full_width,
            )
            
            # Apply custom styling via HTML
            st.markdown(
                f"<style>#{self.id} {{ /* Custom styles here */ }}</style>",
                unsafe_allow_html=True
            )
            
            # Handle click if the button was clicked and not disabled
            if clicked and not self.disabled and self.on_click:
                self.logger.info(f"Executing click handler for button: {self.label}")
                self.on_click()
            
            self.logger.debug("Button component displayed successfully")
            return clicked
        except Exception as e:
            self.logger.error(f"Error displaying button component: {str(e)}", exc_info=True)
            st.error("Error displaying button")
            return False

class PrimaryButton(Button):
    """Primary button component.
    
    A prominent button for primary actions.
    """
    
    def __init__(
        self,
        label: str,
        *,
        on_click: Optional[Callable[[], Any]] = None,
        size: str = "medium",
        disabled: bool = False,
        full_width: bool = False,
        icon: Optional[str] = None,
        icon_position: str = "left",
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a primary button.
        
        Args:
            label: Text to display on the button.
            on_click: Function to call when the button is clicked.
            size: Button size ('small', 'medium', 'large').
            disabled: Whether the button is disabled.
            full_width: Whether the button should take up the full width.
            icon: Optional icon to display on the button.
            icon_position: Position of the icon ('left' or 'right').
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing PrimaryButton component with label: {label}")
        super().__init__(
            label,
            on_click=on_click,
            type="primary",
            size=size,
            disabled=disabled,
            full_width=full_width,
            icon=icon,
            icon_position=icon_position,
            id=id,
            classes=classes,
            attributes=attributes
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("PrimaryButton initialized")


class SecondaryButton(Button):
    """Secondary button component.
    
    A less prominent button for secondary actions.
    """
    
    def __init__(
        self,
        label: str,
        *,
        on_click: Optional[Callable[[], Any]] = None,
        size: str = "medium",
        disabled: bool = False,
        full_width: bool = False,
        icon: Optional[str] = None,
        icon_position: str = "left",
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a secondary button.
        
        Args:
            label: Text to display on the button.
            on_click: Function to call when the button is clicked.
            size: Button size ('small', 'medium', 'large').
            disabled: Whether the button is disabled.
            full_width: Whether the button should take up the full width.
            icon: Optional icon to display on the button.
            icon_position: Position of the icon ('left' or 'right').
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        super().__init__(
            label,
            on_click=on_click,
            type="secondary",
            size=size,
            disabled=disabled,
            full_width=full_width,
            icon=icon,
            icon_position=icon_position,
            id=id,
            classes=classes,
            attributes=attributes
        )


class IconButton(Button):
    """Icon-only button component.
    
    A button that displays only an icon.
    """
    
    def __init__(
        self,
        icon: str,
        *,
        label: str,  # For accessibility, even though it's not displayed
        on_click: Optional[Callable[[], Any]] = None,
        size: str = "medium",
        disabled: bool = False,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize an icon button.
        
        Args:
            icon: Icon to display on the button.
            label: Accessible label for the button (not displayed).
            on_click: Function to call when the button is clicked.
            size: Button size ('small', 'medium', 'large').
            disabled: Whether the button is disabled.
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        # Prepare attributes for accessibility
        button_attributes = attributes or {}
        button_attributes["aria-label"] = label
        button_attributes["title"] = label
        
        super().__init__(
            "",  # Empty label for icon-only button
            on_click=on_click,
            type="icon",
            size=size,
            disabled=disabled,
            full_width=False,  # Icon buttons are never full-width
            icon=icon,
            icon_position="center",  # Special position for icon-only buttons
            id=id,
            classes=classes,
            attributes=button_attributes
        )