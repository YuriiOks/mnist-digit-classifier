# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/cards/card.py
# Description: Card component for displaying styled content
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Optional, List, Dict, Any

from ui.components.base.component import Component
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils

class Card(Component[None]):
    """Card component for displaying styled content with consistency."""

    def __init__(
        self,
        title: str,
        content: str = "",
        icon: str = "",
        elevated: bool = True,
        size: str = "medium",
        *,
        key: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize a Card component.

        Args:
            title: Card title.
            content: Card content (HTML or text).
            icon: Optional icon (emoji or HTML).
            elevated: Card elevation (shadow effect).
            size: Card size ('small', 'medium', 'large').
            key: Unique key for Streamlit.
            id: HTML ID for the component.
            classes: List of CSS classes to apply.
            attributes: Dictionary of HTML attributes.
            **kwargs: Additional keyword arguments.
        """
        # Initialize base properties
        component_classes = classes or []
        if elevated:
            component_classes.append("card-elevated")
        component_classes.append(f"card-{size}")
        
        super().__init__(
            component_type="cards",
            component_name="card",
            id=id,
            classes=component_classes,
            attributes=attributes,
            key=key,
            **kwargs
        )
        
        # Store specific card properties
        self.__title = title
        self.__content = content
        self.__icon = icon
        self.__size = size
        
        self._logger.debug(f"Initialized Card: {self.__title}")

    @property
    def title(self) -> str:
        """Get the card title."""
        return self.__title

    @property
    def content(self) -> str:
        """Get the card content."""
        return self.__content

    @property
    def icon(self) -> str:
        """Get the card icon."""
        return self.__icon

    @property
    def size(self) -> str:
        """Get the card size."""
        return self.__size
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the card to HTML.
        
        Returns:
            HTML representation of the card.
        """
        # Try to render using template
        template_paths = [
            "components/cards/card.html",
            "cards/card.html",
            "components/card.html"
        ]
        
        # Try each template path
        for template_path in template_paths:
            template_content = self.load_template(template_path)
            if template_content:
                context = {
                    "TITLE": self.__title,
                    "CONTENT": self.__content,
                    "ICON": self.__icon
                }
                rendered = self.render_template(template_path, context)
                if rendered:
                    return rendered
        
        # Fallback to direct HTML generation if template loading fails
        class_list = " ".join(self.classes)
        return f"""
        <div class="card {class_list}" id="{self.component_id}">
            <div class="card-title">
                <span class="card-icon">{self.__icon}</span> {self.__title}
            </div>
            <div class="card-content">{self.__content}</div>
        </div>
        """

    @AspectUtils.catch_errors
    def display(self) -> None:
        """
        Display the card in Streamlit.
        
        This method renders the card HTML and displays it in the Streamlit UI.
        """
        # Get the HTML from render()
        html = self.render()
        
        # Display using Streamlit
        st.markdown(html, unsafe_allow_html=True)
        self._logger.debug(f"Displayed Card: {self.__title}")


class FeatureCard(Card):
    """Feature card with specialized styling for feature highlights."""

    def __init__(
        self,
        title: str,
        content: str = "",
        icon: str = "",
        *,
        key: Optional[str] = None,
        id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a feature card.

        Args:
            title: Card title.
            content: Card content (HTML or text).
            icon: Optional icon (emoji or HTML).
            key: Unique key for Streamlit.
            id: HTML ID for the component.
            **kwargs: Additional keyword arguments.
        """
        # Initialize with custom classes and defaults
        classes = kwargs.pop("classes", [])
        classes.extend(["feature-card", "animate-fade-in"])
        
        super().__init__(
            title=title,
            content=content,
            icon=icon,
            elevated=True,
            size="small",  # Feature cards are typically smaller
            key=key,
            id=id,
            classes=classes,
            **kwargs
        )


class WelcomeCard(Card):
    """Welcome card with specialized styling."""

    def __init__(
        self,
        title: str,
        content: str = "",
        icon: str = "",
        *,
        key: Optional[str] = None,
        id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a welcome card.

        Args:
            title: Card title.
            content: Card content (HTML or text).
            icon: Optional icon (emoji or HTML).
            key: Unique key for Streamlit.
            id: HTML ID for the component.
            **kwargs: Additional keyword arguments.
        """
        # Initialize with custom classes and defaults
        classes = kwargs.pop("classes", [])
        classes.extend(["welcome-card", "animate-fade-in"])
        
        super().__init__(
            title=title,
            content=content,
            icon=icon,
            elevated=True,
            size="large",  # Welcome cards are typically larger
            key=key,
            id=id,
            classes=classes,
            **kwargs
        )