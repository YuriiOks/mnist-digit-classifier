# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/cards/card.py
# Description: Base card UI component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import List, Optional, Dict, Any, Literal, Callable

from ui.components.base_component import BaseComponent

logger = logging.getLogger(__name__)

class Card(BaseComponent):
    """Base card component for displaying content with consistent styling."""
    
    def __init__(
        self,
        title: str,
        content: str,
        icon: Optional[str] = None,
        key: Optional[str] = None,
        elevated: bool = False,
        size: Literal["small", "large", "default"] = "default",
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Initialize a new card component.
        
        Args:
            title: The card title text
            key: Optional unique key for the component
            elevated: Whether to display the card with elevation (shadow)
            size: Size of the card, which determines color scheme
                 - "small": Uses secondary color
                 - "large": Uses primary color
                 - "default": No specific size/color (use other classes)
            classes: Additional CSS classes to apply to the card
            attributes: Additional HTML attributes for the card
            **kwargs: Additional keyword arguments for the component
        """
        super().__init__(key=key, **kwargs)
        self.__title = title
        self.__content = content
        self.__icon = icon
        self.__elevated = elevated
        self.__size = size
        self.__classes = classes or []
        self.__attributes = attributes or {}
        
        # Add basic card class
        if "card" not in self.__classes:
            self.__classes.append("card")
        
        # Add elevated class if needed
        if self.__elevated and "elevated" not in self.__classes:
            self.__classes.append("elevated")
        
        # Add size class if specified
        if self.__size == "small":
            self.__classes.append("small")
        elif self.__size == "large":
            self.__classes.append("large")

        self.logger.debug(f"Card initialized with title: {self.__title}, size: {self.__size}, classes: {self.__classes}")

    def render(self, title: str, icon: str, content: str,
               template_loader: Callable[[str], str],
               type_card: str = "welcome") -> str:

        """Create a rendered card for any view.

        Args:
            title: Card title
            icon: Emoji icon to display
            content: HTML content for the card (paragraphs)
  
        Returns:
            str: HTML for the welcome card
        """
        formatted_content = ''.join(f'<p>{p.strip()}</p>' for p in content.split('\n') if p.strip())
        return template_loader(f"/components/controls/cards/{type_card}.html", {
            "title": title,
            "icon": icon,
            "content": formatted_content
        })
    
    def display(self, template_loader: Callable[[str], str]) -> None:
        """Display the card in the Streamlit app."""
        self.logger.debug(f"Displaying card: {self.__title}")
        try:
            # We don't need to load CSS here since it's loaded globally via components_css.py
            # Just render the HTML
            content_to_show = self.render(title=self.__title, 
                                          icon=self.__icon, 
                                          content=self.__content,
                                          template_loader=template_loader,
                                          type_card=self.__classes[0])

            st.markdown(content_to_show, unsafe_allow_html=True)
            self.logger.debug(f"Card {self.__title} displayed successfully")
        except Exception as e:
            self.logger.error(f"Error displaying card: {str(e)}", exc_info=True)
            st.error(f"Error displaying card: {str(e)}")