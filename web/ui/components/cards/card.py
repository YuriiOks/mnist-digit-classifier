# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/cards/card.py
# Description: Base card UI component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import List, Optional, Dict, Any, Literal

from ui.components.base_component import BaseComponent

logger = logging.getLogger(__name__)

class Card(BaseComponent):
    """Base card component for displaying content with consistent styling."""
    
    def __init__(
        self,
        title: str,
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
        self.title = title
        self.elevated = elevated
        self.size = size
        self.classes = classes or []
        self.attributes = attributes or {}
        
        # Add basic card class
        if "card" not in self.classes:
            self.classes.append("card")
        
        # Add elevated class if needed
        if self.elevated and "elevated" not in self.classes:
            self.classes.append("elevated")
        
        # Add size class if specified
        if self.size == "small":
            self.classes.append("small")
        elif self.size == "large":
            self.classes.append("large")
        
        self.logger.debug(f"Card initialized with title: {title}, size: {size}, classes: {self.classes}")
    
    def get_html(self) -> str:
        """Generate the HTML for the card component.
        
        Returns:
            str: The HTML representation of the card
        """
        # Combine all classes
        class_str = " ".join(self.classes)
        
        # Combine all attributes
        attr_str = " ".join([f'{k}="{v}"' for k, v in self.attributes.items()])
        
        # Create card HTML
        html = f"""
        <div class="{class_str}" {attr_str}>
            <div class="card-title">
                {self.title}
            </div>
            <div class="card-content">
                {self.get_content()}
            </div>
        </div>
        """
        
        return html
    
    def get_content(self) -> str:
        """Get the content of the card. To be overridden by subclasses.
        
        Returns:
            str: The HTML content to display in the card
        """
        return "Card content goes here."
    
    def display(self) -> None:
        """Display the card in the Streamlit app."""
        self.logger.debug(f"Displaying card: {self.title}")
        try:
            # We don't need to load CSS here since it's loaded globally via components_css.py
            # Just render the HTML
            st.markdown(self.get_html(), unsafe_allow_html=True)
            self.logger.debug(f"Card {self.title} displayed successfully")
        except Exception as e:
            self.logger.error(f"Error displaying card: {str(e)}", exc_info=True)
            st.error(f"Error displaying card: {str(e)}")