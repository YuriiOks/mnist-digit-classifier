# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/cards/content_card.py
# Description: Card component for displaying formatted content
# Created: 2024-05-01

import streamlit as st
import logging
from typing import List, Optional, Dict, Any, Literal

from ui.components.cards.card import Card

logger = logging.getLogger(__name__)

class ContentCard(Card):
    """Card component specifically for displaying rich content."""
    
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
        """Initialize a new content card component.
        
        Args:
            title: The card title text
            content: The HTML content to display in the card
            icon: Optional icon (emoji or FontAwesome) to display next to the title
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
        # Add content-card class if not provided
        classes = classes or []
        if "content-card" not in classes:
            classes.append("content-card")
            
        super().__init__(
            title=title,
            key=key,
            elevated=elevated,
            size=size,
            classes=classes,
            attributes=attributes,
            **kwargs
        )
        
        self.content = self.sanitize_html_content(content)
        self.icon = icon
        self.logger.debug(f"ContentCard initialized with title: {title}, size: {size}")
    
    def get_html(self) -> str:
        """Generate the HTML for the content card component.
        
        Returns:
            str: The HTML representation of the content card
        """
        # Combine all classes
        class_str = " ".join(self.classes)
        
        # Combine all attributes
        attr_str = " ".join([f'{k}="{v}"' for k, v in self.attributes.items()])
        
        # Add icon to title if provided
        title_with_icon = self.title
        if self.icon:
            title_with_icon = f"""<span class="card-icon">{self.icon}</span> {self.title}"""
        
        # Create card HTML - Using more concise HTML to avoid potential formatting issues
        card_html = f"""<div class="{class_str}" {attr_str}><div class="card-title">{title_with_icon}</div><div class="card-content">{self.content}</div></div>"""
        
        return card_html
    
    def get_content(self) -> str:
        """Get the content of the card.
        
        Returns:
            str: The HTML content to display in the card
        """
        return self.content
    
    def display(self) -> None:
        """Display the content card in the Streamlit app."""
        self.logger.debug(f"Displaying ContentCard: {self.title}")
        try:
            # Generate the HTML
            card_html = self.get_html()
            
            # Render the component - ensure unsafe_allow_html is True
            st.markdown(card_html, unsafe_allow_html=True)
            self.logger.debug("ContentCard displayed successfully")
        except Exception as e:
            self.logger.error(f"Error displaying ContentCard: {str(e)}", exc_info=True)
            st.error(f"Error displaying content card: {str(e)}")

    @staticmethod
    def sanitize_html_content(content: str) -> str:
        """Ensure HTML content is properly formatted for rendering."""
        # Remove any leading/trailing whitespace that could affect rendering
        content = content.strip()
        
        # If content has paragraphs but not wrapped properly, ensure they're formatted
        if not content.startswith('<p>') and not content.startswith('<div'):
            # Split content and wrap each line in paragraph tags if needed
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            formatted_lines = []
            for line in lines:
                if not (line.startswith('<') and line.endswith('>')):
                    line = f'<p>{line}</p>'
                formatted_lines.append(line)
            content = ''.join(formatted_lines)
        
        return content