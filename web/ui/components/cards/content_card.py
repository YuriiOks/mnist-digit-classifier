# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/cards/content_card.py
# Description: Content card component for UI
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List

from ui.components.cards.card import Card

logger = logging.getLogger(__name__)

class ContentCard(Card):
    """Content card component for UI.
    
    This component renders a card with icon, title, and content.
    """
    
    def __init__(
        self,
        title: str = "",
        content: str = "",
        *,
        icon: str = "",
        elevated: bool = False,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a content card component.
        
        Args:
            title: Card title.
            content: Card content (HTML).
            icon: Icon to display with the title.
            elevated: Whether the card should have elevation (shadow).
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dict of HTML attributes to apply to the component.
        """
        logger.debug(
            f"Initializing ContentCard with title: {title}, icon: {icon}"
        )
        class_list = ["content-card"]
        if classes:
            class_list.extend(classes)
            
        super().__init__(
            title=title,
            content=content,
            elevated=elevated,
            id=id,
            classes=class_list,
            attributes=attributes
        )
        self.component_name = "content_card"  # Override the component name
        self.icon = icon
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering."""
        variables = super().get_template_variables()
        variables.update({
            "ICON": self.icon if self.icon else ""
        })
        return variables
    
    def display(self) -> None:
        """Display the content card component."""
        self.logger.debug(f"Displaying content card: {self.title}")
        
        # Use direct HTML generation for maximum reliability
        icon_html = (
            f'<span class="card-icon">{self.icon}</span>' 
            if self.icon else ''
        )
        class_attr = " ".join(self.classes)
        attrs = " ".join(
            [f'{k}="{v}"' for k, v in self.attributes.items()]
        ) if self.attributes else ""
        
        # Ensure content is properly sanitized and formatted
        content = self.content.strip()
        
        # Generate simple, robust HTML
        card_html = f"""
        <div class="{class_attr}" id="{self.id}" {attrs}>
            <div class="card-title">
                {icon_html}
                {self.title}
            </div>
            <div class="card-content">
                {content}
            </div>
        </div>
        """
        
        # Add essential CSS for consistent rendering
        card_css = """
        <style>
        /* Essential card styling */
        .content-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        /* Card title styling */
        .card-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #333333;
            display: flex;
            align-items: center;
        }
        
        /* Card content styling */
        .card-content {
            font-family: 'Nunito', sans-serif;
            line-height: 1.6;
            color: #666666;
            flex-grow: 1;
        }
        
        .card-content p {
            margin-bottom: 0.75rem;
        }
        
        /* Dark mode adjustments */
        [data-theme="dark"] .content-card {
            background-color: #2a2a2a;
            border-color: #444444;
        }
        
        [data-theme="dark"] .card-title {
            color: #e0e0e0;
        }
        
        [data-theme="dark"] .card-content,
        [data-theme="dark"] .card-content p {
            color: #b0b0b0;
        }
        </style>
        """
        
        # Render the component - ensure unsafe_allow_html is True
        st.markdown(card_css + card_html, unsafe_allow_html=True)
        self.logger.debug("ContentCard displayed successfully")

    @staticmethod
    def sanitize_html_content(content):
        """Ensure HTML content is properly formatted for rendering."""
        # Remove any leading/trailing whitespace that could affect rendering
        content = content.strip()
        
        # Ensure content doesn't have extra div tags that could break structure
        if content.startswith('<div') and content.endswith('</div>'):
            # Content already has a div wrapper, keep it as is
            return content
        
        # If content has paragraphs but not wrapped properly, ensure they're formatted
        if '<p>' in content and not content.startswith('<p>'):
            # Split content and wrap each line in paragraph tags if needed
            lines = content.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if (line and not line.startswith('<') 
                        and not line.endswith('>')):
                    line = f'<p>{line}</p>'
                formatted_lines.append(line)
            return ''.join(formatted_lines)
        
        return content