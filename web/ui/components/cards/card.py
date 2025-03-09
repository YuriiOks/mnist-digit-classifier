# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/cards/card.py
# Description: Card component for UI
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List

from ui.components.base.component import Component

logger = logging.getLogger(__name__)

class Card(Component):
    """Card component for UI.
    
    This component renders a card with title and content.
    """
    
    def __init__(
        self,
        title: str = "",
        content: str = "",
        *,
        elevated: bool = False,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a card component.
        
        Args:
            title: Card title.
            content: Card content (HTML).
            elevated: Whether the card should have elevation (shadow).
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing Card with title: {title}")
        class_list = ["card"]
        if elevated:
            class_list.append("card-elevated")
        if classes:
            class_list.extend(classes)
            
        super().__init__(
            component_type="cards", 
            component_name="card",
            id=id,
            classes=class_list,
            attributes=attributes
        )
        self.title = title
        self.content = content
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering."""
        variables = super().get_template_variables()
        variables.update({
            "TITLE": self.title,
            "CONTENT": self.content,
        })
        return variables
    
    def display(self) -> None:
        """Display the card component."""
        self.logger.debug(f"Displaying card: {self.title}")
        
        # Use direct HTML generation for reliability
        class_attr = " ".join(self.classes)
        attrs = " ".join(
            [f'{k}="{v}"' for k, v in self.attributes.items()]
        ) if self.attributes else ""
        
        card_html = f"""
        <div class="{class_attr}" id="{self.id}" {attrs}>
            <div class="card-title">{self.title}</div>
            <div class="card-content">{self.content}</div>
        </div>
        """
        
        # Add essential CSS directly
        card_css = """
        <style>
        .card {
            background-color: var(--color-card, white);
            border: 1px solid var(--color-border, #e0e0e0);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 
                        0 1px 3px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        /* Gradient top border */
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(to right, #4F46E5, #06B6D4);
            opacity: 0.8;
            transition: opacity 0.2s ease;
        }
        
        .card:hover::before {
            opacity: 1;
        }
        
        .card.card-elevated {
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1), 
                        0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1), 
                        0 8px 10px rgba(0, 0, 0, 0.05);
            border-color: rgba(79, 70, 229, 0.3);
        }
        
        .card-title {
            font-family: var(--font-primary, 'Poppins', sans-serif);
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--color-text, #333333);
        }
        
        .card-content {
            font-family: var(--font-secondary, 'Nunito', sans-serif);
            color: var(--color-text-light, #666666);
            line-height: 1.6;
        }
        
        /* Shine effect on hover */
        .card::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                to right,
                rgba(255, 255, 255, 0) 0%,
                rgba(255, 255, 255, 0.1) 50%,
                rgba(255, 255, 255, 0) 100%
            );
            transform: rotate(30deg);
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }
        
        .card:hover::after {
            opacity: 1;
            animation: shine 1.5s ease-in-out;
        }
        
        @keyframes shine {
            0% {
                transform: rotate(30deg) translate(-100%, -100%);
            }
            100% {
                transform: rotate(30deg) translate(100%, 100%);
            }
        }
        
        /* Dark mode adjustments */
        [data-theme="dark"] .card {
            background-color: var(--color-card, #2a2a2a);
            border-color: var(--color-border, #444444);
        }
        
        [data-theme="dark"] .card-title {
            color: var(--color-text, #e0e0e0);
        }
        
        [data-theme="dark"] .card-content {
            color: var(--color-text-light, #b0b0b0);
        }
        
        [data-theme="dark"] .card:hover {
            border-color: rgba(129, 140, 248, 0.4);
        }
        </style>
        """
        
        # Render the component
        st.markdown(card_css + card_html, unsafe_allow_html=True)
        self.logger.debug("Card displayed successfully")