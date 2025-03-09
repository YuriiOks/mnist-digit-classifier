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
        try:
            # Build card HTML
            card_html = f"""
            <div class="card {' '.join(self.classes)}" id="{self.id or ''}" 
                {' '.join([f'{k}="{v}"' for k, v in self.attributes.items()])}>
                <div class="card-title">{self.title}</div>
                <div class="card-content">{self.content}</div>
            </div>
            """
            
            # Add CSS for the card
            card_css = """
            <style>
            /* Card styling */
            .card {
                background-color: var(--color-card, white);
                border: 1px solid var(--color-border, #e5e7eb);
                border-radius: 0.75rem;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                transition: all 0.3s ease;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
            }
            
            /* Card with elevation */
            .card-elevated {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
            }
            
            /* Hover effects */
            .card:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05);
                border-color: var(--color-primary-light, rgba(99, 102, 241, 0.4));
            }
            
            /* Card title */
            .card-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: var(--color-text, #111827);
                line-height: 1.4;
            }
            
            /* Card content */
            .card-content {
                color: var(--color-text-light, #4b5563);
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
        except Exception as e:
            self.logger.error(f"Error displaying card: {str(e)}", exc_info=True)
            st.error(f"Error displaying card: {str(e)}")