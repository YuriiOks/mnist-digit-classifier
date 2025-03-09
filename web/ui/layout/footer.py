# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/footer.py
# Description: Footer component for the application
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ui.components.base.component import Component


logger = logging.getLogger(__name__)


class Footer(Component):
    """Footer component for the application.
    
    This component renders the app footer with copyright text and links.
    """
    
    def __init__(
        self,
        content: Optional[str] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize the footer component.
        
        Args:
            content: Footer content (HTML).
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dict of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing Footer with content: {content}")
        super().__init__(
            component_type="layout",
            component_name="footer",
            id=id,
            classes=classes,
            attributes=attributes
        )
        # Set default content with GitHub links and attribution if no content provided
        if content is None:
            current_year = datetime.now().year
            self.content = (
                f"© {current_year} MNIST Digit Classifier | "
                f"Developed by <a href=\"https://github.com/YuriODev\" "
                f"target=\"_blank\">YuriODev</a> | "
                f"<a href=\"https://github.com/YuriODev/mnist-digit-classifier\" "
                f"target=\"_blank\">"
                f"<span style=\"white-space: nowrap;\">"
                f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" "
                f"height=\"16\" fill=\"currentColor\" viewBox=\"0 0 16 16\" "
                f"style=\"vertical-align: text-bottom; margin-right: 4px;\">"
                f"<path d=\"M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07"
                f".55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-"
                f".09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 "
                f"1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-"
                f"3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12"
                f" 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 "
                f"1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27"
                f".82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07"
                f"-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-"
                f"4.42-3.58-8-8-8z\"/>"
                f"</svg>"
                f"GitHub"
                f"</span></a>"
            )
        else:
            self.content = content
        
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
    
    def display(self) -> None:
        """Display the footer component."""
        self.logger.debug("Displaying footer component")
        try:
            # Create a simplified footer with gradient background to match header
            footer_css = """
            <style>
            /* Footer with gradient background - inverted from header for aesthetics */
            .app-footer {
                background: linear-gradient(90deg, var(--color-secondary, #06B6D4), var(--color-primary-light, #6366F1));
                color: white;
                padding: 0.75rem 1.5rem;
                margin-top: 2rem;
                text-align: center;
                font-size: 0.9rem;
                position: relative;
                border-radius: 0.5rem;
                box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
                overflow: hidden;
            }
            
            /* Footer content */
            .footer-content {
                position: relative;
                z-index: 1;
                text-align: center; /* Ensure content is centered */
            }
            
            /* Links in footer */
            .app-footer a {
                color: white;
                text-decoration: underline;
                font-weight: 500;
                transition: opacity 0.2s ease;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
            }
            
            .app-footer a:hover {
                opacity: 0.9;
                text-decoration: none;
            }
            
            /* Subtle shimmer effect for the footer */
            .app-footer::after {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(
                    to right,
                    rgba(255, 255, 255, 0) 0%,
                    rgba(255, 255, 255, 0.2) 50%,
                    rgba(255, 255, 255, 0) 100%
                );
                transform: rotate(-30deg);
                animation: footerShimmer 8s infinite linear;
                pointer-events: none;
            }
            
            @keyframes footerShimmer {
                0% {
                    transform: rotate(-30deg) translate(-100%, 100%);
                }
                100% {
                    transform: rotate(-30deg) translate(100%, -100%);
                }
            }
            </style>
            """
            
            # Simple footer with just the content
            footer_html = f"""
            <footer class="app-footer">
                <div class="footer-content">
                    {self.content}
                </div>
            </footer>
            """
            
            # Render the footer
            st.markdown(footer_css + footer_html, unsafe_allow_html=True)
            
            self.logger.debug("Footer displayed successfully")
        except Exception as e:
            self.logger.error(
                f"Error displaying footer: {str(e)}", 
                exc_info=True
            )
            st.error("Error loading footer component")
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for footer rendering."""
        try:
            variables = super().get_template_variables()
            variables.update({
                "CONTENT": self.content
            })
            return variables
        except Exception as e:
            self.logger.error(
                f"Error getting template variables: {str(e)}", 
                exc_info=True
            )
            # Return basic variables to prevent rendering failure
            return {
                "CONTENT": self.content
            }