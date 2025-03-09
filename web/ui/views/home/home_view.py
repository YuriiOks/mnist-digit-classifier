# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/home/home_view.py
# Description: Home view for the application
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any

from ui.views.base_view import BaseView
from ui.components.cards.content_card import ContentCard
from core.app_state.navigation_state import NavigationState

logger = logging.getLogger(__name__)

class HomeView(BaseView):
    """Home view for the application."""
    
    def __init__(self):
        """Initialize the home view."""
        super().__init__(
            view_id="home",
            title="Welcome to MNIST Digit Classifier",
            description=(
                "Draw and classify handwritten digits using machine learning"
            ),
            icon="🏠"
        )
        self.logger = logging.getLogger(__name__)
    
    def render(self) -> None:
        """Render the home view content."""
        self.logger.debug("Rendering home view")
        try:
            # Add some custom css for proper alignment
            st.markdown("""
            <style>
            /* Fix content alignment */
            .block-container {
                max-width: 100% !important;
                padding-top: 1rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            
            /* Make headers look better */
            h1, h2, h3 {
                margin-bottom: 1rem;
                margin-top: 0.5rem;
                font-family: var(--font-primary, 'Poppins', sans-serif);
            }
            
            /* Add space around elements */
            .stMarkdown {
                margin-bottom: 0.5rem;
            }
            
            /* Remove empty columns */
            .stColumn:empty {
                display: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Welcome card - FIXED content format for proper rendering
            st.markdown("""
            <div class="card card-elevated content-card welcome-card animate-fade-in">
                <div class="card-title">
                    <span class="card-icon">👋</span>
                    Welcome
                </div>
                <div class="card-content">
                    <p>This application allows you to draw digits and have them recognized 
                       by a machine learning model.</p>
                    <p>The model is trained on the MNIST dataset, which contains thousands 
                       of handwritten digit images.</p>
                    <p>Draw any digit from 0-9 and let our AI predict what you've written!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # How it works section
            st.markdown("<h2>How It Works</h2>", unsafe_allow_html=True)
            
            # Use columns for the cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Draw digit feature
                card1 = ContentCard(
                    title="1. Draw a Digit",
                    icon="✏️",
                    content="""
                    <p>Navigate to the Drawing Canvas and use your mouse or touch to draw any digit from 0-9.</p>
                    <p>You can adjust the brush size and use the clear button to start over.</p>
                    """,
                    elevated=True,
                    classes=["feature-card", "primary-card"]
                )
                card1.display()
            
            with col2:
                # Classify digit feature
                card2 = ContentCard(
                    title="2. Classify the Digit",
                    icon="🔍",
                    content="""
                    <p>Once you've drawn a digit, click the "Classify" button to have the model predict what number you've drawn.</p>
                    <p>The model will analyze your drawing and return its best guess.</p>
                    """,
                    elevated=True,
                    classes=["feature-card", "secondary-card"]
                )
                card2.display()
            
            with col3:
                # View history feature
                card3 = ContentCard(
                    title="3. View History",
                    icon="📊",
                    content="""
                    <p>All your predictions are saved to your session history.</p>
                    <p>You can view past predictions, compare results, and see statistics about your drawing classifications.</p>
                    """,
                    elevated=True,
                    classes=["feature-card", "accent-card"]
                )
                card3.display()
            
            # Call to action section
            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

            # Use columns to place heading and button side by side
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown("<h2 style='margin-top: 5px;'>Ready to try it out?</h2>", unsafe_allow_html=True)

            with col2:
                # Button styling - now with smaller size
                st.markdown("""
                <style>
                /* Start Drawing button styling */
                .stButton button[kind="primary"] {
                    background: linear-gradient(90deg, var(--color-primary, #4F46E5), var(--color-secondary, #06B6D4)) !important;
                    color: white !important;
                    border: none !important;
                    padding: 0.35rem 1.2rem !important; /* Reduced padding for smaller button */
                    font-size: 1rem !important; /* Slightly smaller font */
                    border-radius: 0.5rem !important;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
                    transition: all 0.3s ease !important;
                    margin-top: 0 !important;
                    position: relative !important;
                    overflow: hidden !important;
                    font-family: var(--font-primary, 'Poppins', sans-serif) !important;
                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
                }
                
                /* Shimmer effect on hover */
                .stButton button[kind="primary"]::after {
                    content: '' !important;
                    position: absolute !important;
                    top: -50% !important;
                    left: -50% !important;
                    width: 200% !important;
                    height: 200% !important;
                    background: linear-gradient(
                        to right,
                        rgba(255, 255, 255, 0) 0%,
                        rgba(255, 255, 255, 0.3) 50%,
                        rgba(255, 255, 255, 0) 100%
                    ) !important;
                    transform: rotate(30deg) !important;
                    opacity: 0 !important;
                    transition: opacity 0.3s ease !important;
                    pointer-events: none !important;
                }
                
                .stButton button[kind="primary"]:hover {
                    transform: translateY(-3px) !important;
                    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2) !important;
                }
                
                .stButton button[kind="primary"]:hover::after {
                    opacity: 1 !important;
                    animation: buttonShine 1.5s ease-in-out !important;
                }
                
                @keyframes buttonShine {
                    0% {
                        transform: rotate(30deg) translate(-100%, -100%) !important;
                    }
                    100% {
                        transform: rotate(30deg) translate(100%, 100%) !important;
                    }
                }
                </style>
                """, unsafe_allow_html=True)
                
                if st.button("Start Drawing", type="primary", use_container_width=False):
                    self.logger.info("Start Drawing button clicked")
                    NavigationState.set_active_view("draw")
                    st.rerun()
                
            self.logger.debug("Home view rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering home view: {str(e)}", exc_info=True)
            st.error(f"An error occurred while rendering the home view: {str(e)}")
    
    def _setup(self) -> None:
        """Set up the home view."""
        self.logger.debug("Entering _setup")
        try:
            # Load any required resources here
            pass
        except Exception as e:
            self.logger.error(f"Error setting up home view: {str(e)}", exc_info=True)
        self.logger.debug("Exiting _setup")
    
    def get_view_data(self) -> Dict[str, Any]:
        """Get view-specific data for templates.
        
        Returns:
            Dict[str, Any]: Dictionary of view data
        """
        self.logger.debug("Entering get_view_data")
        try:
            data = super().get_view_data()
            # Add home-specific data
            data.update({
                "sections": ["welcome", "features", "how-it-works", "cta"]
            })
            self.logger.debug("Exiting get_view_data")
            return data
        except Exception as e:
            self.logger.error(f"Error getting view data: {str(e)}", exc_info=True)
            return super().get_view_data()