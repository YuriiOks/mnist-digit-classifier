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

# Try to import the view utils if available
try:
    from utils.ui.view_utils import create_welcome_card
except ImportError:
    # Fallback function if the import fails
    def create_welcome_card(title, icon, content):
        """Create a welcome card with consistent styling."""
        # Format content to ensure paragraphs are properly wrapped
        if not content.startswith('<p>'):
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            content = ''.join([f'<p>{p}</p>' for p in paragraphs])
        
        # Create concise HTML without extra whitespace and newlines
        return f"""<div class="card card-elevated content-card welcome-card large animate-fade-in"><div class="card-title"><span class="card-icon">{icon}</span> {title}</div><div class="card-content">{content}</div></div>"""

logger = logging.getLogger(__name__)

class HomeView(BaseView):
    """Home view for the application."""
    
    def __init__(self):
        """Initialize the home view."""
        super().__init__(
            view_id="home",
            title="Welcome to MNIST Digit Classifier", 
            icon="🏠"
        )
        self.logger = logging.getLogger(__name__)
    
    def render(self) -> None:
        """Render the home view content."""
        self.logger.debug("Rendering home view")
        try:
            # Apply common layout styling
            self.apply_common_layout()
            
            # Apply button styling from our CSS file
            from utils.css.button_css import load_button_css
            load_button_css()
            
            # Welcome card with consistent styling - this replaces the title
            welcome_content = """
            <b>Welcome to the MNIST Digit Classifier!</b>
            
            This application allows you to draw digits and have them recognized by a machine learning model.
            
            The model is trained on the MNIST dataset, which contains thousands of handwritten digit images.
            
            Draw any digit from 0-9 and let our AI predict what you've written!
            """
            welcome_card = create_welcome_card("MNIST Digit Classifier", "👋", welcome_content)
            st.markdown(welcome_card, unsafe_allow_html=True)
            
            # How it works section
            st.markdown("<h2>How It Works</h2>", unsafe_allow_html=True)
            
            # Use columns for the cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Draw digit feature
                card1 = ContentCard(
                    title="1. Draw a Digit",
                    icon="✏️",
                    content="<p>Navigate to the Drawing Canvas and use your mouse or touch to draw any digit from 0-9.</p><p>You can adjust the brush size and use the clear button to start over.</p>",
                    elevated=True,
                    size="small",  # Explicitly use small size for secondary color scheme
                    classes=["feature-card"]
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
                    icon="📖",
                    content="""
                    <p>All your predictions are saved to your session history.</p>
                    <p>You can view past predictions, compare results, and see statistics about your drawing classifications.</p>
                    """,
                    elevated=True,
                    classes=["feature-card", "secondary-card"]
                )
                card3.display()
            
            # Call to action section
            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

            # Use columns to place heading and button side by side
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown("<h2>Ready to try it out?</h2>", unsafe_allow_html=True)

            with col2:
                # Make sure this button is styled correctly through our button CSS
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