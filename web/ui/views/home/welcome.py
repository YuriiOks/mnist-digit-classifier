# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/home/welcome.py
# Description: Welcome section component for the home page
# Created: 2024-05-01

import streamlit as st
import logging

from ui.components.cards import ContentCard
from ui.components.controls import PrimaryButton
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class WelcomeSection:
    """Welcome section for the home page.
    
    Displays a welcome message and brief application description with a call to 
    action.
    """
    
    def __init__(self):
        """Initialize the welcome section component."""
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.logger.debug("Entering __init__")
        
        try:
            self.theme_manager = ThemeManager()
            self.logger.debug("ThemeManager initialized successfully")
        except Exception as e:
            self.logger.error(
                f"Error initializing ThemeManager: {str(e)}", 
                exc_info=True
            )
            raise
            
        self.logger.debug("Exiting __init__")
    
    def display(self) -> None:
        """Display the welcome section."""
        self.logger.debug("Entering display")
        try:
            # Main welcome content
            st.markdown("## Welcome to MNIST Digit Classifier")
            
            self.logger.debug("Creating welcome card")
            welcome_card = ContentCard(
                title="Welcome to MNIST Digit Classifier",
                content="""
                <p>This application demonstrates handwritten digit recognition using 
                a machine learning model trained on the MNIST dataset.</p>
                <p>Draw digits or upload images to see the model in action!</p>
                """,
                elevated=True,
                classes=["welcome-card"]
            )
            welcome_card.display()
            self.logger.debug("Welcome card displayed successfully")
            
            # Quick start guide
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("""
                Ready to try it out? Click the button to navigate to the Drawing 
                Canvas and start classifying digits!
                """)
            
            with col2:
                self.logger.debug("Creating drawing navigation button")
                draw_btn = PrimaryButton(
                    label="Start Drawing",
                    icon="✏️",
                    on_click=self._navigate_to_drawing
                )
                draw_btn.display()
                self.logger.debug("Drawing button displayed successfully")
                
            self.logger.debug("Welcome section displayed successfully")
        except Exception as e:
            self.logger.error(
                f"Error displaying welcome section: {str(e)}", 
                exc_info=True
            )
            st.error("An error occurred while displaying the welcome section")
            
        self.logger.debug("Exiting display")
    
    def _navigate_to_drawing(self) -> None:
        """Navigate to the drawing view."""
        self.logger.debug("Entering _navigate_to_drawing")
        try:
            from core.app_state.navigation_state import NavigationState
            
            self.logger.info("Navigating to drawing view")
            NavigationState.set_active_view("draw")
            st.rerun()
        except Exception as e:
            self.logger.error(
                f"Error navigating to drawing view: {str(e)}", 
                exc_info=True
            )
            # Can't raise because this is called from a callback
            st.error("Failed to navigate to drawing view")
            
        self.logger.debug("Exiting _navigate_to_drawing")