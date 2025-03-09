# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/home/features.py
# Description: Features section component for the home page
# Created: 2024-05-01

import streamlit as st
import logging
from typing import List, Dict, Any

from ui.components.cards import ContentCard
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class FeaturesSection:
    """Features section for the home page.
    
    Displays the key features of the application using cards.
    """
    
    def __init__(self):
        """Initialize the features section component."""
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.logger.debug("Entering __init__")
        
        try:
            self.theme_manager = ThemeManager()
            
            self.features = [
                {
                    "title": "Draw Digits",
                    "icon": "✏️",
                    "content": """
                    <p>Use our interactive canvas to draw digits with your mouse 
                    or touchscreen.</p>
                    <p>Perfect for testing how the model responds to your 
                    handwriting style.</p>
                    """
                },
                {
                    "title": "Upload Images",
                    "icon": "🖼️",
                    "content": """
                    <p>Upload image files containing handwritten digits directly 
                    from your device.</p>
                    <p>Supports common image formats like PNG, JPG, and BMP.</p>
                    """
                },
                {
                    "title": "Instant Classification",
                    "icon": "🔍",
                    "content": """
                    <p>Get immediate predictions after drawing or uploading an image.</p>
                    <p>See confidence scores for each possible digit classification.</p>
                    """
                },
                {
                    "title": "Track History",
                    "icon": "📊",
                    "content": """
                    <p>View a history of your past predictions and their accuracy.</p>
                    <p>Compare different drawings and see how the model performed.</p>
                    """
                }
            ]
            self.logger.debug(f"Initialized with {len(self.features)} feature cards")
        except Exception as e:
            self.logger.error(f"Error initializing FeaturesSection: {str(e)}", exc_info=True)
            raise
            
        self.logger.debug("Exiting __init__")
    
    def display(self) -> None:
        """Display the features section."""
        self.logger.debug("Entering display")
        try:
            # Add explicit styles for feature cards
            st.markdown("""
            <style>
            .feature-card {
                height: 100%;
                margin-bottom: 1.5rem;
                transition: all 0.3s ease;
                border-radius: var(--border-radius, 8px);
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: var(--shadow-lg);
            }
            
            .feature-card .card-content p {
                margin-bottom: 0.75rem;
            }
            
            /* Ensure the feature grid looks good */
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("## Key Features")
            
            # Create two columns for feature cards
            col1, col2 = st.columns(2)
            
            # First row
            with col1:
                self.logger.debug(f"Displaying feature card: {self.features[0]['title']}")
                feature_card = ContentCard(
                    title=self.features[0]["title"],
                    icon=self.features[0]["icon"],
                    content=self.features[0]["content"],
                    elevated=True,
                    classes=["feature-card"]
                )
                feature_card.display()
                
            with col2:
                self.logger.debug(f"Displaying feature card: {self.features[1]['title']}")
                feature_card = ContentCard(
                    title=self.features[1]["title"],
                    icon=self.features[1]["icon"],
                    content=self.features[1]["content"],
                    elevated=True,
                    classes=["feature-card"]
                )
                feature_card.display()
            
            # Second row
            with col1:
                self.logger.debug(f"Displaying feature card: {self.features[2]['title']}")
                feature_card = ContentCard(
                    title=self.features[2]["title"],
                    icon=self.features[2]["icon"],
                    content=self.features[2]["content"],
                    elevated=True,
                    classes=["feature-card"]
                )
                feature_card.display()
                
            with col2:
                self.logger.debug(f"Displaying feature card: {self.features[3]['title']}")
                feature_card = ContentCard(
                    title=self.features[3]["title"],
                    icon=self.features[3]["icon"],
                    content=self.features[3]["content"],
                    elevated=True,
                    classes=["feature-card"]
                )
                feature_card.display()
                
            self.logger.debug("All feature cards displayed successfully")
        except Exception as e:
            self.logger.error(f"Error displaying features: {str(e)}", exc_info=True)
            st.error("Failed to display feature cards")
            
        self.logger.debug("Exiting display")
    
    def get_features(self) -> List[Dict[str, Any]]:
        """Get the list of features.
        
        Returns:
            List[Dict[str, Any]]: List of feature dictionaries
        """
        self.logger.debug("Entering get_features")
        try:
            self.logger.debug(f"Returning {len(self.features)} features")
            return self.features
        except Exception as e:
            self.logger.error(f"Error getting features: {str(e)}", exc_info=True)
            return []
        finally:
            self.logger.debug("Exiting get_features")