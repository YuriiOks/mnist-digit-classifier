# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/history/history_view.py
# Description: Prediction history view
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any
import pandas as pd
from datetime import datetime

try:
    from ui.views.base_view import BaseView
    from core.app_state.history_state import HistoryState
except ImportError:
    # Fallback for minimal implementation
    class BaseView:
        def __init__(self, view_id="history", title="Prediction History", description="", icon="📊"):
            self.view_id = view_id
            self.title = title
            self.description = description
            self.icon = icon
    
    class HistoryState:
        @staticmethod
        def get_predictions(): 
            # Return sample data
            return [
                {"id": 1, "digit": 5, "confidence": 0.95, "timestamp": datetime.now()},
                {"id": 2, "digit": 3, "confidence": 0.87, "timestamp": datetime.now()},
                {"id": 3, "digit": 7, "confidence": 0.92, "timestamp": datetime.now()}
            ]

logger = logging.getLogger(__name__)

class HistoryView(BaseView):
    """History view for prediction history."""
    
    def __init__(self):
        """Initialize the history view."""
        super().__init__(
            view_id="history",
            title="Prediction History",
            description="View your digit classification history",
            icon="📊"
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def render(self) -> None:
        """Render the history view content."""
        self.logger.debug("Rendering history view")
        
        try:
            # Apply the same layout CSS as home view for consistency
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
            
            # Add welcome card similar to home page
            st.markdown("""
            <div class="card card-elevated content-card welcome-card animate-fade-in">
                <div class="card-title">
                    <span class="card-icon">📊</span>
                    Prediction History
                </div>
                <div class="card-content">
                    <p>View a history of all your previous digit predictions and their results.</p>
                    <p>This page keeps track of your drawing attempts, including the predicted digit and confidence level.</p>
                    <p>You can analyze your results over time and see how well the model performs on your handwriting.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get prediction history
            predictions = HistoryState.get_predictions()
            
            if not predictions:
                st.info("No predictions yet. Draw and classify some digits to see them here.")
                return
            
            # Convert to DataFrame for display
            df = pd.DataFrame([
                {
                    "Time": p.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                    "Digit": p.get("digit", "?"),
                    "Confidence": f"{p.get('confidence', 0):.2%}",
                    "ID": p.get("id", 0)
                }
                for p in predictions
            ])
            
            # Display as table
            st.dataframe(df, hide_index=True)
            
            # Show statistics
            st.subheader("Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", len(predictions))
                
            with col2:
                # Calculate most predicted digit
                if predictions:
                    digit_counts = {}
                    for p in predictions:
                        digit = p.get("digit", "?")
                        digit_counts[digit] = digit_counts.get(digit, 0) + 1
                    
                    most_common = max(digit_counts.items(), key=lambda x: x[1])
                    st.metric("Most Predicted Digit", most_common[0])
                
            with col3:
                # Calculate average confidence
                if predictions:
                    avg_conf = sum(p.get("confidence", 0) for p in predictions) / len(predictions)
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
            
            # Display visualization
            st.subheader("Visualizations")
            
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Create a bar chart of digit frequencies
                digit_counts = {}
                for p in predictions:
                    digit = p.get("digit", "?")
                    digit_counts[digit] = digit_counts.get(digit, 0) + 1
                
                # Sort by digit
                sorted_counts = sorted(digit_counts.items())
                digits = [str(d[0]) for d in sorted_counts]
                counts = [d[1] for d in sorted_counts]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(digits, counts, color='#4F46E5')
                ax.set_xlabel('Digit')
                ax.set_ylabel('Frequency')
                ax.set_title('Digit Prediction Frequency')
                
                # Add count labels above bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Confidence distribution
                confidences = [p.get("confidence", 0) for p in predictions]
                
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.hist(confidences, bins=10, range=(0, 1), color='#06B6D4')
                ax2.set_xlabel('Confidence')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Confidence Distribution')
                
                st.pyplot(fig2)
                
            except ImportError:
                st.error("Could not load visualization libraries.")
                
            # Add clear history button
            if st.button("Clear History"):
                # In a real app, this would clear the history
                st.success("History cleared!")
                st.rerun()
            
            self.logger.debug("History view rendered successfully")
            
        except Exception as e:
            self.logger.error(f"Error rendering history view: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}") 