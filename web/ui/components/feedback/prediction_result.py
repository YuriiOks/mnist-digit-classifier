# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/components/feedback/prediction_result.py
# Description: Prediction result display component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Callable, Optional, List, Union

from ui.components.base.component import Component
from core.app_state.history_state import HistoryState

logger = logging.getLogger(__name__)


class PredictionResult(Component):
    """Prediction result display component."""

    def __init__(
        self,
        prediction: Optional[Dict[str, Any]] = None,
        on_correction: Optional[Callable[[int], None]] = None,
        classes: List[str] = None,
        attributes: Dict[str, str] = None,
    ):
        """Initialize prediction result component.

        Args:
            prediction: Prediction data dictionary
            on_correction: Callback for user correction
            classes: Additional CSS classes
            attributes: Additional HTML attributes
        """
        super().__init__(classes=classes, attributes=attributes)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.prediction = prediction
        self.on_correction = on_correction

    def render(self) -> None:
        """Render the prediction result component."""
        self.logger.debug("Rendering prediction result")

        try:
            # Custom CSS
            st.markdown(
                """
            <style>
            .prediction-result {
                margin-bottom: 2rem;
                border-radius: 8px;
                padding: 1.5rem;
                background-color: var(--color-bg-secondary, #f8f9fa);
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
            
            .prediction-header {
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
                font-weight: 600;
                color: var(--color-text-primary, #333);
            }
            
            .prediction-digit {
                font-size: 3rem;
                font-weight: 700;
                margin-right: 1rem;
                color: var(--color-primary, #6366F1);
            }
            
            .prediction-confidence {
                font-size: 1.25rem;
                color: var(--color-text-secondary, #666);
            }
            
            .prediction-feedback {
                display: flex;
                align-items: center;
                margin-top: 1rem;
            }
            
            .feedback-correct {
                color: var(--color-success, #10B981);
                font-size: 1.5rem;
                margin-right: 0.5rem;
            }
            
            .feedback-incorrect {
                color: var(--color-danger, #EF4444);
                font-size: 1.5rem;
                margin-right: 0.5rem;
            }
            
            .prediction-correction {
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid var(--color-border, #e5e7eb);
            }
            
            .prediction-distribution {
                margin-top: 1.5rem;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # Get current prediction if not provided
            if not self.prediction:
                self.prediction = HistoryState.get_current_prediction()

            # Display prediction result
            if self.prediction:
                prediction_data = self.prediction.get("prediction", {})
                digit = prediction_data.get("digit")
                confidence = prediction_data.get("confidence", 0)
                probabilities = prediction_data.get("probabilities", {})

                # Create container
                with st.container():
                    # Title
                    st.subheader("Prediction Result")

                    # Result card
                    st.markdown(
                        f"""
                        <div class="prediction-result">
                            <div class="prediction-header">
                                <div class="prediction-digit">{digit}</div>
                                <div class="prediction-confidence">
                                    Confidence: {confidence:.2%}
                                </div>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # User correction UI
                    st.markdown(
                        "<div class='prediction-correction'>",
                        unsafe_allow_html=True,
                    )
                    st.write("Is this correct?")

                    # Thumbs up/down buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("👍 Yes", key="thumbs_up"):
                            st.success("Great! Thank you for the feedback.")

                    with col2:
                        if st.button("👎 No", key="thumbs_down"):
                            # Show correction input
                            st.markdown(
                                "<div class='correction-input'>",
                                unsafe_allow_html=True,
                            )
                            st.write("What's the correct digit?")

                            # Create a row of digit buttons
                            digit_cols = st.columns(10)
                            for i in range(10):
                                with digit_cols[i]:
                                    if st.button(str(i), key=f"digit_{i}"):
                                        if self.on_correction:
                                            self.on_correction(i)
                                        else:
                                            # Default behavior
                                            entry_id = self.prediction.get("id")
                                            if entry_id:
                                                HistoryState.set_user_correction(
                                                    entry_id, i
                                                )
                                                st.success(
                                                    f"Corrected to {i}. Thank you!"
                                                )
                                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Probability distribution
                    if probabilities:
                        st.markdown(
                            "<div class='prediction-distribution'>",
                            unsafe_allow_html=True,
                        )
                        st.write("Probability Distribution")

                        # Convert to list for chart
                        probs = [probabilities.get(str(i), 0) for i in range(10)]

                        # Bar chart
                        st.bar_chart({"Probability": probs})
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(
                    "No prediction available. Draw a digit or upload an image and click 'Predict'."
                )

        except Exception as e:
            self.logger.error(
                f"Error rendering prediction result: {str(e)}", exc_info=True
            )
            st.error(f"An error occurred while rendering prediction results: {str(e)}")
