# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/views/draw_view.py
# Description: Draw view implementation
# Created: 2025-03-17
# Updated: 2025-03-30 (Refactored to use db_manager)

import streamlit as st
import logging
import time
import random
import base64
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import io
from datetime import datetime  # Added for timestamp

# Removed psycopg2 import

from ui.views.base_view import View
from ui.components.cards.card import WelcomeCard, FeatureCard

# Removed PredictionResult import as it doesn't call DB directly
from ui.components.inputs.canvas import DrawingCanvas
from ui.components.inputs.file_upload import ImageUpload
from ui.components.inputs.url_input import ImageUrlInput
from core.app_state.canvas_state import CanvasState
from core.app_state.history_state import (
    HistoryState,
)  # Still used for managing current pred state if desired
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils

# Removed prediction_service import
from core.database.db_manager import db_manager  # Import db_manager


class DrawView(View):
    """Draw view for the MNIST Digit Classifier application."""

    def __init__(self):
        """Initialize the draw view."""
        super().__init__(
            name="draw",
            title="Digit Recognition",
            description="Choose a method to input a digit for recognition.",
        )
        self.show_header = False  # Use welcome card instead
        self.__logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def __load_view_data(
        self,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Load necessary JSON data for the view."""
        welcome_data = resource_manager.load_json_resource(
            "draw/welcome_card.json"
        ) or {
            "title": "Draw & Recognize Digits",
            "icon": "✏️",
            "content": "Welcome to the drawing tool! Choose a method to input a digit.",
        }
        # Ensure tabs_data is a list, even if loading fails
        tabs_data = resource_manager.load_json_resource("draw/tabs.json") or []
        tips_data = resource_manager.load_json_resource("draw/tips.json") or {
            "title": "Tips for Best Results",
            "items": ["Draw clearly", "Center digit", "Use thick stroke"],
        }
        return welcome_data, tabs_data, tips_data

    def __initialize_session_state(self) -> None:
        """Initialize session state variables for the draw view."""
        defaults = {
            "active_tab": "draw",
            "prediction_made": False,
            "prediction_correct": None,
            "show_correction": False,
            "predicted_digit": None,
            "confidence": None,
            "canvas_key": "canvas_initial",
            "file_upload_key": "file_uploader_initial",
            "url_input_key": "url_input_initial",
            "reset_counter": 0,
            "current_db_id": None,  # Store the latest DB ID (integer)
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def __render_welcome_card(self, welcome_data: Dict[str, Any]) -> None:
        """Render the welcome card."""
        WelcomeCard(
            title=welcome_data.get("title", "Draw & Recognize Digits"),
            content=welcome_data.get("content", "Welcome to the drawing tool!"),
            icon=welcome_data.get("icon", "✏️"),
        ).display()

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def __render_feature_card(self, tab_data: Dict[str, Any]) -> None:
        """Render the feature card for the current tab."""
        if not tab_data:  # Handle case where tabs_data might be empty list
            self.__logger.warning("No tab data provided for feature card.")
            return
        FeatureCard(
            title=tab_data.get("title", "Input Method"),
            content=tab_data.get("content", "Select an input method."),
            icon=tab_data.get("icon", "❓"),
        ).display()

    def __render_tab_buttons(self) -> None:
        """Render tab selection buttons."""
        tab_cols = st.columns(3)
        tabs = [
            ("draw", "✏️ Draw Digit"),
            ("upload", "📤 Upload Image"),
            ("url", "🔗 Enter URL"),
        ]
        for i, (tab_id, label) in enumerate(tabs):
            with tab_cols[i]:
                if st.button(
                    label,
                    key=f"tab_{tab_id}",
                    type=(
                        "primary"
                        if st.session_state.active_tab == tab_id
                        else "secondary"
                    ),
                    use_container_width=True,
                ):
                    st.session_state.active_tab = tab_id
                    # Reset prediction state when switching tabs
                    st.session_state.prediction_made = False
                    st.session_state.current_db_id = None
                    CanvasState.clear_all()  # Clear image data on tab switch
                    st.rerun()

    def __render_draw_tab(self, draw_data) -> None:
        """Render the draw digit tab content."""
        tab_cols = st.columns([2, 3])  # Adjust column ratio
        with tab_cols[0]:
            self.__render_feature_card(draw_data)
            stroke_width = st.slider(
                "Brush Width",
                10,
                25,
                15,
                key=f"stroke_{st.session_state.reset_counter}",
            )

        with tab_cols[1]:
            # Instructions
            st.markdown(
                """
            <div class="canvas-instructions" style="text-align: center; margin-bottom: 1rem; font-size: 0.9rem; color: var(--color-text-muted);">
                Draw a single digit (0-9) clearly in the center.
            </div>
            """,
                unsafe_allow_html=True,
            )

            from streamlit_drawable_canvas import st_canvas  # Import locally

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color="#000000",  # Consider making this theme-aware later
                background_color="#FFFFFF",  # Consider making this theme-aware later
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=st.session_state.canvas_key,  # Use dynamic key for reset
                display_toolbar=True,
            )
            # Store canvas result if drawing occurred
            if (
                canvas_result.image_data is not None
                and canvas_result.image_data.shape[0] > 0
            ):  # Check if data exists
                # Check if it's not just an empty/cleared canvas (check mean intensity)
                mean_intensity = canvas_result.image_data[
                    :, :, :3
                ].mean()  # Check RGB channels mean
                if (
                    mean_intensity < 250
                ):  # Heuristic: If not almost pure white, assume drawing exists
                    try:
                        img_array = canvas_result.image_data
                        # Convert RGBA to L (Grayscale) correctly
                        img = Image.fromarray(img_array).convert("L")
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_bytes = buffer.getvalue()

                        # Only update state if image data is different from current
                        current_data = CanvasState.get_image_data()
                        if current_data != img_bytes:
                            CanvasState.set_image_data(img_bytes)
                            CanvasState.set_input_type(CanvasState.CANVAS_INPUT)
                            st.session_state["canvas_drawing_detected"] = True
                            self.__logger.debug("Canvas image data updated.")
                    except Exception as e:
                        self.__logger.error(
                            f"Error processing canvas image: {e}", exc_info=True
                        )
                        st.warning("Could not process drawing from canvas.")

    def __render_upload_tab(self, upload_data) -> None:
        """Render the upload image tab content."""
        tab_cols = st.columns([2, 3])
        with tab_cols[0]:
            self.__render_feature_card(upload_data)
        with tab_cols[1]:
            uploaded_file = st.file_uploader(
                "Upload an image of a handwritten digit",
                type=["png", "jpg", "jpeg"],
                key=st.session_state.file_upload_key,  # Use dynamic key
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image Preview", width=280)
                    uploaded_file.seek(0)
                    img_bytes = uploaded_file.getvalue()
                    # Only update state if different
                    if CanvasState.get_image_data() != img_bytes:
                        CanvasState.set_image_data(img_bytes)
                        CanvasState.set_input_type(CanvasState.UPLOAD_INPUT)
                        st.session_state["upload_detected"] = True
                        self.__logger.debug("Upload image data updated.")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    self.__logger.error(f"Error processing upload: {e}", exc_info=True)

    def __render_url_tab(self, url_data) -> None:
        """Render the URL input tab content."""
        import requests  # Keep import local

        tab_cols = st.columns([2, 3])
        with tab_cols[0]:
            self.__render_feature_card(url_data)
        with tab_cols[1]:
            url = st.text_input(
                "Image URL",
                key=st.session_state.url_input_key,  # Use dynamic key
                placeholder="https://example.com/digit.png",
                label_visibility="collapsed",
            )
            # Keep track of URL entered to only load on button click
            st.session_state["entered_url"] = url

            if st.button("Load Image from URL", key="load_url_image", disabled=not url):
                entered_url = st.session_state.get("entered_url")
                if not entered_url:
                    st.warning("Please enter a URL first.")
                    return
                try:
                    with st.spinner("Loading image from URL..."):
                        response = requests.get(
                            entered_url, timeout=10, stream=True
                        )  # Use stream=True for large files
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "").lower()

                        # More robust image type check
                        allowed_types = [
                            "image/png",
                            "image/jpeg",
                            "image/jpg",
                            "image/gif",
                            "image/webp",
                        ]
                        if not any(content_type.startswith(t) for t in allowed_types):
                            st.error(
                                f"URL does not point to a valid image type (found: {content_type})."
                            )
                            return

                        img_bytes = response.content
                        # Verify it's actually image data
                        try:
                            image = Image.open(io.BytesIO(img_bytes))
                            image.verify()  # Verify structure
                            # Re-open after verify
                            image = Image.open(io.BytesIO(img_bytes))
                            st.image(image, caption="Image from URL Preview", width=280)

                            # Store validated image data
                            # Re-save as PNG bytes for consistency
                            buffer = io.BytesIO()
                            image.convert("L").save(
                                buffer, format="PNG"
                            )  # Convert to grayscale PNG
                            final_img_bytes = buffer.getvalue()

                            CanvasState.set_image_data(final_img_bytes)
                            CanvasState.set_input_type(CanvasState.URL_INPUT)
                            st.session_state["url_image_loaded"] = True
                            self.__logger.debug("URL image data updated.")
                            st.success("Image loaded successfully!")
                            # Don't rerun here, let user click predict

                        except (IOError, SyntaxError) as img_err:
                            st.error("Failed to process the image data from the URL.")
                            self.__logger.error(
                                f"URL image processing error: {img_err}"
                            )

                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching image: Check URL and connection.")
                    self.__logger.error(f"URL fetch error: {e}", exc_info=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    self.__logger.error(f"Image load error: {e}", exc_info=True)

            # Display preview if already loaded via URL method AND input type is URL
            elif CanvasState.get_input_type() == CanvasState.URL_INPUT:
                img_bytes = CanvasState.get_image_data()
                if img_bytes:
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, caption="Loaded Image Preview", width=280)
                    except Exception as e:
                        self.__logger.warning(
                            f"Failed to display cached URL image: {e}"
                        )
                        CanvasState.clear_all()  # Clear potentially corrupted data

    def __render_action_buttons(self) -> None:
        """Render action buttons (Clear, Predict)."""
        button_cols = st.columns(2)
        with button_cols[0]:
            if st.button(
                "Clear All", key="clear_all", type="secondary", use_container_width=True
            ):
                st.session_state.prediction_made = False
                st.session_state.current_db_id = None  # Clear stored DB ID
                timestamp = int(time.time() * 1000)  # Use timestamp for uniqueness
                st.session_state.canvas_key = f"canvas_{timestamp}"
                st.session_state.file_upload_key = f"file_uploader_{timestamp}"
                st.session_state.url_input_key = f"url_input_{timestamp}"
                # Clear the entered URL state as well
                if "entered_url" in st.session_state:
                    del st.session_state["entered_url"]
                st.session_state.reset_counter += 1
                CanvasState.clear_all()
                self.__logger.info("Cleared all inputs and prediction state.")
                st.rerun()
        with button_cols[1]:
            # Disable predict button if no image data is available
            predict_disabled = CanvasState.get_image_data() is None
            if st.button(
                "Predict",
                key="predict",
                type="primary",
                use_container_width=True,
                disabled=predict_disabled,
            ):
                image_data = CanvasState.get_image_data()
                # Re-check just in case state is weird
                if image_data is None:
                    st.error(
                        "No image data available. Please draw, upload or load an image first."
                    )
                    return

                st.info(
                    f"Processing image data: {len(image_data)} bytes from {CanvasState.get_input_type()}"
                )
                try:
                    with st.spinner("🤖 Analyzing digit..."):
                        # Import classifier locally inside the function
                        from model.digit_classifier import DigitClassifier

                        classifier = DigitClassifier()
                        predicted_digit, confidence = classifier.predict(image_data)

                        # Update session state for UI display
                        st.session_state.predicted_digit = predicted_digit
                        st.session_state.confidence = confidence
                        st.session_state.prediction_made = True
                        st.session_state.show_correction = False
                        st.session_state.prediction_correct = None

                        input_type = CanvasState.get_input_type()
                        encoded_image = base64.b64encode(image_data).decode("utf-8")

                        # Log prediction to database using db_manager
                        db_id = db_manager.add_prediction(
                            {
                                "digit": predicted_digit,
                                "confidence": confidence,
                                "input_type": input_type,
                                "image_data": encoded_image,  # Pass base64 string
                                "timestamp": datetime.now(),
                                "true_label": None,  # Initially no true label
                            }
                        )

                        if db_id is not None:
                            st.session_state.current_db_id = (
                                db_id  # Store the integer DB ID
                            )
                            self.__logger.info(f"Prediction logged with DB ID: {db_id}")
                        else:
                            st.error("Failed to log prediction to database.")
                            st.session_state.current_db_id = (
                                None  # Ensure it's None on failure
                            )

                    st.success(
                        f"Predicted digit: {predicted_digit} with {confidence:.2%} confidence"
                    )
                    # Rerun to update the prediction display area cleanly
                    st.rerun()

                except (
                    ConnectionError
                ) as e:  # Specific error for model connection issues
                    st.error(f"Model Service Error: {str(e)}")
                    self.__logger.error(f"Connection error during prediction: {str(e)}")
                except ValueError as e:  # Specific error for image processing issues
                    st.error(f"Image Processing Error: {str(e)}")
                    self.__logger.error(f"Value error during prediction: {str(e)}")
                except Exception as e:  # Catch-all for unexpected errors
                    st.error(f"An unexpected error occurred during prediction.")
                    self.__logger.error(
                        f"Unexpected error in prediction: {str(e)}", exc_info=True
                    )

    def __render_prediction_result(self) -> None:
        """Render prediction result and feedback options if a prediction has been made."""
        if st.session_state.get("prediction_made"):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display the predicted digit
                predicted_digit = st.session_state.get("predicted_digit", "?")
                st.markdown(
                    f"""<div style="text-align: center; font-size: 8rem; font-weight: bold; color: var(--color-primary); line-height: 1;">
                    {predicted_digit}</div>""",
                    unsafe_allow_html=True,
                )

            with col2:
                # Display confidence
                confidence = st.session_state.get("confidence", 0.0)
                st.markdown("<h4>Confidence</h4>", unsafe_allow_html=True)
                confidence_pct = f"{confidence * 100:.1f}%"
                # Use Streamlit's progress bar which handles theming
                st.progress(max(0.01, confidence), text=confidence_pct)

                # Confidence message
                confidence_level = (
                    "Low"
                    if confidence < 0.6
                    else ("Medium" if confidence < 0.9 else "High")
                )
                st.markdown(
                    f"<p>The model has <b>{confidence_level} confidence</b> in this prediction.</p>",
                    unsafe_allow_html=True,
                )

                # Feedback section
                st.markdown("<h4>Is this correct?</h4>", unsafe_allow_html=True)
                feedback_cols = st.columns(2)
                unique_suffix = f"{st.session_state.reset_counter}_{st.session_state.get('current_db_id', 'none')}"

                with feedback_cols[0]:
                    if st.button(
                        "👍 Yes", key=f"yes_{unique_suffix}", use_container_width=True
                    ):
                        st.session_state.prediction_correct = True
                        st.session_state.show_correction = False
                        db_id = st.session_state.get("current_db_id")
                        pred_digit = st.session_state.get("predicted_digit")
                        if db_id is not None and pred_digit is not None:
                            try:
                                success = db_manager.update_prediction(
                                    db_id, {"true_label": pred_digit}
                                )
                                if success:
                                    self.__logger.info(
                                        f"Logged confirmation for DB ID: {db_id}"
                                    )
                                    st.success("Thank you for confirming!")
                                else:
                                    st.warning("Could not save feedback.")
                            except Exception as e:
                                self.__logger.error(f"Error saving 'Yes' feedback: {e}")
                                st.error("Failed to save feedback.")
                        # Rerun to potentially disable feedback buttons after click
                        time.sleep(1)  # Brief pause to show message
                        st.rerun()

                with feedback_cols[1]:
                    if st.button(
                        "👎 No", key=f"no_{unique_suffix}", use_container_width=True
                    ):
                        st.session_state.prediction_correct = False
                        st.session_state.show_correction = True
                        # Rerun to show the correction input
                        st.rerun()

            # Correction Input (only shown if 'No' was clicked)
            if st.session_state.get("show_correction"):
                st.markdown(
                    "<h4>What's the correct digit?</h4>", unsafe_allow_html=True
                )
                digit_cols = st.columns(10)
                for i in range(10):
                    with digit_cols[i]:
                        if st.button(str(i), key=f"correct_{i}_{unique_suffix}"):
                            db_id = st.session_state.get("current_db_id")
                            if db_id is not None:
                                try:
                                    success = db_manager.update_prediction(
                                        db_id, {"true_label": i}
                                    )
                                    if success:
                                        self.__logger.info(
                                            f"Logged correction for DB ID {db_id} to {i}"
                                        )
                                        st.success(
                                            f"Thank you! Recorded correct digit as {i}."
                                        )
                                    else:
                                        st.warning("Could not save correction.")
                                except Exception as e:
                                    self.__logger.error(f"Error saving correction: {e}")
                                    st.error("Failed to save correction.")
                            else:
                                st.warning(
                                    "Cannot save correction: Prediction ID not found."
                                )

                            st.session_state.show_correction = False
                            # Rerun to hide correction buttons and update display
                            time.sleep(1)  # Brief pause
                            st.rerun()

            # Probability Distribution Placeholder (if needed later)
            # if st.session_state.get("probabilities"):
            #    st.markdown("<h4>Probability Distribution (Placeholder)</h4>", unsafe_allow_html=True)
            #    # Add bar chart here if you get probabilities from your model API

    def __render_tip_card(self, tips_data: Dict[str, Any]) -> None:
        """Render the tips card."""
        if tips_data and tips_data.get("items"):
            items = tips_data.get("items", [])
            # Format as an unordered list for better HTML structure
            list_items = "".join([f"<li>{tip}</li>" for tip in items])
            content = f"<ul style='padding-left: 20px; margin: 0;'>{list_items}</ul>"
            FeatureCard(
                title=tips_data.get("title", "💡 Tips"),
                content=content,
                icon="",  # Icon is part of title now
            ).display()

    def render(self) -> None:
        """Render the draw view content."""
        self.__initialize_session_state()
        welcome_data, tabs_data, tips_data = self.__load_view_data()

        self.__render_welcome_card(welcome_data)
        self.__render_tab_buttons()
        st.markdown("<hr>", unsafe_allow_html=True)

        active_tab_data = {}
        if tabs_data:  # Ensure tabs_data is not empty
            active_tab_id = st.session_state.active_tab
            # Find the data for the active tab
            active_tab_data = next(
                (t for t in tabs_data if t.get("id") == active_tab_id),
                tabs_data[0] if tabs_data else {},
            )

        if st.session_state.active_tab == "draw":
            self.__render_draw_tab(active_tab_data)
        elif st.session_state.active_tab == "upload":
            self.__render_upload_tab(active_tab_data)
        elif st.session_state.active_tab == "url":
            self.__render_url_tab(active_tab_data)

        st.markdown("<hr>", unsafe_allow_html=True)
        self.__render_action_buttons()
        self.__render_prediction_result()
        st.markdown("<hr>", unsafe_allow_html=True)
        self.__render_tip_card(tips_data)
