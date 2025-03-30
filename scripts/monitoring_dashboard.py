# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: scripts/monitoring_dashboard.py
# Description: Simple performance monitoring dashboard
# Created: 2025-03-26
# Updated: 2025-03-30

import os
import sys
import json
import requests
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import argparse
import logging
from collections import deque

# Set up page config
st.set_page_config(
    page_title="MNIST Classifier - Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Function to encode image for API
def encode_image(image):
    """Encode image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Function to make a prediction
def predict_digit(image, api_url):
    """Make a prediction using the API."""
    # Encode image
    base64_img = encode_image(image)

    # Send request
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"image": base64_img},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None


# Function to get metrics
def get_metrics(api_url):
    """Get metrics from the API."""
    try:
        response = requests.get(f"{api_url}/metrics", timeout=5)

        if response.status_code == 200:
            return response.json().get("metrics", {})
        else:
            st.error(f"Error getting metrics: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None


# Function to check health
def check_health(api_url):
    """Check health of the API."""
    try:
        response = requests.get(f"{api_url}/health", timeout=2)

        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False


# Main function
def main():
    """
    Main function to run the Streamlit app.
    This function sets up the sidebar, tabs, and handles user interactions.
    It includes features for live testing, performance metrics, and request history.
    """
    # Sidebar
    st.sidebar.title("📊 MNIST Classifier Dashboard")

    # API URL
    api_url = st.sidebar.text_input("API URL", value="http://localhost:5000")

    # Check if API is available
    api_available = check_health(api_url)

    if api_available:
        st.sidebar.success("✅ API is online")
    else:
        st.sidebar.error("❌ API is offline")

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Live Testing", "📈 Performance Metrics", "🔄 History"]
    )

    # Tab 1: Live Testing
    with tab1:
        st.title("🔍 Live Testing")

        # Two columns: Input and Results
        col1, col2 = st.columns(2)

        with col1:
            st.header("Input")

            # Canvas for drawing
            from streamlit_drawable_canvas import st_canvas

            # Specify canvas parameters
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=20,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )

            # Add a button to predict
            predict_button = st.button("Predict Digit")

            # Add a button to clear canvas
            clear_button = st.button("Clear Canvas")

            if clear_button:
                # This doesn't directly clear the canvas, but forces a rerun
                # which effectively resets the canvas
                st.rerun()

        with col2:
            st.header("Results")

            if predict_button and canvas_result.image_data is not None:
                # Get image data from canvas
                image_data = canvas_result.image_data

                # Convert to PIL image
                pil_image = Image.fromarray(image_data.astype("uint8")).convert("L")

                # Make prediction
                with st.spinner("Predicting..."):
                    result = predict_digit(pil_image, api_url)

                if result:
                    prediction = result.get("prediction")
                    confidence = result.get("confidence")

                    # Display results
                    st.markdown(f"### Prediction: **{prediction}**")
                    st.progress(confidence)
                    st.markdown(f"Confidence: **{confidence:.2%}**")

                    # Display timing information
                    preprocessing_time = result.get("preprocessing_time_ms", 0)
                    inference_time = result.get("inference_time_ms", 0)

                    st.markdown("### Timing:")
                    st.markdown(f"- Preprocessing: **{preprocessing_time:.2f} ms**")
                    st.markdown(f"- Inference: **{inference_time:.2f} ms**")
                    st.markdown(
                        f"- Total: **{preprocessing_time + inference_time:.2f} ms**"
                    )

    # Tab 2: Performance Metrics
    with tab2:
        st.title("📈 Performance Metrics")

        if api_available:
            # Get metrics
            metrics = get_metrics(api_url)

            if metrics:
                # Display request count
                request_count = metrics.get("request_count", 0)
                st.markdown(f"### Total Requests: **{request_count}**")

                # Create columns for metrics
                col1, col2, col3 = st.columns(3)

                # Preprocessing time metrics
                with col1:
                    st.markdown("### Preprocessing Time (ms)")
                    preproc_metrics = metrics.get("preprocessing_time_ms", {})
                    st.markdown(f"- Avg: **{preproc_metrics.get('avg', 0):.2f} ms**")
                    st.markdown(f"- Min: **{preproc_metrics.get('min', 0):.2f} ms**")
                    st.markdown(f"- Max: **{preproc_metrics.get('max', 0):.2f} ms**")
                    st.markdown(f"- P95: **{preproc_metrics.get('p95', 0):.2f} ms**")

                # Inference time metrics
                with col2:
                    st.markdown("### Inference Time (ms)")
                    inference_metrics = metrics.get("inference_time_ms", {})
                    st.markdown(f"- Avg: **{inference_metrics.get('avg', 0):.2f} ms**")
                    st.markdown(f"- Min: **{inference_metrics.get('min', 0):.2f} ms**")
                    st.markdown(f"- Max: **{inference_metrics.get('max', 0):.2f} ms**")
                    st.markdown(f"- P95: **{inference_metrics.get('p95', 0):.2f} ms**")

                # Total time metrics
                with col3:
                    st.markdown("### Total Time (ms)")
                    total_metrics = metrics.get("total_time_ms", {})
                    st.markdown(f"- Avg: **{total_metrics.get('avg', 0):.2f} ms**")
                    st.markdown(
                        f"- Throughput: **{1000/total_metrics.get('avg', 1):.2f} img/s**"
                    )

                # Add auto-refresh button
                auto_refresh = st.checkbox("Auto-refresh (every 5s)")

                if auto_refresh:
                    st.empty()
                    time.sleep(5)
                    st.rerun()
            else:
                st.warning("No metrics available. Make a few predictions first.")
        else:
            st.error("API is offline. Cannot fetch metrics.")

    # Tab 3: History
    with tab3:
        st.title("🔄 Request History")

        # If we've stored history in session state, display it
        if "history" not in st.session_state:
            st.session_state.history = []

        # Display history
        if st.session_state.history:
            # Create a dataframe
            history_df = pd.DataFrame(st.session_state.history)

            # Display the dataframe
            st.dataframe(history_df)

            # Create a plot of prediction times
            if "timestamp" in history_df.columns and len(history_df) > 1:
                st.markdown("### Prediction Time Trend")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(
                    history_df["timestamp"],
                    history_df["total_time_ms"],
                    marker="o",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Total Time (ms)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            st.info("No history available. Make predictions to see history.")


# Run the app
if __name__ == "__main__":
    main()