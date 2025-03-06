# Updated drawing_panel.py with image upload/URL options
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import requests
from io import BytesIO
import numpy as np

def render_drawing_panel():
    """Render the drawing canvas with tools and controls."""
    # Center-aligned instructions with custom CSS
    st.markdown("""
    <div class="canvas-container">
        <p class="canvas-instructions">Draw a single digit (0-9) in the canvas below, or upload an image:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Draw Digit", "Upload Image", "Image URL"])
    
    with tab1:
        # Initialize canvas state for clearing
        if "canvas_cleared" not in st.session_state:
            st.session_state.canvas_cleared = False
        
        # Canvas key based on cleared state
        canvas_key = f"canvas_{st.session_state.canvas_cleared}"
        
        # Container to center the canvas
        st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
        
        # Draw the canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=canvas_key,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons - in centered container
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            clear_button = st.button("🧹 Clear Canvas")
            if clear_button:
                st.session_state.canvas_cleared = not st.session_state.canvas_cleared
                st.experimental_rerun()
        with col2:
            predict_button = st.button("🔍 Predict Digit", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # For tracking which input method is active
    image_source = None
    uploaded_image = None
    
    with tab2:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image of a digit:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Image", width=280)
                
                # Store the image for prediction
                uploaded_image = image
                image_source = "upload"
                
                # Prediction button for uploaded image
                predict_upload_button = st.button("🔍 Predict Uploaded Image", type="primary")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="url-container">', unsafe_allow_html=True)
        url = st.text_input("Enter URL of a digit image:")
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content)).convert("L")
                
                # Display the image from URL
                st.image(image, caption="Image from URL", width=280)
                
                # Store the image for prediction
                uploaded_image = image
                image_source = "url"
                
                # Prediction button for URL image
                predict_url_button = st.button("🔍 Predict URL Image", type="primary")
            except Exception as e:
                st.error(f"Error loading image from URL: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Determine which predict button is active
    predict_active = False
    if tab1._active and predict_button:
        predict_active = True
    elif tab2._active and 'predict_upload_button' in locals() and predict_upload_button:
        predict_active = True
    elif tab3._active and 'predict_url_button' in locals() and predict_url_button:
        predict_active = True
    
    # Return canvas result or uploaded image
    if image_source in ["upload", "url"] and uploaded_image is not None:
        # Convert PIL image to format compatible with canvas result
        img_array = np.array(uploaded_image)
        if img_array.ndim == 2:  # If grayscale, add channel dimension
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.repeat(img_array, 3, axis=2)  # Make it RGB
        
        # Create a dict similar to canvas_result
        result = {
            "image_data": img_array,
            "json_data": {},
        }
        return result, False, predict_active
    else:
        return canvas_result, clear_button, predict_active