import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import io
import requests
import datetime
from utils.resource_loader import ResourceLoader

def render_drawing():
    """Render the drawing page with canvas and prediction."""
    # Load custom tab styles
    ResourceLoader.load_css(["css/components/tabs.css"])
    
    st.markdown("""
    <div class="content-card">
        <h1>Draw a Digit</h1>
        <p>Use the canvas below to draw a digit from 0-9, then click "Predict" to see the result.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a two-column layout for the main content
    left_col, right_col = st.columns(2)
    
    # Left column - Drawing panel
    with left_col:
        st.markdown("""
        <div class="content-card">
            <h2 style="text-align: center; margin-bottom: 1rem;">Input Methods</h2>
        """, unsafe_allow_html=True)
        
        # Define the tabs for different input methods
        input_tabs = st.tabs(["Draw", "Upload", "URL"])
        
        # First tab - Drawing Canvas
        with input_tabs[0]:
            # Canvas for drawing
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=st.session_state.brush_size,
                stroke_color=st.session_state.brush_color,
                background_color=st.session_state.get("canvas_bg", "#FFFFFF"),
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            # Store canvas result in session state for later use
            if canvas_result.image_data is not None:
                # Convert from RGBA to grayscale for the model
                img = canvas_result.image_data
                if img.shape[-1] == 4:  # If RGBA
                    # Keep only the alpha channel which has the drawing
                    alpha = img[:, :, 3]
                    # Create a black and white image from the alpha channel
                    bw_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    bw_img[alpha > 0] = 255
                    
                    # Store the processed image
                    st.session_state.canvas_result = bw_img
                    st.session_state.input_method = "draw"
                else:
                    st.session_state.canvas_result = img
                    st.session_state.input_method = "draw"
            
            # Buttons for interaction
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Canvas", key="clear_canvas"):
                    # This doesn't actually clear the canvas, but on next rerun it will be reset
                    st.session_state.canvas_result = None
                    st.session_state.prediction = None
                    st.session_state.confidence = None
            
            with col2:
                if st.button("Predict", key="predict_digit"):
                    process_prediction("draw")
        
        # Second tab - Upload Image
        with input_tabs[1]:
            uploaded_file = st.file_uploader("Upload an image of a digit", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                
                # Process the image
                try:
                    # Convert to grayscale and resize to match the canvas size
                    image = image.convert("L")  # Convert to grayscale
                    image = image.resize((300, 300))  # Resize to match canvas
                    
                    # Convert to numpy array
                    img_array = np.array(image)
                    
                    # Store the processed image
                    st.session_state.canvas_result = img_array
                    st.session_state.input_method = "upload"
                    
                    # Button to make prediction
                    if st.button("Predict", key="predict_upload"):
                        process_prediction("upload")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        # Third tab - URL Input
        with input_tabs[2]:
            url = st.text_input("Enter URL of a digit image", key="url_input")
            
            if url:
                try:
                    # Attempt to fetch and display the image
                    response = requests.get(url)
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        st.image(image, caption="Image from URL", width=300)
                        
                        # Process the image
                        image = image.convert("L")  # Convert to grayscale
                        image = image.resize((300, 300))  # Resize
                        
                        # Convert to numpy array
                        img_array = np.array(image)
                        
                        # Store the processed image
                        st.session_state.canvas_result = img_array
                        st.session_state.input_method = "url"
                        
                        # Button to make prediction
                        if st.button("Predict", key="predict_url"):
                            process_prediction("url")
                    else:
                        st.error(f"Failed to fetch image: {response.status_code}")
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Right column - Prediction result
    with right_col:
        st.markdown("""
        <div class="content-card">
            <h2 style="text-align: center; margin-bottom: 1rem;">Prediction Result</h2>
        """, unsafe_allow_html=True)
        
        if st.session_state.prediction is not None:
            # Display prediction result
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div class="prediction-result">{st.session_state.prediction}</div>
                <div class="confidence-badge">Confidence: {st.session_state.confidence:.2%}</div>
                <div style="margin-top: 2rem;">
                    <p>Was this prediction correct?</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes", key="feedback_yes"):
                    st.session_state.feedback_yes = True
                    st.session_state.feedback_no = False
                    if st.session_state.history:
                        # Set the actual value same as prediction
                        st.session_state.history[-1]["actual"] = st.session_state.prediction
                    st.success("Thank you for your feedback!")
            
            with col2:
                if st.button("👎 No", key="feedback_no"):
                    st.session_state.feedback_yes = False
                    st.session_state.feedback_no = True
            
            # If user says no, allow them to provide the correct label
            if st.session_state.get("feedback_no", False):
                st.markdown("<p>What was the correct digit?</p>", unsafe_allow_html=True)
                correct_digit = st.number_input("Correct digit", min_value=0, max_value=9, step=1)
                if st.button("Submit Correction", key="submit_correction"):
                    st.session_state.history[-1]["actual"] = correct_digit
                    st.success("Thank you for your feedback!")
        else:
            # Empty state
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px;">
                <div style="font-size: 4rem; color: #ccc; margin-bottom: 1rem;">🔍</div>
                <div style="text-align: center; color: #666;">
                    Draw a digit and click "Predict" to see the result
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True) 

def process_prediction(input_method):
    """Process the prediction based on the current canvas result."""
    if st.session_state.canvas_result is not None:
        # In a real app, you would process the image and make a prediction
        import random
        st.session_state.prediction = random.randint(0, 9)
        st.session_state.confidence = random.uniform(0.7, 0.99)
        
        # Save to history
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Convert numpy array to PIL Image for storage
        img_pil = Image.fromarray(st.session_state.canvas_result)
        
        # Add to history
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "image": img_pil,
            "prediction": st.session_state.prediction,
            "confidence": st.session_state.confidence,
            "actual": None,  # Will be set if user provides feedback
            "input_method": input_method  # Store the input method used
        })
    else:
        st.warning("Please provide an input image first.") 