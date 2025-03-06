import streamlit as st
from datetime import datetime
from utils.db import log_prediction
import numpy as np
from PIL import Image
import io
import base64
import requests
import time
import random

def get_image_from_canvas(canvas_data):
    """Convert canvas data to PIL Image."""
    if canvas_data is not None and canvas_data.image_data is not None:
        # Get image data from the canvas
        img_data = canvas_data.image_data
        # Convert to PIL Image
        return Image.fromarray(img_data.astype(np.uint8))
    return None

def predict_digit(image):
    """Send image to model service and get prediction."""
    if image is None:
        return None, None
        
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Send to model service
        response = requests.post(
            f"{st.session_state.model_url}/predict",
            json={"image": img_str},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["prediction"], result["confidence"]
        else:
            st.error(f"Error from model service: {response.text}")
            return None, None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def render_celebration_animation():
    """Render a custom celebration animation with CSS."""
    # Generate random confetti pieces
    confetti_html = """
    <div class="celebration-container">
    """
    
    # Add 50 confetti pieces with random colors, sizes and animations
    colors = ["#ff718d", "#fdff6a", "#ffcf4b", "#f7c0bb", "#621055", "#e900ff", "#00f2ff"]
    
    for i in range(50):
        color = random.choice(colors)
        left = random.randint(0, 100)
        size = random.randint(5, 15)
        delay = random.random() * 5  # 0-5 second delay
        
        confetti_html += f"""
        <div class="confetti" style="left: {left}%; width: {size}px; height: {size}px; 
                                     background-color: {color}; animation-delay: {delay}s;"></div>
        """
    
    confetti_html += """
    </div>
    
    <style>
    .celebration-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 9999;
    }
    
    .confetti {
        position: absolute;
        top: -20px;
        background-color: #f00;
        border-radius: 50%;
        animation: fall 5s ease-out forwards;
    }
    
    @keyframes fall {
        0% {
            transform: translateY(0) rotate(0deg);
            opacity: 1;
        }
        80% {
            opacity: 1;
        }
        100% {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
        }
    }
    </style>
    """
    
    return confetti_html

def render_prediction_panel(canvas_result, predict_button):
    """Render the prediction display with simple styling to match screenshot."""
    # Empty state - show placeholder with instructions
    if not predict_button:
        st.markdown(
            """
            <div class="empty-prediction">
                <div class="empty-icon">✏️</div>
                <div class="empty-text">Draw a digit on the left panel and click 'Predict Digit' to see the result</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
        
    # Get image from canvas
    image = get_image_from_canvas(canvas_result)
    
    if image is None:
        st.warning("Please draw a digit first!")
        return
        
    # Process the prediction
    with st.spinner('Analyzing your drawing...'):
        digit, confidence = predict_digit(image)
    
    if digit is not None:
        # Store prediction in session state for feedback
        if "current_prediction" not in st.session_state:
            st.session_state.current_prediction = {}
        st.session_state.current_prediction = {
            "digit": digit,
            "confidence": confidence
        }
        
        # Display the prediction
        st.markdown(f"""
        <div class="prediction-result">
            <div class="predicted-digit">{digit}</div>
            <div class="confidence">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple feedback form
        with st.form(key="feedback_form"):
            true_label = st.number_input(
                "Enter correct digit (0-9):",
                min_value=0,
                max_value=9,
                value=digit,
                step=1
            )
            
            submit_button = st.form_submit_button("Submit Feedback")
            
            if submit_button:
                try:
                    log_prediction(
                        st.session_state.db_connection,
                        digit,
                        true_label,
                        confidence
                    )
                    
                    # Add to session history
                    if "digit_history" not in st.session_state:
                        st.session_state.digit_history = []
                        
                    st.session_state.digit_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "prediction": digit,
                        "true_label": true_label,
                        "confidence": confidence
                    })
                    
                    # Enhanced celebration for correct predictions
                    if int(digit) == int(true_label):
                        # Show built-in balloons
                        st.balloons()
                        
                        # Show confetti animation
                        st.markdown(render_celebration_animation(), unsafe_allow_html=True)
                        
                        # Show success message with larger text and emoji
                        st.markdown("""
                        <div style="text-align: center; padding: 20px; background-color: #d4edda; 
                                    border-radius: 10px; margin: 20px 0;">
                            <h1 style="color: #28a745; font-size: 24px;">🎉 CORRECT! 🎉</h1>
                            <p style="font-size: 18px;">Great job! Your prediction was right!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Play a sound effect (note: this is a bit of a hack since Streamlit doesn't directly support audio)
                        st.markdown("""
                        <audio autoplay>
                            <source src="data:audio/mpeg;base64,SUQzBAAAAAABEVRYWFgAAAAtAAADY29tbWVudABCaWdTb3VuZEJhbmsuY29tIC8gTGFTb25vdGhlcXVlLm9yZwBURU5DAAAAHQAAA1N3aXRjaCBQbHVzIMKpIE5DSCBTb2Z0d2FyZQBUSVQyAAAABgAAAzIyMzUAVFNTRQAAAA8AAANMYXZmNTcuODMuMTAwAAAAAAAAAAAAAAD/80DEAAAAA0gAAAAATEFNRTMuMTAwVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/zQsRbAAADSAAAAABVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/zQMSkAAADSAAAAABVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" type="audio/mpeg">
                        </audio>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"Thanks for the feedback! The model predicted {digit} but the correct digit was {true_label}.")
                    
                    # Give user time to see the celebration before rerunning
                    time.sleep(2.5)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.error("Failed to get prediction. Please try again.") 