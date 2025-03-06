import os
import sys
import streamlit as st

# Add the current directory to the Python path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Configure the page
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import our custom modules
from utils.theme_manager import ThemeManager
from components.header import render_header
from components.footer import render_footer

# Initialize theme
ThemeManager.initialize()
ThemeManager.load_theme_resources()

# Render header (which also creates the theme toggle)
render_header()

# Main content container with some spacing after header
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Create a two-column layout for the main content
left_col, right_col = st.columns(2)

# Left column - Drawing panel
with left_col:
    st.markdown(
        "<h2 style='text-align: center;'>Draw a digit</h2>", 
        unsafe_allow_html=True
    )
    st.write("Use the canvas below to draw a digit from 0-9")
    
    # Placeholder for canvas component (we'll implement this later)
    st.markdown("""
    <div style="
        border: 2px dashed #ccc; 
        border-radius: 5px; 
        height: 300px; 
        display: flex; 
        justify-content: center; 
        align-items: center;
        margin-bottom: 20px;
    ">
        Canvas component will go here
    </div>
    """, unsafe_allow_html=True)
    
    # Buttons for interaction
    col1, col2 = st.columns(2)
    with col1:
        st.button("Clear Canvas", key="clear_canvas")
    with col2:
        st.button("Predict", key="predict_button")

# Right column - Prediction panel
with right_col:
    st.markdown(
        "<h2 style='text-align: center;'>Prediction</h2>", 
        unsafe_allow_html=True
    )
    st.write("Draw a digit on the canvas and click 'Predict'")
    
    # Placeholder for prediction result
    st.markdown("""
    <div style="
        border: 2px dashed #ccc; 
        border-radius: 5px; 
        height: 300px; 
        display: flex; 
        justify-content: center; 
        align-items: center;
        flex-direction: column;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px; margin-bottom: 10px;">
            Prediction result will appear here
        </div>
        <div style="font-size: 16px; color: #666;">
            Draw a digit and click "Predict"
        </div>
    </div>
    """, unsafe_allow_html=True)

# History section
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h3>Prediction History</h3>", unsafe_allow_html=True)
st.markdown("""
<div style="
    border: 2px dashed #ccc; 
    border-radius: 5px; 
    height: 200px; 
    display: flex; 
    justify-content: center; 
    align-items: center;
    margin-bottom: 20px;
">
    No prediction history yet
</div>
""", unsafe_allow_html=True)

# Add spacer to push footer down
st.markdown("<div style='min-height: 50px;'></div>", unsafe_allow_html=True)

# Render footer
render_footer()