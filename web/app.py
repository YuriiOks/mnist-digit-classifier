import streamlit as st
import sys
import os

# Add the current directory to the Python path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Configure the page
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our custom modules - Move imports after path setup
from utils.theme_manager import ThemeManager
from components.header import render_header
from components.footer import render_footer

# Initialize theme
ThemeManager.initialize()
ThemeManager.load_theme_resources()
ThemeManager.create_theme_toggle()

# Render header
render_header()

# Main content container with some spacing after header
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Main app content
st.write("## Draw a digit below")
st.write("Use the canvas to draw a digit from 0-9")
st.write("Main content will go here")

# Add spacer to push footer down
st.markdown("<div style='min-height: 50vh;'></div>", unsafe_allow_html=True)

# Render footer
render_footer()
