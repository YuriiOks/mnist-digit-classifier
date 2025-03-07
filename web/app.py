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
    initial_sidebar_state="expanded"
)

# Import our custom modules
from utils.theme_manager import ThemeManager
from components.header import render_header
from components.footer import render_footer
from components.sidebar import render_sidebar

# Import views
from views.home import render_home
from views.drawing import render_drawing
from views.history import render_history
from views.settings import render_settings

# Initialize theme
ThemeManager.initialize()
ThemeManager.load_theme_resources()

# Initialize session state variables
if 'canvas_result' not in st.session_state:
    st.session_state.canvas_result = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'brush_size' not in st.session_state:
    st.session_state.brush_size = 20
if 'brush_color' not in st.session_state:
    st.session_state.brush_color = "#000000"

# Render header (which also creates the theme toggle)
render_header()

# Render sidebar and get the selected menu item
selected = render_sidebar()

# Main content container with some spacing after header
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Display different content based on selected menu item
if selected == "Home":
    render_home()
elif selected == "Drawing":
    render_drawing()
elif selected == "History":
    render_history()
elif selected == "Settings":
    render_settings()

# Add spacer to push footer down
st.markdown("<div style='min-height: 50px;'></div>", unsafe_allow_html=True)

# Render footer
render_footer()