import streamlit as st
import os
import sys
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from components.drawing_panel import render_drawing_panel
    from components.prediction_panel import render_prediction_panel
    from components.history_panel import render_history_panel
    from utils.db import get_connection
except Exception as e:
    st.error(f"Error importing modules: {str(e)}")
    logger.error(f"Import error: {str(e)}", exc_info=True)

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "model_url" not in st.session_state:
    st.session_state.model_url = os.environ.get("MODEL_URL", "http://model:5000")
if "digit_history" not in st.session_state:
    st.session_state.digit_history = []
if "db_connection" not in st.session_state:
    try:
        st.session_state.db_connection = get_connection()
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        st.session_state.db_connection = None
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def main():
    # Header with theme toggle
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("✏️ MNIST Digit Classifier")
    with col2:
        if st.button("🌙" if st.session_state.theme == "light" else "☀️"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.experimental_rerun()
    
    # Inject CSS - simpler approach
    with open(os.path.join(os.path.dirname(__file__), "static", "styles.css"), "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # Main content with columns - simplified without the divider
    left_col, right_col = st.columns(2)
    
    # Left column - Drawing panel with centered heading
    with left_col:
        st.markdown('<div class="column-container">', unsafe_allow_html=True)
        st.markdown("<h2>Draw a digit</h2>", unsafe_allow_html=True)
        canvas_result, clear_button, predict_button = render_drawing_panel()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column - Prediction panel with centered heading
    with right_col:
        st.markdown('<div class="column-container">', unsafe_allow_html=True)
        st.markdown("<h2>Prediction</h2>", unsafe_allow_html=True)
        render_prediction_panel(canvas_result, predict_button)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom section for history
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<h3>Prediction History</h3>", unsafe_allow_html=True)
    render_history_panel(st.session_state.db_connection)
    
    # Footer
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "MNIST Digit Classifier | Developed by YuriODev | © 2025 All rights reserved", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 