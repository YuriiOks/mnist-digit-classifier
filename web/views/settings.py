import streamlit as st
from utils.theme_manager import ThemeManager

def render_settings():
    """Render the settings page."""
    st.markdown("""
    <div class="content-card">
        <h1>Settings</h1>
        <p>Configure application preferences and drawing options.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme settings
    render_theme_settings()
    
    # Canvas settings
    render_canvas_settings()
    
    # Application info
    render_app_info()

def render_theme_settings():
    """Render theme settings section."""
    st.markdown("""
    <div class="content-card">
        <h2>Theme Settings</h2>
    """, unsafe_allow_html=True)
    
    theme_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="theme_toggle_settings")
    
    # Check if theme toggle changed
    if theme_mode != st.session_state.dark_mode:
        ThemeManager.toggle_dark_mode()
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_canvas_settings():
    """Render canvas settings section."""
    st.markdown("""
    <div class="content-card">
        <h2>Canvas Settings</h2>
    """, unsafe_allow_html=True)
    
    # Brush size
    brush_size = st.slider("Brush Size", min_value=1, max_value=50, value=st.session_state.brush_size)
    if brush_size != st.session_state.brush_size:
        st.session_state.brush_size = brush_size
    
    # Brush color
    brush_color = st.color_picker("Brush Color", value=st.session_state.brush_color)
    if brush_color != st.session_state.brush_color:
        st.session_state.brush_color = brush_color
    
    # Preview
    st.markdown("<h3>Brush Preview</h3>", unsafe_allow_html=True)
    preview_size = min(brush_size * 2, 100)
    st.markdown(f"""
    <div style="margin: 20px auto; width: {preview_size}px; height: {preview_size}px; 
                border-radius: 50%; background-color: {brush_color}; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
    """, unsafe_allow_html=True)
    
    # Reset to defaults
    if st.button("Reset to Defaults"):
        st.session_state.brush_size = 20
        st.session_state.brush_color = "#000000"
        st.success("Canvas settings reset to defaults.")
        st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_app_info():
    """Render application information section."""
    st.markdown("""
    <div class="content-card">
        <h2>About This Application</h2>
        <p style="margin-top: 1rem;">
            This MNIST Digit Classifier uses a machine learning model trained on the MNIST dataset, 
            which contains 70,000 examples of handwritten digits.
        </p>
        <p style="margin-top: 0.5rem;">
            The application is built with Streamlit and uses a convolutional neural network (CNN) 
            for digit classification.
        </p>
        <div style="margin-top: 1.5rem;">
            <h3>Version Information</h3>
            <ul style="margin-left: 1.5rem; margin-top: 0.5rem;">
                <li>Application Version: 1.0.0</li>
                <li>Model Version: MNIST-CNN-v1</li>
                <li>Last Updated: March 2023</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True) 