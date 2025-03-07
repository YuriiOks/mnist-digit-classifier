import streamlit as st
from utils.resource_loader import ResourceLoader
from utils.theme_manager import ThemeManager

def load_theme_specific_css():
    """Load theme-specific CSS based on current theme."""
    theme = "dark" if st.session_state.dark_mode else "light"
    ResourceLoader.load_css([
        "css/views/home/base.css",
        f"css/themes/{theme}/home.css"
    ])

def render_welcome_section():
    """Render the welcome section of the home page."""
    welcome_html = ResourceLoader.load_template("views/home/welcome_card.html")
    st.markdown(welcome_html, unsafe_allow_html=True)

def render_feature_card(icon: str, title: str, description: str):
    """Render a feature card with the given content."""
    feature_html = ResourceLoader.load_template(
        "views/home/feature_card.html",
        ICON=icon,
        TITLE=title,
        DESCRIPTION=description
    )
    st.markdown(feature_html, unsafe_allow_html=True)

def render_features_section():
    """Render the features section with three columns."""
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("✏️", "Draw a Digit", "Use the interactive canvas to draw any digit from 0-9 and see the prediction in real-time."),
        ("🔍", "High Accuracy", "Our model is trained on thousands of handwritten digit samples for high prediction accuracy."),
        ("📊", "Track History", "Review your previous predictions and track the model's performance over time.")
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            render_feature_card(icon, title, desc)

def render_how_to_use():
    """Render the how-to-use section."""
    how_to_html = ResourceLoader.load_template("views/home/how_to_use.html")
    st.markdown(how_to_html, unsafe_allow_html=True)

def render_home():
    """Render the home page with all its sections."""
    # Load theme-specific CSS
    load_theme_specific_css()
    
    # Render all sections
    render_welcome_section()
    render_features_section()
    render_how_to_use() 