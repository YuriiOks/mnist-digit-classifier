import streamlit as st
import os
import sys
import datetime

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

# Base styles to ensure full width and proper spacing
st.markdown("""
<style>
/* Force full width on containers */
.block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1rem !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Arial', sans-serif;
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* Hide toggle button and any extra elements */
div.row-widget.stButton, button[kind="secondary"] {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
    position: absolute !important;
    top: -9999px !important;
    left: -9999px !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Animation */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Initialize dark mode in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    """Toggle dark mode state and trigger page rerun"""
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

# Create the hidden toggle button
with st.container():
    dark_mode_toggle = st.button(
        "🔄", 
        key="dark_mode_toggle", 
        on_click=toggle_dark_mode
    )

# Header styles
st.markdown("""
<style>
/* Header styles */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(to right, #2c3e50, #4ca1af);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    width: 100%;
    position: relative;
}
.header-logo-title {
    display: flex;
    align-items: center;
    margin: 0 auto;
}
.header-logo {
    font-size: 2rem;
    margin-right: 10px;
    animation: pulse 2s infinite ease-in-out;
}
.header-title {
    color: white;
    font-size: 1.8rem;
    font-weight: 600;
    margin: 0;
}

/* Dark mode toggle button */
.dark-mode-toggle {
    position: absolute;
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    font-size: 1.2rem;
    z-index: 1000;
}
.dark-mode-toggle:hover {
    transform: translateY(-50%) scale(1.1);
    background: rgba(0, 0, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Render Header
st.markdown(f"""
<div class="header-container">
    <div class="header-logo-title">
        <div class="header-logo">✏️</div>
        <div class="header-title">MNIST Digit Classifier</div>
    </div>
    <div class="dark-mode-toggle" id="darkModeToggle" 
         onclick="toggleDarkMode()">
        {" 🌙" if st.session_state.dark_mode else " ☀️"}
    </div>
</div>

<script>
    function toggleDarkMode() {{
        // Find the Streamlit button and click it programmatically
        const buttons = document.querySelectorAll('button');
        for (let button of buttons) {{
            if (button.innerText === '🔄') {{
                button.click();
                break;
            }}
        }}
    }}
</script>
""", unsafe_allow_html=True)

# Apply dark mode styles if enabled
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    .header-container {
        background: linear-gradient(to right, #1a1a2e, #16213e);
    }
    .footer-container {
        background: linear-gradient(to right, #1a1a2e, #16213e);
    }
    .dark-mode-toggle {
        background: rgba(255, 255, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Main content container with some spacing after header
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Main app content
st.write("## Draw a digit below")
st.write("Use the canvas to draw a digit from 0-9")
st.write("Main content will go here")

# Add spacer to push footer down
st.markdown("<div style='min-height: 50vh;'></div>", unsafe_allow_html=True)

# Footer styles
st.markdown("""
<style>
.footer-container {
    background: linear-gradient(to right, #2c3e50, #4ca1af);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    margin-top: 50px;
    margin-bottom: 20px;
    box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.05);
    width: 100%;
}
.footer-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
}
.footer-info {
    display: flex;
    flex-direction: column;
    margin-bottom: 10px;
}
.project-name {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 5px;
}
.developer-info {
    font-size: 0.9rem;
    opacity: 0.9;
}
.developer-info a {
    color: #a1ffce;
    text-decoration: none;
    font-weight: bold;
    transition: color 0.3s ease;
}
.developer-info a:hover {
    text-decoration: underline;
    color: white;
}
.copyright-info {
    font-size: 0.9rem;
    opacity: 0.8;
}
@media (max-width: 576px) {
    .footer-content {
        flex-direction: column;
        text-align: center;
    }
    .footer-info {
        margin-bottom: 15px;
    }
}
</style>
""", unsafe_allow_html=True)

# Render Footer
current_year = datetime.datetime.now().year
st.markdown(f"""
<div class="footer-container">
    <div class="footer-content">
        <div class="footer-info">
            <div class="project-name">MNIST Digit Classifier</div>
            <div class="developer-info">
                Developed by <a href="https://github.com/YuriiOks" 
                target="_blank">YuriiOks</a>
            </div>
        </div>
        <div class="copyright-info">
            © {current_year} All Rights Reserved
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
