import streamlit as st
from components.header import render_header
from components.footer import render_footer

# Configure the page
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render header
render_header()

# Main content container with some spacing after header
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Your main app content goes here
st.write("## Draw a digit below")
st.write("Use the canvas to draw a digit from 0-9")

# Add placeholder for your main content
st.write("Main content will go here")

# Ensure there's content to push the footer down
st.markdown("<div style='min-height: 50vh;'></div>", unsafe_allow_html=True)

# Render footer
render_footer()
