import streamlit as st
from streamlit_drawable_canvas import st_canvas

def render_drawing_panel():
    """Render the drawing canvas with tools and controls."""
    # Simple instructions
    st.markdown("Draw a single digit (0-9) in the canvas below:")
    
    # Initialize canvas state for clearing
    if "canvas_cleared" not in st.session_state:
        st.session_state.canvas_cleared = False
    
    # Canvas key based on cleared state
    canvas_key = f"canvas_{st.session_state.canvas_cleared}"
    
    # Draw the canvas directly
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=canvas_key,
    )
    
    # Action buttons - simplified and styled to match your screenshot
    col1, col2 = st.columns(2)
    with col1:
        clear_button = st.button("🧹 Clear Canvas")
        if clear_button:
            st.session_state.canvas_cleared = not st.session_state.canvas_cleared
            st.experimental_rerun()
    with col2:
        predict_button = st.button("🔍 Predict Digit", type="primary")
    
    return canvas_result, clear_button, predict_button 