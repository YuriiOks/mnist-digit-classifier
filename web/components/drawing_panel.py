import streamlit as st
from streamlit_drawable_canvas import st_canvas

def render_drawing_panel():
    """Render the drawing canvas with tools and controls."""
    # Create tabs first
    tab1, tab2, tab3 = st.tabs(["Draw Digit", "Upload Image", "Image URL"])
    
    with tab1:
        # Move the instructions inside the tab with centered alignment
        st.markdown("""
        <div class="canvas-instructions-container">
            <p class="canvas-instructions">Draw a single digit (0-9) in the canvas below, or upload an image:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize canvas state for clearing
        if "canvas_cleared" not in st.session_state:
            st.session_state.canvas_cleared = False
        
        # Canvas key based on cleared state
        canvas_key = f"canvas_{st.session_state.canvas_cleared}"
        
        # Center-aligned canvas container
        st.markdown('<div class="canvas-center-container">', unsafe_allow_html=True)
        
        # Draw the canvas
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons in a centered container
        col1, col2 = st.columns(2)
        with col1:
            clear_button = st.button("🧹 Clear Canvas")
            if clear_button:
                st.session_state.canvas_cleared = not st.session_state.canvas_cleared
                st.experimental_rerun()
        with col2:
            predict_button = st.button("🔍 Predict Digit", type="primary")
    
    with tab2:
        st.file_uploader("Upload an image of a digit:", type=["jpg", "jpeg", "png"])
        
    with tab3:
        st.text_input("Enter URL of a digit image:")
    
    return canvas_result, clear_button, predict_button