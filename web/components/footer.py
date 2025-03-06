import streamlit as st

def render_footer():
    """Render the application footer with credits only - no links."""
    footer_html = """
    <div class="footer">
        <div class="footer-content">
            <p>MNIST Digit Classifier | Developed by YuriODev</p>
            <p class="copyright">© 2025 All rights reserved</p>
        </div>
    </div>
    """
    
    st.markdown(footer_html, unsafe_allow_html=True) 