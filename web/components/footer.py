import streamlit as st
import datetime

def render_footer():
    """Render the application footer with attribution and copyright."""
    
    current_year = datetime.datetime.now().year
    
    # Footer styling
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
    
    # Force a bit of space before the footer
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    
    # Footer content
    st.markdown(f"""
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-info">
                <div class="project-name">MNIST Digit Classifier</div>
                <div class="developer-info">
                    Developed by <a href="https://github.com/YuriiOks" target="_blank">YuriiOks</a>
                </div>
            </div>
            <div class="copyright-info">
                © {current_year} All Rights Reserved
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
