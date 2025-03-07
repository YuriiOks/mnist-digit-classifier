import streamlit as st
from datetime import datetime
from utils.resource_loader import ResourceLoader

def render_footer():
    """Render the application footer."""
    # Get the current year
    current_year = datetime.now().year
    
    # Check if we have a footer template
    try:
        # Try to load the footer template
        footer_html = ResourceLoader.load_template(
            "footer.html", 
            CURRENT_YEAR=str(current_year)
        )
    except:
        # If no template exists, create a simple footer
        footer_html = f"""
        <footer class="app-footer">
            <div class="container">
                <div class="copyright-info">
                    © {current_year} All Rights Reserved
                </div>
            </div>
        </footer>
        <style>
        .app-footer {{
            margin-top: 2rem;
            padding: 1rem 0;
            border-top: 1px solid var(--card-border);
            text-align: center;
            font-size: 0.9rem;
            color: var(--text-color);
            opacity: 0.8;
        }}
        .copyright-info {{
            margin-top: 0.5rem;
        }}
        </style>
        """
    
    # Render the footer
    st.markdown(footer_html, unsafe_allow_html=True)
