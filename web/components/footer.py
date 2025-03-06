import streamlit as st
import datetime
from utils.resource_loader import ResourceLoader

def render_footer():
    """Render the application footer."""
    current_year = datetime.datetime.now().year
    footer_html = ResourceLoader.load_template("footer.html", CURRENT_YEAR=current_year)
    st.markdown(footer_html, unsafe_allow_html=True)
