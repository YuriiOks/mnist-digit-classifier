import streamlit as st
from utils.theme_manager import ThemeManager
from utils.resource_loader import ResourceLoader

def render_settings():
    """Render the settings page."""
    # Load settings CSS
    ResourceLoader.load_css(["css/components/settings.css"])
    
    # Main settings header
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
    """Render theme settings section with an improved toggle."""
    # Prepare template variables
    current_theme = "Dark Mode" if st.session_state.dark_mode else "Light Mode"
    checkbox_state = "checked" if st.session_state.dark_mode else ""
    toggle_label = "Switch to Light Mode" if st.session_state.dark_mode else "Switch to Dark Mode"
    
    # Load and render the theme settings template
    theme_settings_html = ResourceLoader.load_template(
        "views/settings/theme_settings.html",
        CURRENT_THEME=current_theme,
        CHECKBOX_STATE=checkbox_state,
        TOGGLE_LABEL=toggle_label
    )
    
    st.markdown(theme_settings_html, unsafe_allow_html=True)
    
    # Hidden button for theme toggle, activated by JavaScript
    if st.button("Toggle Theme", key="theme_toggle_settings", on_click=ThemeManager.toggle_dark_mode):
        pass
    
    # Add JavaScript to connect checkbox to the hidden button
    st.markdown("""
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const checkbox = document.getElementById('themeCheckbox');
            if (checkbox) {
                checkbox.addEventListener('change', function() {
                    // Find and click the hidden button
                    const buttons = Array.from(document.querySelectorAll('button'));
                    const themeButton = buttons.find(button => 
                        button.innerText.includes('Toggle Theme')
                    );
                    if (themeButton) {
                        themeButton.click();
                    }
                });
            }
        });
    </script>
    """, unsafe_allow_html=True)

def render_canvas_settings():
    """Render canvas settings section with improved visuals."""
    # First part with Streamlit controls
    st.markdown("""
    <div class="settings-card">
        <div class="settings-header">
            <h2>Canvas Settings</h2>
            <div class="settings-icon">🖌️</div>
        </div>
        <div class="settings-content">
    """, unsafe_allow_html=True)
    
    # Brush size slider
    brush_size = st.slider("Brush Size", min_value=1, max_value=50, value=st.session_state.brush_size)
    if brush_size != st.session_state.brush_size:
        st.session_state.brush_size = brush_size
    
    # Brush color picker
    brush_color = st.color_picker("Brush Color", value=st.session_state.brush_color)
    if brush_color != st.session_state.brush_color:
        st.session_state.brush_color = brush_color
    
    # Preview
    preview_size = min(brush_size * 2, 100)
    st.markdown(f"""
    <div class="brush-preview-container">
        <h3>Brush Preview</h3>
        <div class="brush-preview" style="width: {preview_size}px; height: {preview_size}px; background-color: {brush_color};"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset button (styled better)
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 1.5rem;">
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Reset to Defaults", key="reset_canvas", use_container_width=True):
        st.session_state.brush_size = 20
        st.session_state.brush_color = "#000000"
        st.success("Canvas settings reset to defaults.")
        st.rerun()
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_app_info():
    """Render application information with improved layout."""
    # Load app info from template
    app_info_html = ResourceLoader.load_template(
        "views/settings/app_info_settings.html",
        APP_VERSION="1.0.0",
        MODEL_VERSION="MNIST-CNN-v1",
        LAST_UPDATED="March 2023"
    )
    
    st.markdown(app_info_html, unsafe_allow_html=True) 