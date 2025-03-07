import streamlit as st
from utils.theme_manager import ThemeManager
from utils.resource_loader import ResourceLoader
from utils.settings_manager import SettingsManager

def render_settings():
    """Render the settings page with a 3-column grid layout."""
    # Load settings CSS, but use the improved ResourceLoader method
    # to ensure CSS is loaded as styles only once, not as content
    ResourceLoader.load_css([
        "css/components/settings/base.css",
        "css/components/settings/theme.css",
        "css/components/settings/canvas.css",
        "css/components/settings/app_info.css",
        "css/components/settings/grid.css"
    ])
    
    # Single spacing element with proper margin
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    
    # Main settings header
    header_html = ResourceLoader.load_template("views/settings/header.html")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Add spacing between settings header and content
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    # Create a 3-column grid layout
    col1, col2, col3 = st.columns(3)
    
    # Render theme settings in column 1
    with col1:
        render_theme_settings()
    
    # Render canvas settings in column 2  
    with col2:
        render_canvas_settings()
    
    # Render app info in column 3
    with col3:
        render_app_info()
    
    # Load settings JavaScript - use the improved method
    ResourceLoader.load_js(["settings/theme_toggle.js"])

def render_theme_settings():
    """Render theme settings section with improved toggle."""
    # Get theme settings
    theme_settings = SettingsManager.get_theme_settings()
    
    # Start settings card
    st.markdown("""
    <div class="settings-card">
    """, unsafe_allow_html=True)
    
    # Render card header
    header_html = ResourceLoader.load_template(
        "views/settings/sections/theme_card_header.html",
        TITLE=theme_settings["title"],
        ICON=theme_settings["icon"]
    )
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Start content section
    st.markdown("""
    <div class="settings-content">
    """, unsafe_allow_html=True)
    
    # Render current theme info
    current_theme_html = ResourceLoader.load_template(
        "views/settings/sections/current_theme_info.html",
        CURRENT_THEME=theme_settings["current_theme"]
    )
    st.markdown(current_theme_html, unsafe_allow_html=True)
    
    # Render theme toggle
    toggle_html = ResourceLoader.load_template(
        "views/settings/controls/theme_toggle.html",
        CHECKBOX_STATE=theme_settings["checkbox_state"],
        TOGGLE_LABEL=theme_settings["toggle_label"]
    )
    st.markdown(toggle_html, unsafe_allow_html=True)
    
    # Hidden button for theme toggle, activated by JavaScript
    if st.button("Toggle Theme", key="theme_toggle_settings", on_click=ThemeManager.toggle_dark_mode):
        pass
    
    # Close content and card divs
    st.markdown("""
    </div>
    </div>
    """, unsafe_allow_html=True)

def render_canvas_settings():
    """Render canvas settings section with improved visuals."""
    # Get canvas settings configuration
    canvas_config = SettingsManager.get_canvas_settings()
    
    # Start settings card
    st.markdown("""
    <div class="settings-card">
    """, unsafe_allow_html=True)
    
    # Render card header
    header_html = ResourceLoader.load_template(
        "views/settings/sections/theme_card_header.html",
        TITLE=canvas_config.get("title", "Canvas Settings"),
        ICON=canvas_config.get("icon", "🖌️")
    )
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Start content section
    st.markdown("""
    <div class="settings-content">
    """, unsafe_allow_html=True)
    
    # Brush size slider
    brush_size_config = canvas_config.get("brushSize", {})
    brush_size = st.slider(
        brush_size_config.get("label", "Brush Size"),
        min_value=brush_size_config.get("min", 1),
        max_value=brush_size_config.get("max", 50),
        step=brush_size_config.get("step", 1),
        value=st.session_state.brush_size
    )
    if brush_size != st.session_state.brush_size:
        st.session_state.brush_size = brush_size
    
    # Brush color picker
    brush_color = st.color_picker("Brush Color", value=st.session_state.brush_color)
    if brush_color != st.session_state.brush_color:
        st.session_state.brush_color = brush_color
    
    # Preview
    preview_size = min(brush_size * 2, 100)
    preview_html = ResourceLoader.load_template(
        "views/settings/controls/brush_preview.html",
        PREVIEW_TITLE=canvas_config.get("previewTitle", "Brush Preview"),
        PREVIEW_SIZE=str(preview_size),
        BRUSH_COLOR=brush_color
    )
    st.markdown(preview_html, unsafe_allow_html=True)
    
    # Reset button
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 1.5rem;">
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Reset to Defaults", key="reset_canvas", use_container_width=True):
        SettingsManager.reset_canvas_to_defaults()
        st.success("Canvas settings reset to defaults.")
        st.rerun()
    
    # Close content and card divs
    st.markdown("""
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

def render_app_info():
    """Render application information with improved layout."""
    # Get app info configuration
    app_info = SettingsManager.get_app_info()
    versions = app_info.get("versions", {})
    
    # Start settings card
    st.markdown("""
    <div class="settings-card">
    """, unsafe_allow_html=True)
    
    # Render card header
    header_html = ResourceLoader.load_template(
        "views/settings/sections/theme_card_header.html",
        TITLE=app_info.get("title", "About This Application"),
        ICON=app_info.get("icon", "ℹ️")
    )
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Start content section
    st.markdown("""
    <div class="settings-content">
    """, unsafe_allow_html=True)
    
    # App info content
    app_info_html = ResourceLoader.load_template(
        "views/settings/sections/app_info_content.html",
        DESCRIPTION=app_info.get("description", ""),
        TECH_STACK=app_info.get("techStack", ""),
        APP_VERSION=versions.get("app", "1.0.0"),
        MODEL_VERSION=versions.get("model", "MNIST-CNN-v1"),
        LAST_UPDATED=versions.get("lastUpdated", "March 2025")
    )
    st.markdown(app_info_html, unsafe_allow_html=True)
    
    # Close content and card divs
    st.markdown("""
    </div>
    </div>
    """, unsafe_allow_html=True) 