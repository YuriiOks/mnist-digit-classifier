import streamlit as st
import time

def show_loading_animation(message="Processing..."):
    """Show an animated loading indicator with message."""
    loading_html = f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-message">{message}</div>
    </div>
    
    <style>
    .loading-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem 0;
    }}
    
    .loading-spinner {{
        width: 50px;
        height: 50px;
        border: 5px solid rgba(67, 97, 238, 0.2);
        border-radius: 50%;
        border-top-color: var(--accent-primary);
        animation: spin 1s ease-in-out infinite;
    }}
    
    .loading-message {{
        margin-top: 1rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    """
    
    # Create a placeholder for the loading animation
    placeholder = st.empty()
    placeholder.markdown(loading_html, unsafe_allow_html=True)
    
    return placeholder

def simulate_loading(placeholder, duration=1.0):
    """Simulate a loading process for the given duration."""
    time.sleep(duration)
    placeholder.empty() 