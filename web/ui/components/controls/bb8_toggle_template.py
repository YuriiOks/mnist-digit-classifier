# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/bb8_toggle_template.py
# Description: HTML template for BB8 toggle component with two-phase animation
# Created: 2025-03-17

"""HTML template for the BB8 toggle component."""

import uuid  # Add this import

# The complete HTML template for the BB8 toggle component
BB8_TOGGLE_TEMPLATE = """
<!-- BB8 toggle wrapper with unique ID -->
<div style="text-align:center; padding-top:20px; padding-bottom:20px;">
  <!-- Clickable area that triggers animation -->
  <a href="#" id="bb8-toggle" style="display:inline-block; text-decoration:none;">
  <a href="#" class="bb8-toggle-clickable" onclick="animateBB8Toggle('{wrapper_id}', '{current_theme}'); return false;">
    <label class="bb8-toggle">
      <input class="bb8-toggle__checkbox" type="checkbox" {checked}>
      <div class="bb8-toggle__container">
        <div class="bb8-toggle__scenery">
          <div class="bb8-toggle__star"></div>
          <div class="bb8-toggle__star"></div>
          <div class="bb8-toggle__star"></div>
          <div class="bb8-toggle__star"></div>
          <div class="bb8-toggle__star"></div>
          <div class="bb8-toggle__star"></div>
          <div class="bb8-toggle__star"></div>
          <div class="tatto-1"></div>
          <div class="tatto-2"></div>
          <div class="gomrassen"></div>
          <div class="hermes"></div>
          <div class="chenini"></div>
          <div class="bb8-toggle__cloud"></div>
          <div class="bb8-toggle__cloud"></div>
          <div class="bb8-toggle__cloud"></div>
        </div>
        <div class="bb8">
          <div class="bb8__head-container">
            <div class="bb8__antenna"></div>
            <div class="bb8__antenna"></div>
            <div class="bb8__head"></div>
          </div>
          <div class="bb8__body"></div>
        </div>
        <div class="artificial__hidden">
          <div class="bb8__shadow"></div>
        </div>
      </div>
    </label>
  </a>
</div>
"""

def get_bb8_toggle_template(is_dark_mode: bool = False) -> str:
    """
    Get the BB8 toggle HTML template with the correct state.
    
    Args:
        is_dark_mode: Whether to set the toggle to dark mode (checked state)
        
    Returns:
        HTML template string with appropriate checked state
    """
    # Create a unique wrapper ID for this toggle instance
    wrapper_id = f"bb8-toggle-{uuid.uuid4().hex[:8]}"
    
    # Set the checkbox state based on current theme
    checked_attr = 'checked' if is_dark_mode else ''
    
    # Set the current theme for the JavaScript function
    current_theme = 'dark' if is_dark_mode else 'light'
    
    # Replace all placeholders in the template
    template = BB8_TOGGLE_TEMPLATE.replace('{wrapper_id}', wrapper_id)
    template = template.replace('{current_theme}', current_theme)
    template = template.replace('{checked}', checked_attr)
    
    return template
