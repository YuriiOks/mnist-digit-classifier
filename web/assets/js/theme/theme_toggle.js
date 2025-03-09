/**
 * MNIST Digit Classifier
 * Copyright (c) 2025
 * File: assets/js/theme/theme_toggle.js
 * Description: JavaScript for theme toggling functionality
 * Created: 2024-05-01
 */

(function() {
  /**
   * Apply theme to document and update UI elements
   * @param {string} theme - Theme name ('light' or 'dark')
   */
  window.applyTheme = function(theme) {
    // Set data-theme attribute on html element
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update any theme toggles on the page
    const toggles = document.querySelectorAll('.theme-toggle input[type="checkbox"]');
    toggles.forEach(toggle => {
      toggle.checked = theme === 'dark';
    });
    
    // Store theme preference in localStorage
    localStorage.setItem('theme-preference', theme);
    
    // Dispatch custom event for theme change
    const event = new CustomEvent('themeChanged', { detail: { theme: theme } });
    document.dispatchEvent(event);
    
    // Update any theme-specific images
    updateThemeSpecificAssets(theme);
  };
  
  /**
   * Update theme-specific assets like images and icons
   * @param {string} theme - Theme name ('light' or 'dark')
   */
  function updateThemeSpecificAssets(theme) {
    // Update theme-specific images
    const themeImages = document.querySelectorAll('[data-theme-image]');
    themeImages.forEach(img => {
      const basePath = img.getAttribute('data-theme-image');
      if (basePath) {
        img.src = `assets/images/${theme}/${basePath}`;
      }
    });
    
    // Update theme-specific icons
    const themeIcons = document.querySelectorAll('[data-theme-icon]');
    themeIcons.forEach(icon => {
      const iconName = icon.getAttribute('data-theme-icon');
      if (iconName) {
        icon.src = `assets/images/icons/${theme}/${iconName}.svg`;
      }
    });
  }
  
  /**
   * Initialize theme toggle UI elements
   */
  function initializeThemeToggles() {
    // Find all theme toggle elements
    const toggles = document.querySelectorAll('.theme-toggle input[type="checkbox"]');
    
    // Set initial state based on current theme
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    toggles.forEach(toggle => {
      toggle.checked = currentTheme === 'dark';
      
      // Add event listener to each toggle
      toggle.addEventListener('change', function(e) {
        const newTheme = e.target.checked ? 'dark' : 'light';
        
        // This will trigger Streamlit rerun with new theme
        if (window.Streamlit) {
          // Use Streamlit's communication mechanism
          const toggleData = {
            toggled: newTheme === 'dark',
            theme: newTheme
          };
          window.Streamlit.setComponentValue(toggleData);
        }
        
        // Apply theme changes immediately for better UX
        window.applyTheme(newTheme);
      });
    });
  }
  
  // Initialize when DOM is fully loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeThemeToggles);
  } else {
    initializeThemeToggles();
  }
  
  // Listen for Streamlit load event
  if (window.Streamlit) {
    window.Streamlit.addEventListener('streamlit:render', function() {
      initializeThemeToggles();
    });
  }
})(); 