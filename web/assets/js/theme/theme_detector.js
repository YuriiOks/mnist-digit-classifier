/**
 * MNIST Digit Classifier
 * Copyright (c) 2025
 * File: assets/js/theme/theme_detector.js
 * Description: JavaScript for detecting system theme preferences
 * Created: 2024-05-01
 */

(function() {
  /**
   * Detect system color scheme preference
   * @returns {string} - Detected preference ('light' or 'dark')
   */
  function getSystemPreference() {
    // Check if prefers-color-scheme media query is supported
    if (window.matchMedia) {
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
      } else {
        return 'light';
      }
    }
    
    // Fallback if media query not supported
    return 'light';
  }
  
  /**
   * Detect and apply system theme preference
   */
  window.detectSystemThemePreference = function() {
    // Get stored preference (if any)
    const storedPreference = localStorage.getItem('theme-preference');
    
    // If no stored preference, use system preference
    if (!storedPreference) {
      const systemPreference = getSystemPreference();
      
      // Apply detected theme
      if (window.applyTheme) {
        window.applyTheme(systemPreference);
      }
      
      // Save to Streamlit state if Streamlit is available
      if (window.Streamlit) {
        const themeData = {
          theme: systemPreference,
          system_detected: true
        };
        window.Streamlit.setComponentValue(themeData);
      }
    }
    
    // Listen for system preference changes
    if (window.matchMedia) {
      const darkMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      
      // Add change listener if supported
      if (darkMediaQuery.addEventListener) {
        darkMediaQuery.addEventListener('change', function(e) {
          // Only apply if user hasn't set a preference
          if (!localStorage.getItem('theme-preference')) {
            const newTheme = e.matches ? 'dark' : 'light';
            
            // Apply new theme
            if (window.applyTheme) {
              window.applyTheme(newTheme);
            }
            
            // Update Streamlit
            if (window.Streamlit) {
              const themeData = {
                theme: newTheme,
                system_detected: true
              };
              window.Streamlit.setComponentValue(themeData);
            }
          }
        });
      }
    }
  };
  
  // Auto-detect when script loads
  setTimeout(function() {
    if (window.detectSystemThemePreference) {
      window.detectSystemThemePreference();
    }
  }, 100);
})(); 