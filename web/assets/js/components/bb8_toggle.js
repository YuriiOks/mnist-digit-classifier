/**
 * MNIST Digit Classifier
 * Copyright (c) 2025
 * File: assets/js/components/controls/bb8_toggle.js
 * Description: JavaScript for BB8 toggle integration with theme system
 * Created: 2025-03-16
 */

(function() {
    /**
     * Initialize BB8 toggle for theme switching
     */
    function initializeBB8Toggle() {
      // Find all BB8 toggle checkboxes
      const toggles = document.querySelectorAll('.bb8-toggle__checkbox');
      if (!toggles.length) return;
      
      // Set initial state based on current theme
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
      toggles.forEach(toggle => {
        toggle.checked = currentTheme === 'dark';
        
        // Add event listener to each toggle
        toggle.addEventListener('change', function(e) {
          const newTheme = e.target.checked ? 'dark' : 'light';
          
          // Apply theme changes visually
          document.documentElement.setAttribute('data-theme', newTheme);
          
          // Store preference in localStorage
          localStorage.setItem('theme-preference', newTheme);
          
          // Communicate with Streamlit if available
          if (window.parent && window.parent.postMessage) {
            window.parent.postMessage({
              type: 'streamlit:setComponentValue',
              value: {
                theme: newTheme,
                bb8_toggle_checkbox: e.target.checked
              }
            }, '*');
          }
        });
      });
    }
    
    // Initialize when DOM is fully loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initializeBB8Toggle);
    } else {
      initializeBB8Toggle();
    }
    
    // Reinitialize when Streamlit reruns
    window.addEventListener('message', function(e) {
      if (e.data.type === 'streamlit:render') {
        setTimeout(initializeBB8Toggle, 100);
      }
    });
  })();