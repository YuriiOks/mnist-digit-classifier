// Simplified theme toggle script
document.addEventListener('DOMContentLoaded', function() {
    console.log('Theme toggle script initialized');
    
    // Function to find and click the theme toggle button
    function findAndClickToggleButton() {
        // First try the settings button
        const settingsButton = document.querySelector('button[key="theme_toggle_settings"]');
        if (settingsButton) {
            console.log('Found settings toggle button, clicking it');
            settingsButton.click();
            return true;
        }
        
        // Then try the checkbox in settings
        const themeCheckbox = document.querySelector('input[key="theme_toggle_checkbox"]');
        if (themeCheckbox) {
            console.log('Found theme checkbox, clicking it');
            themeCheckbox.click();
            return true;
        }
        
        return false;
    }
    
    // Setup all theme toggle buttons in the document
    function setupThemeToggles() {
        // Setup sidebar toggle
        const sidebarToggle = document.getElementById('sidebar-theme-toggle');
        if (sidebarToggle && !sidebarToggle.hasListener) {
            sidebarToggle.hasListener = true;
            sidebarToggle.addEventListener('click', function() {
                if (!findAndClickToggleButton()) {
                    // Last resort: URL parameter
                    const url = new URL(window.location);
                    url.searchParams.set('toggle_theme', 'true');
                    window.location.href = url.toString();
                }
            });
        }
        
        // Setup any other theme toggles
        const darkModeToggle = document.getElementById('darkModeToggle');
        if (darkModeToggle && !darkModeToggle.hasListener) {
            darkModeToggle.hasListener = true;
            darkModeToggle.addEventListener('click', function() {
                findAndClickToggleButton();
            });
        }
    }
    
    // Run immediately and after a delay
    setupThemeToggles();
    setTimeout(setupThemeToggles, 500);
    
    // Watch for DOM changes to catch dynamically added toggles
    const observer = new MutationObserver(function() {
        setupThemeToggles();
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
});