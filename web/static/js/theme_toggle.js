// Theme toggle script that works with the slider toggle
document.addEventListener('DOMContentLoaded', function() {
    console.log('Theme toggle script initialized');
    
    // Function to toggle theme via different methods
    function toggleTheme() {
        console.log('Toggling theme');
        
        // Try to find and click the hidden settings button
        const settingsButton = document.querySelector('button[key="theme_toggle_settings"]');
        if (settingsButton) {
            console.log('Found settings toggle button, clicking it');
            settingsButton.click();
            return;
        }
        
        // Use URL parameter as fallback
        console.log('Using URL parameter as fallback');
        const url = new URL(window.location);
        url.searchParams.set('toggle_theme', 'true');
        window.location.href = url.toString();
    }
    
    // Handle clicks on the sidebar theme toggle
    function setupSidebarToggle() {
        const sidebarToggle = document.getElementById('sidebar-theme-toggle');
        if (sidebarToggle && !sidebarToggle.hasListener) {
            console.log('Setting up sidebar toggle');
            sidebarToggle.hasListener = true;
            
            sidebarToggle.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('Sidebar toggle clicked');
                toggleTheme();
            });
            
            // Also handle clicks on the icon inside
            const icon = sidebarToggle.querySelector('.icon');
            if (icon) {
                icon.addEventListener('click', function(e) {
                    e.stopPropagation();
                    console.log('Icon clicked');
                    toggleTheme();
                });
            }
        }
    }
    
    // Handle the slider toggle in settings
    function setupSliderToggle() {
        // Look for the theme checkbox in the settings page
        const themeCheckbox = document.getElementById('themeCheckbox');
        if (themeCheckbox && !themeCheckbox.hasListener) {
            console.log('Setting up theme checkbox toggle');
            themeCheckbox.hasListener = true;
            
            // Set initial state based on current theme
            const isDarkMode = document.body.classList.contains('dark');
            themeCheckbox.checked = isDarkMode;
            
            themeCheckbox.addEventListener('change', function() {
                console.log('Theme checkbox toggled to', this.checked);
                toggleTheme();
            });
            
            // Also add click handler to the entire toggle wrapper for better UX
            const toggleWrapper = document.querySelector('.toggle-wrapper');
            if (toggleWrapper && !toggleWrapper.hasListener) {
                toggleWrapper.hasListener = true;
                toggleWrapper.addEventListener('click', function(e) {
                    // Don't trigger if clicking directly on the checkbox (to avoid double toggle)
                    if (e.target !== themeCheckbox) {
                        console.log('Toggle wrapper clicked');
                        // Simulate checkbox toggle
                        themeCheckbox.checked = !themeCheckbox.checked;
                        // Trigger change event
                        themeCheckbox.dispatchEvent(new Event('change'));
                    }
                });
            }
        }
    }
    
    // Initialize both toggle mechanisms
    function initToggles() {
        setupSidebarToggle();
        setupSliderToggle();
    }
    
    // Run immediately
    initToggles();
    
    // Also run after a short delay to catch dynamically added elements
    setTimeout(initToggles, 1000);
    
    // Observer for DOM changes to catch dynamically added toggles
    const observer = new MutationObserver(function() {
        initToggles();
    });
    
    // Start observing the document body
    observer.observe(document.body, { childList: true, subtree: true });
});