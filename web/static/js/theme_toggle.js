// Add immediate console log to check if file is loaded at all
console.log('Theme toggle script loaded!');

// Handle the theme toggle button clicks
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded in theme_toggle.js');
    
    // Set up the toggle button event listener
    const toggleButton = document.getElementById('darkModeToggle');
    
    if (toggleButton) {
        console.log('Theme toggle button found in JS file');
        
        // Add click event handler to redirect with the toggle parameter
        toggleButton.addEventListener('click', function() {
            console.log('Toggle button clicked, redirecting with parameter');
            window.location.href = '?toggle_theme=true';
        });
    } else {
        console.log('Theme toggle button not found - will be added to DOM later');
        
        // Set up a mutation observer to watch for the button being added later
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes) {
                    const toggleButton = document.getElementById('darkModeToggle');
                    if (toggleButton && !toggleButton.hasEventListener) {
                        console.log('Found toggle button after DOM update');
                        toggleButton.addEventListener('click', function() {
                            window.location.href = '?toggle_theme=true';
                        });
                        toggleButton.hasEventListener = true;
                    }
                }
            });
        });
        
        // Start observing the document body for DOM changes
        observer.observe(document.body, { childList: true, subtree: true });
    }
});