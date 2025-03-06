// Updated theme_toggle.js
document.addEventListener('DOMContentLoaded', function() {
    // Set up the toggle button event listener
    const toggleButton = document.getElementById('darkModeToggle');
    if (toggleButton) {
        toggleButton.addEventListener('click', function() {
            // Get the current icon element
            const iconElement = toggleButton.querySelector('.toggle-icon');
            
            // Toggle between sun and moon emoji directly in the DOM
            if (iconElement.textContent === '☀️') {
                iconElement.textContent = '🌙';
            } else {
                iconElement.textContent = '☀️';
            }
            
            // Also click the hidden Streamlit button to update the state
            const streamlitButtons = document.querySelectorAll('button');
            for (let button of streamlitButtons) {
                if (button.innerText === "🔄") {
                    button.click();
                    break;
                }
            }
        });
    }
});