// Theme toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    // Set up the toggle button event listener
    const toggleButton = document.getElementById('darkModeToggle');
    if (toggleButton) {
        toggleButton.addEventListener('click', function() {
            // Find and click the hidden Streamlit button
            const streamlitButtons = document.querySelectorAll('button');
            for (let button of streamlitButtons) {
                if (button.getAttribute('data-testid') === 'baseButton-secondary') {
                    button.click();
                    break;
                }
            }
        });
    }
});