// Theme toggle functionality for settings page
document.addEventListener('DOMContentLoaded', function() {
    const checkbox = document.getElementById('themeCheckbox');
    if (checkbox) {
        checkbox.addEventListener('change', function() {
            // Find and click the hidden button
            const buttons = Array.from(document.querySelectorAll('button'));
            const themeButton = buttons.find(button => 
                button.innerText.includes('Toggle Theme')
            );
            if (themeButton) {
                themeButton.click();
            }
        });
    }
}); 