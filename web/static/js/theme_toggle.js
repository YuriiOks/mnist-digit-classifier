// Theme toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    // Set initial body class based on current theme
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    if (isDarkMode) {
        document.body.classList.add('dark');
    }
    
    // Add event listener to theme toggle
    const themeToggle = document.getElementById('darkModeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleDarkMode);
    }
});

function toggleDarkMode() {
    // Find and click the hidden Streamlit button
    const toggleButton = document.querySelector('button[key="dark_mode_toggle"]');
    if (toggleButton) {
        toggleButton.click();
        
        // Toggle body class for immediate visual feedback
        const isDarkMode = !document.body.classList.contains('dark');
        if (isDarkMode) {
            document.body.classList.add('dark');
            localStorage.setItem('darkMode', 'true');
        } else {
            document.body.classList.remove('dark');
            localStorage.setItem('darkMode', 'false');
        }
        
        // Add a subtle animation for better UX
        const toggleIcon = document.querySelector('.toggle-icon');
        if (toggleIcon) {
            toggleIcon.style.animation = 'none';
            setTimeout(() => {
                toggleIcon.style.animation = 'float 2s ease-in-out infinite';
            }, 10);
        }
    } else {
        console.error('Theme toggle button not found');
    }
}