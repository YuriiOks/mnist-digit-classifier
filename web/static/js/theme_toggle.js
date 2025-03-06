// Theme toggle functionality
function toggleDarkMode() {
    // Find the Streamlit button and click it programmatically
    const buttons = document.querySelectorAll('button');
    for (let button of buttons) {
        if (button.innerText === '🔄') {
            button.click();
            break;
        }
    }
} 