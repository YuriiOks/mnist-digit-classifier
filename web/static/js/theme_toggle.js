// Add immediate console log to check if file is loaded at all
console.log('Theme toggle script loaded!');

// Improved theme toggle script with extensive debugging
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded in theme_toggle.js');
    
    // Set up the toggle button event listener
    const toggleButton = document.getElementById('darkModeToggle');
    
    if (toggleButton) {
        console.log('Theme toggle button found in JS file');
        
        // Store original background for later reset
        const originalBackground = window.getComputedStyle(toggleButton).backgroundColor;
        
        toggleButton.addEventListener('click', function() {
            console.log('Theme toggle button clicked in JS file');
            
            // Visual feedback for click
            this.style.backgroundColor = 'rgba(255, 0, 0, 0.5)';
            setTimeout(() => {
                this.style.backgroundColor = originalBackground;
            }, 500);
            
            // Debug what Streamlit has injected
            console.log('Document body class:', document.body.className);
            console.log('Streamlit visible elements:', document.querySelectorAll('[data-testid]').length);
            
            // Look for the hidden button by various means
            const approaches = [
                { name: 'data-testid', selector: 'button[data-testid="baseButton-secondary"]' },
                { name: 'dark_mode_toggle', selector: 'button[key="dark_mode_toggle"]' },
                { name: 'secondary type', selector: 'button[type="secondary"]' }
            ];
            
            let targetButton = null;
            
            // Try each approach
            for (const approach of approaches) {
                const button = document.querySelector(approach.selector);
                console.log(`Looking by ${approach.name}:`, button ? 'Found' : 'Not found');
                if (button) {
                    targetButton = button;
                    break;
                }
            }
            
            // If not found, try with full button search
            if (!targetButton) {
                console.log('Trying full button scan...');
                const allButtons = document.querySelectorAll('button');
                console.log('Total buttons:', allButtons.length);
                
                // Log detailed info for each button
                allButtons.forEach((btn, i) => {
                    const info = {
                        text: btn.innerText,
                        attributes: {},
                        dimensions: {
                            width: btn.offsetWidth,
                            height: btn.offsetHeight,
                            visible: btn.offsetParent !== null
                        }
                    };
                    
                    // Get all attributes
                    Array.from(btn.attributes).forEach(attr => {
                        info.attributes[attr.name] = attr.value;
                    });
                    
                    console.log(`Button ${i} details:`, info);
                    
                    // Check for specific text or key
                    if (btn.innerText === '🔄' || 
                        btn.getAttribute('key') === 'dark_mode_toggle' ||
                        btn.getAttribute('data-key') === 'dark_mode_toggle') {
                        console.log(`Found potential target button ${i}`);
                        targetButton = btn;
                    }
                });
            }
            
            // Click the button if found
            if (targetButton) {
                console.log('Clicking target button:', targetButton);
                targetButton.click();
            } else {
                console.error('Could not find any suitable button to click');
                alert('Theme toggle button not found. Check console for details.');
            }
        });
    } else {
        console.error('Theme toggle button not found with ID: darkModeToggle');
    }
});