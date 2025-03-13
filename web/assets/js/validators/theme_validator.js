/**
 * MNIST Digit Classifier
 * Copyright (c) 2025
 * File: assets/js/validators/theme_validator.js
 * Description: Validates theme variables for consistency
 * Created: 2024-05-05
 */

/**
 * Validates a theme configuration object against the required schema
 * @param {Object} theme - The theme configuration object
 * @returns {Array} - Array of validation errors or empty array if valid
 */
function validateTheme(theme) {
  const errors = [];
  
  // Required top-level properties
  const requiredProps = ['name', 'displayName', 'colors', 'settings', 'fonts'];
  requiredProps.forEach(prop => {
    if (!theme[prop]) {
      errors.push(`Missing required property: ${prop}`);
    }
  });
  
  // Required color properties
  const requiredColors = [
    'primary', 'secondary', 'accent', 
    'background', 'backgroundAlt', 'card', 'cardAlt',
    'text', 'textLight', 'textMuted', 'border',
    'success', 'warning', 'error', 'info'
  ];
  
  if (theme.colors) {
    requiredColors.forEach(color => {
      if (!theme.colors[color]) {
        errors.push(`Missing required color: ${color}`);
      }
    });
  }
  
  // Required settings
  const requiredSettings = ['borderRadius', 'shadowStrength', 'buttonStyle', 'inputStyle'];
  if (theme.settings) {
    requiredSettings.forEach(setting => {
      if (!theme.settings[setting]) {
        errors.push(`Missing required setting: ${setting}`);
      }
    });
  }
  
  // Required fonts
  const requiredFonts = ['primary', 'heading', 'code'];
  if (theme.fonts) {
    requiredFonts.forEach(font => {
      if (!theme.fonts[font]) {
        errors.push(`Missing required font: ${font}`);
      }
    });
  }
  
  return errors;
}

export { validateTheme }; 