
├── app.py                      # Main entry point
├── requirements.txt            # Dependencies
├── streamlit_config.toml       # Streamlit configuration
│
├── core/
│   ├── __init__.py
│   ├── app_state/              # Application state management
│   │   ├── __init__.py
│   │   ├── session_state.py    # Global session state definition
│   │   ├── theme_state.py      # Theme-specific state
│   │   ├── canvas_state.py     # Canvas-specific state
│   │   ├── history_state.py    # History-specific state
│   │   └── settings_state.py   # Settings-specific state
│   ├── errors/                 # Error handling
│   │   ├── __init__.py
│   │   ├── error_handler.py    # Base error handler
│   │   ├── ui_errors.py        # UI-specific errors
│   │   └── service_errors.py   # Service-specific errors
│   └── config/                 # Configuration management
│       ├── __init__.py
│       └── app_config.py       # App configuration
│
├── services/                   # Business logic
│   ├── __init__.py
│   ├── prediction/             # Prediction services
│   │   ├── __init__.py
│   │   ├── digit_classifier.py # Digit classification logic
│   │   └── model_service.py    # Model interaction
│   ├── image/                  # Image processing
│   │   ├── __init__.py
│   │   ├── processor.py        # Image processing functions
│   │   └── converter.py        # Image format conversion
│   └── data/                   # Data management
│       ├── __init__.py
│       ├── history_service.py  # History data service
│       └── export_service.py   # Data export functionality
│
├── ui/                         # UI components hierarchy
│   ├── __init__.py
│   ├── theme/                  # Theme management
│   │   ├── __init__.py
│   │   ├── theme_manager.py    # Theme application logic
│   │   ├── theme_loader.py     # Theme resources loading
│   │   └── theme_utils.py      # Theme utility functions
│   ├── layout/                 # Layout components
│   │   ├── __init__.py
│   │   ├── page_layout.py      # Overall page layout
│   │   ├── header.py           # Header component
│   │   ├── footer.py           # Footer component
│   │   ├── sidebar.py          # Sidebar component
│   │   └── container.py        # Container component
│   ├── components/             # Reusable components
│   │   ├── __init__.py
│   │   ├── base/               # Base component classes
│   │   │   ├── __init__.py
│   │   │   ├── component.py    # Base component class
│   │   │   └── composite.py    # Composite component class
│   │   ├── cards/              # Card components
│   │   │   ├── __init__.py
│   │   │   ├── card.py         # Basic card
│   │   │   ├── content_card.py # Content card
│   │   │   ├── feature_card.py # Feature card
│   │   │   └── settings_card.py # Settings card
│   │   ├── inputs/             # Input components
│   │   │   ├── __init__.py
│   │   │   ├── canvas.py       # Drawing canvas
│   │   │   ├── file_upload.py  # File upload component
│   │   │   └── url_input.py    # URL input component
│   │   ├── navigation/         # Navigation components
│   │   │   ├── __init__.py
│   │   │   ├── tabs.py         # Tabs component
│   │   │   ├── menu.py         # Menu component
│   │   │   └── breadcrumbs.py  # Breadcrumbs component
│   │   ├── feedback/           # Feedback components
│   │   │   ├── __init__.py
│   │   │   ├── alerts.py       # Alert components
│   │   │   └── results.py      # Results display
│   │   └── controls/           # Control components
│   │       ├── __init__.py
│   │       ├── buttons.py      # Button components
│   │       ├── toggles.py      # Toggle components
│   │       └── sliders.py      # Slider components
│   └── views/                  # Page views
│       ├── __init__.py
│       ├── base_view.py        # Base view class
│       ├── home/               # Home page
│       │   ├── __init__.py
│       │   ├── home_view.py    # Home page view
│       │   ├── welcome.py      # Welcome section
│       │   └── features.py     # Features section
│       ├── drawing/            # Drawing page
│       │   ├── __init__.py
│       │   ├── drawing_view.py # Drawing page view
│       │   ├── canvas_panel.py # Canvas panel
│       │   └── results_panel.py # Results panel
│       ├── history/            # History page
│       │   ├── __init__.py
│       │   ├── history_view.py # History page view
│       │   ├── history_table.py# History table
│       │   └── analytics.py    # Analytics section
│       └── settings/           # Settings page
│           ├── __init__.py
│           ├── settings_view.py# Settings page view
│           ├── theme_settings.py # Theme settings
│           └── canvas_settings.py # Canvas settings
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── streamlit/              # Streamlit-specific utilities
│   │   ├── __init__.py
│   │   ├── st_utils.py         # Streamlit utilities
│   │   └── st_patchers.py      # Streamlit functionality extensions
│   ├── file/                   # File utilities
│   │   ├── __init__.py
│   │   ├── file_loader.py      # File loading utilities
│   │   └── path_utils.py       # Path manipulation utilities
│   ├── html/                   # HTML utilities
│   │   ├── __init__.py
│   │   ├── html_loader.py      # HTML loading
│   │   └── template_engine.py  # Template processing
│   ├── css/                    # CSS utilities
│   │   ├── __init__.py
│   │   ├── css_loader.py       # CSS loading
│   │   └── css_injector.py     # CSS injection
│   └── js/                     # JavaScript utilities
│       ├── __init__.py
│       ├── js_loader.py        # JS loading
│       └── js_injector.py      # JS injection
│
├── assets/                     # Static assets
│   ├── css/
│   │   ├── global/             # Global styles
│   │   │   ├── variables.css   # CSS variables
│   │   │   ├── reset.css       # CSS reset
│   │   │   ├── typography.css  # Typography styles
│   │   │   └── animations.css  # Animation styles
│   │   ├── themes/             # Theme styles
│   │   │   ├── light/          # Light theme
│   │   │   │   ├── variables.css # Light theme variables
│   │   │   │   ├── components/ # Component styles for light theme
│   │   │   │   └── views/      # View styles for light theme
│   │   │   └── dark/           # Dark theme
│   │   │       ├── variables.css # Dark theme variables
│   │   │       ├── components/ # Component styles for dark theme
│   │   │       └── views/      # View styles for dark theme
│   │   ├── components/         # Component styles
│   │   │   ├── layout/         # Layout styles
│   │   │   │   ├── header.css
│   │   │   │   ├── footer.css
│   │   │   │   └── sidebar.css
│   │   │   ├── cards/          # Card styles
│   │   │   │   ├── base.css
│   │   │   │   ├── content_card.css
│   │   │   │   └── settings_card.css
│   │   │   ├── inputs/         # Input styles
│   │   │   │   ├── canvas.css
│   │   │   │   └── file_upload.css
│   │   │   ├── navigation/     # Navigation styles
│   │   │   │   ├── tabs.css
│   │   │   │   └── menu.css
│   │   │   ├── feedback/       # Feedback styles
│   │   │   │   ├── alerts.css
│   │   │   │   └── results.css
│   │   │   └── controls/       # Control styles
│   │   │       ├── buttons.css
│   │   │       └── toggles.css
│   │   └── views/              # View-specific styles
│   │       ├── home.css
│   │       ├── drawing.css
│   │       ├── history.css
│   │       └── settings.css
│   ├── js/                     # JavaScript
│   │   ├── theme/              # Theme JS
│   │   │   ├── theme_toggle.js
│   │   │   └── theme_detector.js
│   │   ├── components/         # Component JS
│   │   │   ├── canvas.js
│   │   │   └── tabs.js
│   │   └── utils/              # JS utilities
│   │       ├── dom_utils.js
│   │       └── accessibility.js
│   └── images/                 # Static images
│       ├── logo.png
│       └── icons/              # Icon assets
│
└── templates/                  # HTML templates
    ├── components/             # Component templates
    │   ├── layout/             # Layout templates
    │   │   ├── header.html
    │   │   ├── footer.html
    │   │   └── sidebar.html
    │   ├── cards/              # Card templates
    │   │   ├── content_card.html
    │   │   ├── feature_card.html
    │   │   └── settings_card.html
    │   ├── inputs/             # Input templates
    │   │   └── url_input.html
    │   ├── navigation/         # Navigation templates
    │   │   ├── tabs.html
    │   │   └── menu.html
    │   ├── feedback/           # Feedback templates
    │   │   ├── alert.html
    │   │   └── results.html
    │   └── controls/           # Control templates
    │       ├── button.html
    │       └── toggle.html
    └── views/                  # View templates
        ├── home/               # Home templates
        │   ├── welcome.html
        │   └── features.html
        ├── drawing/            # Drawing templates
        │   ├── canvas_panel.html
        │   └── results_panel.html
        ├── history/            # History templates
        │   ├── history_table.html
        │   └── analytics.html
        └── settings/           # Settings templates
            ├── theme_settings.html
            └── canvas_settings.html