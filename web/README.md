# Web Application (Streamlit Frontend)

This directory contains the source code for the user-facing web application of the MNIST Digit Classifier project. It is built using Streamlit and provides an interactive interface for users to draw digits, upload images, view predictions, and manage their prediction history.

![Web Application Screenshot](../outputs/docs/web_app_main.png)

## 🌐 Overview

The web application serves as the primary interface for users to interact with the MNIST digit classification model. It handles user input (drawing, file uploads, URLs), communicates with the backend model service for predictions, interacts with the database to log history, and presents results and settings to the user in an intuitive way.

## ✨ Features

* **Interactive Drawing Canvas:** Allows users to draw digits directly in the browser using mouse or touch input.  
  ![Drawing Canvas Interaction](../outputs/docs/drawing_canvas.png)  
  *Placeholder: GIF showing a user drawing a digit on the canvas.*  
* **Image Upload:** Users can upload existing image files (PNG, JPG, JPEG) containing handwritten digits.  
* **URL Input:** Allows users to provide a URL pointing to an image of a digit online.  
* **Real-time Prediction Display:** Shows the model's predicted digit and the associated confidence score.  
  ![Prediction Result Display](../outputs/docs/prediction_result.png)  
* **Feedback Mechanism:** Users can indicate whether a prediction was correct and provide the true label if it was wrong.  
* **Prediction History:** Displays a paginated view of past predictions, including timestamps, predicted digits, confidence, and user feedback.  
  ![History View Example](../outputs/docs/history_view.png)  
* **Theme Customization:** Includes settings to switch between light and dark themes.  
* **Responsive Design:** Adapts layout for different screen sizes (desktop, tablet, mobile).

## 💻 Technology Stack

* **Framework:** [Streamlit](https://streamlit.io/) (for building the interactive web UI)  
* **Drawing:** [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)  
* **HTTP Requests:** [Requests](https://requests.readthedocs.io/en/latest/) (for communicating with the model API)  
* **Image Processing:** [Pillow](https://python-pillow.org/) (for handling image data)  
* **Database Interaction:** [psycopg2-binary](https://www.psycopg.org/docs/) (for connecting to PostgreSQL)  
* **Data Handling:** [Pandas](https://pandas.pydata.org/) (potentially used in history view)  
* **Styling:** Custom CSS & HTML Templates  

## 📁 Folder Structure

The `web/` directory is organized as follows:

```plaintext
web/
├── Dockerfile              # Docker build instructions for the web service
├── requirements.txt        # Python dependencies for the web app
├── app.py                  # Main Streamlit application entry point
├── core/                   # Core logic, state management, database interactions
│   ├── app_state/          # Session state management (theme, navigation, etc.)
│   ├── database/           # Database connection and query management (db_manager)
│   └── errors/             # Custom error handling classes
├── ui/                     # User interface components, layouts, and views
│   ├── components/         # Reusable UI elements (buttons, cards, inputs)
│   ├── layout/             # Overall page structure (header, footer, sidebar)
│   ├── theme/              # Theme management (CSS, JS, config)
│   └── views/              # Page-level components (Home, Draw, History, Settings)
├── assets/                 # Static files (CSS, JS, images, templates, config JSONs)
│   ├── css/
│   ├── js/
│   ├── config/
│   ├── images/
│   └── templates/
├── model/                  # Client-side code to interact with the model service
│   └── digit_classifier.py # Class handling model API communication
└── utils/                  # Shared utility functions (resource loading, aspects)
```

## 🧩 Key Components

* **app.py:** The main entry point that orchestrates the Streamlit application, manages routing between views, and initializes core components.  
* **core/:** Contains the application’s brain.  
  * **app_state/:** Manages session state using Streamlit’s `st.session_state` for the current theme, active view, canvas data, and prediction history cache.  
  * **database/:** Handles all communication with the PostgreSQL database via the db_manager singleton.  
* **ui/:** Defines the user interface.  
  * **views/:** Each Python file represents a distinct page or view (e.g., HomeView, DrawView).  
  * **components/:** Contains reusable UI elements like buttons, cards, and the drawing canvas, built using Streamlit widgets and custom HTML/CSS/JS.  
  * **layout/:** Defines the overall page structure, including the header, footer, and sidebar.  
  * **theme/:** Manages light/dark themes, CSS loading, and variable injection.  
* **assets/:** Stores all static files required by the frontend, including CSS stylesheets, JavaScript files, configuration JSONs, HTML templates, and images.  
* **model/digit_classifier.py:** A client class responsible for sending image data to the separate model service API and receiving predictions.

## ⚙️ Setup & Running

This web application is designed to run as a service within the project's Docker Compose setup.

1. **Prerequisites:** Ensure you have Docker and Docker Compose installed, as outlined in the main project `README.md`.  
2. **Build & Run:** Navigate to the project’s root directory (`mnist-digit-classifier/`) and run:
   ```bash
   docker-compose up --build -d web
   ```
3. **Access:** Once the container is running, open [http://localhost:8501](http://localhost:8501) in your web browser.  
4. **Dependencies:** The web service depends on the model service (for predictions) and the db service (for history). Docker Compose manages the network connections between these services.  
5. **Configuration:** Environment variables (e.g., `MODEL_URL`, `DB_HOST`) are passed from the `docker-compose.yml` file to the container.

## 🔑 Key Refactoring Notes

During development, the web application underwent a refactoring process to address HTML rendering issues, code organization, and user interface improvements. Some of these challenges closely mirrored those experienced in the `HistoryView` class, including:

1. **HTML Rendering Difficulties**  
   - **Issue**: Raw HTML tags occasionally appeared in the UI due to misplaced or malformed HTML snippets.  
   - **Solution**: Consolidate rendering into a single Markdown call with `unsafe_allow_html=True` and ensure that closing tags and string formatting are carefully handled.

2. **Multiline String Formatting in Python**  
   - **Issue**: Large multiline f-strings led to tricky whitespace/indentation bugs.  
   - **Solution**: Keep content on the same line as the triple quotes or use a carefully managed indentation strategy.

3. **Card Rendering Strategy**  
   - **Issue**: Inconsistent approach for generating “cards” or UI containers.  
   - **Solution**: Create and reuse helper methods (or separate HTML fragments) for each component of the card. Combine them in one place to ensure fewer overlapping HTML tags.

4. **Code Organization & Redundancy**  
   - **Issue**: The application had grown a handful of “long and complex” methods in different UI modules. This made it difficult to pinpoint bugs and maintain consistent formatting.  
   - **Solution**: Introduce clear helper methods for repeated logic (e.g., string formatting for HTML, database operations for history retrieval).

5. **Improved User Experience**  
   - **Issue**: Filters, pagination, and visual design lacked clarity for end users.  
   - **Solution**: Implement more intuitive controls for filtering and paging, improve spacing/typography, and highlight correct or incorrect predictions more boldly.

## 🚧 Challenges & Difficulties

Beyond the typical software development hurdles, here are a few notable difficulties encountered:

- **Streamlit “Markdown vs. HTML” Pitfalls**  
  Streamlit’s `st.markdown` with `unsafe_allow_html=True` is powerful but can be tricky if you generate large blocks of HTML. Using multiline strings with incorrect indentation or missing quotes can cause raw HTML to show.  
- **State Management**  
  Handling interactive states (canvas drawings, input methods, and filtering) required robust session-level variables. We used Streamlit’s session state or custom state managers to ensure data persisted properly between user actions.  
- **Pagination & Filtering**  
  Pagination and dynamic filtering of history logs demanded deeper integration with the underlying database logic. Some refactoring was needed to unify how data was retrieved and then sliced for display.  
- **Cross-Component Dependencies**  
  The `history_view.py`, `draw_view.py`, etc., each had to share certain data (like the predicted digit or user feedback). Centralizing this data flow in dedicated helper classes or state managers improved clarity but required a careful reorganization of methods.  
- **Version Mismatches**  
  Mismatched library versions (particularly with Streamlit and supporting libraries) sometimes caused layout or performance issues. Always ensure `requirements.txt` is consistent across all services.

## 🛠️ Running Locally

1. **Install Prerequisites**: Docker & Docker Compose, Git.  
2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```
3. **Start Services**:
   ```bash
   docker-compose up --build
   ```
4. **Access the Web App**: Go to [http://localhost:8501](http://localhost:8501).

## 🐳 Dockerfile Overview

The included `Dockerfile` uses a Python 3.9 Slim base image, installs dependencies, copies the web code, and sets up Streamlit to run on port 8501. This container is orchestrated alongside the `model` (Flask API) and `db` (PostgreSQL) services in the main `docker-compose.yml`.

## 🗄️ Future Enhancements

- **User Accounts**: Add login/auth for multi-user history tracking.  
- **Refined Logging**: Centralized logging to capture client events and server logs for advanced debugging.  
- **Real Database Integration**: Expand the local in-memory or partial DB approach so that user logs persist beyond container restarts.  
- **Dynamic Retraining**: Optionally feed user-corrected labels back into the model training pipeline.  
- **Advanced Visualization**: More charts and metrics integrated into the UI to highlight model performance.

## 📝 Conclusion

Refactoring the Streamlit web interface has brought greater clarity to how HTML is generated, improved user experience, and yielded a more maintainable codebase—important steps toward a more robust and production-ready MNIST Digit Classifier application. Whether it’s centralizing card rendering, taming multiline f-strings, or better organizing code logic, each small fix has added up to a smoother, more extensible user experience.