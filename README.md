# MNIST Digit Classifier
![GitHub contributors](https://img.shields.io/github/contributors/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![Forks](https://img.shields.io/github/forks/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![Issues](https://img.shields.io/github/issues/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![License](https://img.shields.io/github/license/YuriiOks/mnist-digit-classifier?style=for-the-badge)

## 🌐 Overview

This project implements an end-to-end machine learning application that recognizes hand-drawn digits (0-9). It features:
- A **Streamlit** web interface, where users can draw digits, upload image files, or provide image URLs for classification.
- A **PyTorch** CNN model served via a **Flask** API.
- A **PostgreSQL** database to log user predictions and feedback.

![Application Screenshot](outputs/docs/web_app_main.png)

**Live Demo:** [http://157.180.73.218](http://157.180.73.218)  
*(Note: Replace with your actual deployment URL/IP.)*

---

## ✨ Features

- **Interactive Drawing Canvas**: Draw digits directly in your browser.
- **Multiple Input Methods**: Supports drawing, file upload, and image URL submission.
- **Real-time Prediction**: Instantly receive the model’s predicted digit with a confidence score.
- **Feedback Mechanism**: Provide the correct label if the model’s prediction is wrong.
- **Prediction History**: View past predictions, confidence scores, timestamps, and user feedback.
- **Theme Customization**: Basic light/dark theme switching.
- **Fully Containerized**: Easy to deploy using Docker and Docker Compose.
- **End-to-End ML Pipeline**: Model training, evaluation, inference API, and frontend integration.
- **Apple Silicon MPS Acceleration**: Faster training and inference on compatible hardware.

---

## 💻 Technology Stack

- **Machine Learning**:  
  - [PyTorch](https://pytorch.org/), [TorchVision](https://pytorch.org/vision/stable/index.html), [Scikit-learn](https://scikit-learn.org/stable/), [Matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/)

- **API Backend**:  
  - [Flask](https://flask.palletsprojects.com/) + [Gunicorn](https://gunicorn.org/)

- **Frontend**:  
  - [Streamlit](https://streamlit.io/)  
  - [Pillow](https://python-pillow.org/) for image processing  
  - [Requests](https://requests.readthedocs.io/en/latest/) for communicating with the model API  
  - [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) for drawing  
  - [Pandas](https://pandas.pydata.org/) for displaying or filtering data

- **Database**:  
  - [PostgreSQL](https://www.postgresql.org/) with [psycopg2-binary](https://www.psycopg.org/docs/) for logging predictions

- **Containerization & Deployment**:  
  - [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)  
  - Self-managed VPS (Hetzner) *(placeholder: update if using a different provider)*

---

## 📂 Documentation
This repository is structured to facilitate easy navigation and understanding of the project components. Each major component has its own directory with a README file for detailed documentation.
- **Model**: Contains the machine learning model, training scripts, and Flask API for serving predictions.
- **Web**: Contains the Streamlit frontend application, including UI components and views.
- **Database**: Contains the SQL schema for initializing the PostgreSQL database.

You can find detailed documentation for each component in their respective directories:
- [Model README](model/README.md)
- [Web README](web/README.md)
- [Database README](database/README.md)

## 📁 Project Structure

```plaintext
mnist-digit-classifier/
├── model/               # 🧠 ML model, training scripts, and Flask API
├── web/                 # 🖥️ Streamlit frontend (UI components, views, states)
├── database/            # 💾 Database schema initialization (PostgreSQL)
├── scripts/             # 🛠️ Utility scripts (migrations, checks, etc.)
├── utils/               # 🔩 Project-wide utility modules (mps_verification, environment setup)
├── docker-compose.yml   # 🐳 Docker Compose for orchestrating containers
├── LICENSE              # 🏷️ MIT License file
├── README.md            # 📖 Main project README
└── docs/                # 📚 Documentation (e.g. ERD, PDD)
└── outputs/             # 📈 Generated outputs (plots, logs)
└── saved_models/        # 💾 Saved model weights (if not tracked by Git)
```

For more detailed information, see:

- [Model README](model/README.md)  
- [Web README](web/README.md)  
- [Database README](database/README.md)

---

## 🚀 Getting Started

### Prerequisites

- **Docker**: Install Docker Desktop (Windows/Mac) or Docker Engine (Linux).  
- **Docker Compose**: Usually included with Docker Desktop or available as a separate package.  
- **Git**: For cloning this repository.

### Local Development

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YuriiOks/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```
   *(Replace the URL above with your own fork if applicable.)*

2. **(Optional) Set Environment Variables**:  
   Create a `.env` file at the project root if you need to override defaults (e.g., DB credentials).

3. **Build & Start Services**:
   ```bash
   docker-compose up --build -d
   ```
   - `--build`: Forces Docker to rebuild images from the Dockerfiles.
   - `-d`: Runs containers in the background.

4. **Access the Web App**:  
   Visit [http://localhost:8501](http://localhost:8501) in your browser.

5. **Check Logs** (optional):  
   ```bash
   docker-compose logs -f          # Stream logs from all containers
   docker-compose logs -f web      # Only logs from the web container
   docker-compose logs -f model    # Only logs from the model container
   ```

6. **Shut Down** (optional):  
   ```bash
   docker-compose down             # Stop containers
   docker-compose down --volumes   # Also remove volume data (DB logs)
   ```

---

## 🌐 Production Deployment

*(Placeholder instructions — adapt for your environment.)*

1. **Set Up a VPS** (e.g., Hetzner) with Docker & Docker Compose installed.  
2. **Clone this repository** onto the server.  
3. **Create a `.env` file** with production configuration (secure passwords, etc.).  
4. **Run**:
   ```bash
   docker-compose up --build -d
   ```
5. **Expose Ports / Configure Firewall** to allow public access (e.g., port 8501 or a reverse proxy at 80/443).
6. **(Recommended) Use a Reverse Proxy** (Nginx, Caddy, etc.) for HTTPS and domain routing.

---

## 🏋️ Development Process

### 1. Model Development
- CNN Model (`model.py`) built with PyTorch, trained via `train.py`.  
- Data augmentation, evaluation, and temperature scaling are implemented in `utils/`.  
- Achieves over **99% accuracy** on MNIST, so extensive hyperparameter tuning was less critical (though you can modify `train.py` for your own experiments).

### 2. Model Serving
- Flask app (`model/app.py`) loads the trained `.pt` file.  
- Provides `/predict` and `/health` endpoints for inference and status checks.  
- Uses Gunicorn in production for robust concurrency handling.

### 3. Web Interface
- Streamlit application (`web/app.py`) for user interaction.  
- Supports drawing with a [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) component.  
- Basic light/dark theme toggle (future improvements planned).

### 4. Database Integration
- A PostgreSQL instance logs predictions: digit, confidence, timestamp, user feedback, and more.  
- SQL schema initialization in `database/init.sql`, run automatically at container startup.

### 5. Containerization
- Each component (model, web, db) runs in its own Docker container.  
- Docker Compose manages networking, environment variables, and persistent volumes.

### 6. Performance & Apple Silicon MPS
- **MPS acceleration** on Apple Silicon drastically cuts training and inference times.  
- Additional speedups possible with CUDA on compatible GPUs or further model optimization.

---

## 📚 Documentation

The repository includes various documents in `docs/` (Engineering Requirements, Project Design, etc.) plus in-depth READMEs within each component folder:

- [Model README](model/README.md)  
- [Web README](web/README.md)  
- [Database README](database/README.md)  

---

## 🚀 Future Enhancements

- **User Authentication**: Restrict or track usage per user.  
- **Retraining Pipeline**: Automatically retrain the model using user-provided true labels.  
- **Advanced Analytics**: Additional dashboards for real-time performance monitoring.  
- **Support for Other Datasets**: Expand beyond MNIST.  
- **CI/CD**: Automated testing, builds, and deployments (GitHub Actions, Jenkins, etc.).  
- **More Robust Theming**: Expand the basic dark/light toggle with more sophisticated UI options.

---

## 🙏 Contributing

Contributions are welcome! Please feel free to open issues or submit Pull Requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is distributed under the **MIT License**. See [LICENSE](./LICENSE) for more information.

---

## ✨ Acknowledgements

- **PyTorch** & **TorchVision** for the ML framework  
- **MNIST Dataset** by Yann LeCun  
- **Streamlit** for the UI framework  
- **Flask** & **Gunicorn** for the Python web service layer  
- **PostgreSQL** for data management  
- **Docker** for containerization  

---

*Thank you for checking out the MNIST Digit Classifier! We hope you find it useful and educational.*