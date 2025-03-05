# MNIST Digit Classifier

![GitHub contributors](https://img.shields.io/github/contributors/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![Forks](https://img.shields.io/github/forks/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![Issues](https://img.shields.io/github/issues/YuriiOks/mnist-digit-classifier?style=for-the-badge)
![License](https://img.shields.io/github/license/YuriiOks/mnist-digit-classifier?style=for-the-badge)

## Overview

This project implements an end-to-end machine learning application that recognizes hand-drawn digits using a model trained on the MNIST dataset. The application features a web interface where users can draw digits, receive predictions, and provide feedback about prediction accuracy.

![Application Screenshot](https://via.placeholder.com/800x400?text=MNIST+Digit+Classifier)

**Live Demo:** [http://65.108.44.28](http://65.108.44.28)

## Features

- **Interactive Drawing Canvas**: Draw digits directly in your browser
- **Real-time Prediction**: Instantly receive predictions with confidence scores
- **Feedback Mechanism**: Provide the true label to help evaluate model performance
- **Prediction History**: View past predictions and user-provided feedback
- **Fully Containerized**: Complete application packaged with Docker
- **End-to-End ML Pipeline**: From model training to deployment

## Technology Stack

- **Machine Learning**: PyTorch for model development and inference
- **Frontend**: Streamlit for interactive web application
- **Database**: PostgreSQL for storing prediction logs
- **Containerization**: Docker and Docker Compose
- **Deployment**: Self-managed VPS (Hetzner)

## Project Structure

The project is organized into three main components:

- **Model**: PyTorch CNN model trained on MNIST dataset
- **Web**: Streamlit application providing the user interface
- **Database**: PostgreSQL database for logging predictions

```
mnist-digit-classifier/
├── model/               # Model training and inference service
├── web/                 # Streamlit web application
├── database/            # Database initialization scripts
├── docker-compose.yml   # Docker Compose configuration
└── docs/                # Project documentation
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. Start the application with Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Access the web application at [http://localhost:8501](http://localhost:8501)

### Production Deployment

1. Set up a VPS with Docker and Docker Compose installed
2. Clone the repository on the server
3. Create a `.env` file with production settings
4. Start the application with Docker Compose:
   ```bash
   docker-compose up -d
   ```

5. Configure firewall settings to allow access to port 8501

## Development Process

### 1. Model Development

The model is a Convolutional Neural Network (CNN) trained on the MNIST dataset using PyTorch. The training script handles data preprocessing, model architecture definition, training, and model saving.

### 2. Web Interface

The Streamlit application provides an intuitive interface for users to interact with the model. It includes:
- A drawing canvas for digit input
- Real-time prediction display
- User feedback collection
- History visualization

### 3. Database Integration

PostgreSQL stores prediction logs including:
- Timestamp
- Predicted digit
- User-provided true label
- Prediction confidence

### 4. Containerization

The application is containerized using Docker with separate containers for each component:
- Model service container
- Web application container
- Database container

Docker Compose orchestrates the multi-container setup.

### 5. Deployment

The application is deployed on a Hetzner VPS with:
- Docker and Docker Compose installed
- Port 8501 exposed for public access
- Data persistence through Docker volumes

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Engineering Requirements Document](docs/ERD.md)
- [Project Design Document](docs/PDD.md)
- [Project Structure Document](docs/structure.md)
- [Database Entity Relationship Diagram](docs/diagrams/database_erd.md)

## Future Enhancements

- User authentication system
- Model retraining based on user feedback
- Advanced analytics dashboard
- Support for multiple models/datasets
- CI/CD pipeline for automated deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PyTorch](https://pytorch.org/) for machine learning framework
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for training data
- [Streamlit](https://streamlit.io/) for web interface development
- [PostgreSQL](https://www.postgresql.org/) for database management
- [Docker](https://www.docker.com/) for containerization