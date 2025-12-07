# Seamless Playlist Generator

A distributed system that generates seamless music playlists by analyzing audio characteristics. It uses deep learning models to create vector embeddings of uploaded tracks and finds similar songs using vector similarity search.

## Features

- **MP3 Upload & Processing**: Upload audio files directly through a modern web interface.
- **Intelligent Analysis**: Uses deep learning models to extract audio features and generate embeddings.
- **Similarity Search**: Finds semantically similar tracks using `pgvector` for vector database operations.
- **Distributed Architecture**: Scalable microservices architecture using Docker, Celery, and Apache Spark.
- **Real-time Status**: Track upload and processing status in real-time.

## Architecture

The project is built as a set of containerized microservices:

- **Frontend**: React application (Vite) that provides the user interface for uploads and playlist management.
- **Backend API**: FastAPI service managing file uploads, task dispatching, and database interactions.
- **Worker**: Celery workers handling background tasks, orchestrating the communication between storage, database, and inference services.
- **Database**: PostgreSQL with `pgvector` extension for storing track metadata and high-dimensional vector embeddings.
- **Inference Engine**: Apache Spark cluster for distributed model inference, served via an NGINX proxy.
- **Storage**: MinIO (S3-compatible) for checking file existence (configured in backend).
- **Message Broker**: Valkey (Redis compatible) for task queue management.

## Prerequisites

- **Docker** and **Docker Compose** installed on your machine.

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Seamless-Playlist-Generator
   ```

2. **Start the services**:
   Run the following command to build and start the entire stack:
   ```bash
   docker-compose up --build
   ```

   This will spin up the frontend, backend, database, spark cluster, and all supporting services.

3. **Access the Application**:
   Open your browser and navigate to:
   - **Frontend**: [http://localhost:8080](http://localhost:8080)

## Service Endpoints

- **Frontend**: `http://localhost:8080`
- **Backend API**: `http://localhost:8081`
- **Inference Proxy**: `http://localhost:8082`
- **Spark Master UI**: `http://localhost:8083`
- **Spark Worker UI**: `http://localhost:8084`
- **Spark History Server**: `http://localhost:18080`

## Development

The project is organized into three main directories:
- `frontend/`: React + Vite application.
- `backend/`: FastAPI application and Celery workers.
- `model_deployment/`: Spark cluster configuration and inference scripts.

Logs and data for Spark are persisted in the `model_deployment/spark-logs` and `model_deployment/data` directories.
