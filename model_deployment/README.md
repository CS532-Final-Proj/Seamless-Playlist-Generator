# Model Deployment

## Overview
This directory contains the code and configuration files necessary for deploying the music embedding model. It includes scripts for feature extraction, model training, and inference.

## Contents
- `train.py`: Main script for training the music embedding model using Spark.
- `inference.py`: API server for generating music embeddings using the trained model.
- `common.py`: Common utility functions used by both training and inference scripts for feature extraction and processing.

## Setup
1. Ensure you have Docker and Docker Compose installed on your system.
2. Create a `model.env` file in this directory with the necessary environment variables (see `model.env.example` for reference).
3. Run `make up` from the root directory to build and start the Docker containers.

## Usage
- To train the model, run `make model`.
- To run inference, use `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"location": "https://example.com/song.mp3"}'`.