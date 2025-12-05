from locust import HttpUser, task


class EmbeddingsUser(HttpUser):
    """
    Load test the /predict endpoint for generating embeddings from MP3 files.

    Example:
        locust -f locustfile.py --host=http://localhost:8000
    """

    @task
    def retrieve_embedding(self):
        """Request embedding prediction for an MP3 file."""
        payload = {"location": "s3://532/019a705d-309c-76b5-bbd1-8614db77dd3b.mp3"}

        self.client.post("/predict", json=payload)
