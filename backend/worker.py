from celery import Celery
import os

from dotenv import load_dotenv

load_dotenv()

celery = Celery("worker")

celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)
