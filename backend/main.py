import logging
import os
import uuid

import minio
from dotenv import load_dotenv
from fastapi import FastAPI

from tasks import run_process_upload
from utils import create_presigned_post

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")

s3_client = minio.Minio(
    endpoint=os.getenv("S3_ENDPOINT"),
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("S3_REGION"),
    secure=False,
)

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/upload")
async def get_upload_params():
    upload_id = str(uuid.uuid7())
    return {
        "upload_url": create_presigned_post(
            s3_client=s3_client, bucket_name=S3_BUCKET, object_name=f"{upload_id}.mp3"
        ),
        "upload_id": upload_id,
    }


@app.post("/process/{upload_id}")
async def process_upload(upload_id: str):
    logging.info(f"Processing upload with ID: {upload_id}")

    # Check if the file exists in S3
    try:
        s3_client.stat_object(S3_BUCKET, f"{upload_id}.mp3")
    except minio.error.S3Error as e:
        logging.error(f"File not found in S3: {e}")
        return {"error": "Upload not found."}

    # Enqueue task
    run_process_upload.delay(upload_id)

    return {"message": f"Processing upload with ID: {upload_id}"}
