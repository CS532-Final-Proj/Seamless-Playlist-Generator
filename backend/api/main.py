import logging
import os
import uuid
from datetime import timedelta

import minio
from dotenv import load_dotenv
from fastapi import FastAPI
from minio.error import S3Error

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


def create_presigned_post(
    bucket_name,
    object_name,
):
    """Generate a presigned URL S3 POST request to upload a file"""

    # Generate a presigned S3 POST URL
    try:
        response = s3_client.presigned_put_object(
            bucket_name, object_name, expires=timedelta(minutes=5)
        )
    except S3Error as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response


@app.get("/upload")
async def get_upload_params():
    upload_id = str(uuid.uuid7())
    return {
        "upload_url": create_presigned_post(S3_BUCKET, f"{upload_id}.mp3"),
        "upload_id": upload_id,
    }
