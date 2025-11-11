import logging
import os
from time import sleep

import minio
from worker import celery

s3_client = minio.Minio(
    endpoint=os.getenv("S3_ENDPOINT"),
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("S3_REGION"),
    secure=False,
)


def fetch_file_from_s3(upload_id: str):
    logging.info(f"Fetching file for upload ID: {upload_id} from S3")

    response = s3_client.get_object(
        bucket_name=os.getenv("S3_BUCKET"),
        object_name=f"{upload_id}.mp3",
    )
    file_data = response.read()
    response.close()
    response.release_conn()

    logging.debug(f"Fetched file of size: {len(file_data)} bytes")

    return file_data


@celery.task
def run_process_upload(upload_id: str):
    logging.info(f"Processing upload: {upload_id}")

    # Fetch file from S3
    file = fetch_file_from_s3(upload_id)

    sleep(30)  # Simulate processing time

    # TODO: Add model inference and processing logic here

    logging.info(f"Completed processing for upload: {upload_id}")

    return {"status": "completed", "upload_id": upload_id}
