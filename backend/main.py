from dotenv import load_dotenv

load_dotenv()

# ruff: noqa: E402
import logging
import os
import uuid
from contextlib import asynccontextmanager

from db import init_models, get_db, AsyncSession, get_playlist_by_id
import minio
from fastapi import FastAPI, Depends
from tasks import run_process_upload
from utils import create_presigned_post

S3_BUCKET = os.getenv("S3_BUCKET")

s3_client = minio.Minio(
    endpoint=os.getenv("S3_ENDPOINT"),
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("S3_REGION"),
    secure=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logging.info("API is starting up...")

    # Initialize database models
    await init_models()
    logging.info("Database models initialized.")

    yield

    # Cleanup tasks

    logging.info("API is shutting down...")


app = FastAPI(lifespan=lifespan)


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


@app.post("/uploads/{upload_id}")
async def process_upload(upload_id: str):
    logging.info(f"Processing upload with ID: {upload_id}")

    # Check if the file exists in S3
    try:
        s3_client.stat_object(S3_BUCKET, f"{upload_id}.mp3")
    except minio.error.S3Error as e:
        logging.error(f"File not found in S3: {e}")
        return {"error": "Upload not found."}

    # Enqueue task
    run_process_upload.apply_async(args=[upload_id], task_id=upload_id)

    return {"message": f"Processing upload with ID: {upload_id}"}


@app.get("/results/{upload_id}")
async def get_results(upload_id: str, db: AsyncSession = Depends(get_db)):
    logging.info(f"Fetching results for upload ID: {upload_id}")

    res = run_process_upload.AsyncResult(task_id=upload_id)
    res.ready()

    if res.successful():
        playlist = await get_playlist_by_id(db, upload_id)

        if playlist:
            logging.debug(f"Fetched playlist: {playlist}")

        return {"status": "completed", "result": res.result}
    elif res.failed():
        return {"status": "failed", "error": str(res.result)}
    else:
        return {"status": "processing"}
