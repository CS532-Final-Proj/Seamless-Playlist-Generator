from dotenv import load_dotenv

load_dotenv()

# ruff: noqa: E402
import logging
import os
import uuid
from contextlib import asynccontextmanager

from db import init_models, get_db, AsyncSession, get_playlist_by_id
import minio
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tasks import run_process_upload
# from utils import create_presigned_post

S3_BUCKET = os.getenv("S3_BUCKET", "532")
FILE_SERVER_URL = os.getenv("FILE_SERVER_URL", "http://localhost:5050")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# @app.get("/upload")
# async def get_upload_params():
#     upload_id = str(uuid.uuid4())
#     return {
#         "upload_url": create_presigned_post(
#             s3_client=s3_client, bucket_name=S3_BUCKET, object_name=f"{upload_id}.mp3"
#         ),
#         "upload_id": upload_id,
#     }


# @app.post("/uploads/{upload_id}")
# async def process_upload(upload_id: str):
#     logging.info(f"Processing upload with ID: {upload_id}")

#     # Check if the file exists in S3
#     try:
#         s3_client.stat_object(S3_BUCKET, f"{upload_id}.mp3")
#     except minio.error.S3Error as e:
#         logging.error(f"File not found in S3: {e}")
#         return {"error": "Upload not found."}

#     # Enqueue task
#     run_process_upload.apply_async(args=[upload_id], task_id=upload_id)

#     return {"message": f"Processing upload with ID: {upload_id}"}


# @app.get("/results/{upload_id}")
# async def get_results(upload_id: str, db: AsyncSession = Depends(get_db)):
#     logging.info(f"Fetching results for upload ID: {upload_id}")

#     res = run_process_upload.AsyncResult(task_id=upload_id)
#     res.ready()

#     if res.successful():
#         playlist = await get_playlist_by_id(db, upload_id)

#         if playlist:
#             logging.debug(f"Fetched playlist: {playlist}")

#         return {"status": "completed", "result": res.result}
#     elif res.failed():
#         return {"status": "failed", "error": str(res.result)}
#     else:
#         return {"status": "processing"}


@app.post("/api/upload-and-search")
async def upload_and_search(file: UploadFile = File(...)):
    """Upload MP3 to S3, enqueue processing task, and return upload ID for polling"""

    if not file.content_type or "audio" not in file.content_type:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an MP3 file."
        )

    upload_id = str(uuid.uuid4())

    try:
        logging.info(f"Uploading {file.filename} to S3 as {upload_id}.mp3")
        content = await file.read()

        import io

        s3_client.put_object(
            S3_BUCKET,
            f"{upload_id}.mp3",
            io.BytesIO(content),
            len(content),
            content_type="audio/mpeg",
        )

        logging.info(f"File uploaded successfully: {upload_id}.mp3")

        logging.info(f"Enqueueing processing task for {upload_id}")
        run_process_upload.apply_async(args=[upload_id], task_id=upload_id)

        return {"upload_id": upload_id, "status": "processing"}

    except Exception as e:
        logging.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/results/{upload_id}")
async def get_task_results(upload_id: str, db: AsyncSession = Depends(get_db)):
    """Poll for task results"""
    logging.info(f"Checking status for upload ID: {upload_id}")

    res = run_process_upload.AsyncResult(task_id=upload_id)

    if res.successful():
        playlist = await get_playlist_by_id(db, upload_id)

        if playlist and playlist.tracks:
            tracks = [
                {
                    "track_id": pt.track_id,
                    "title": pt.track.title if pt.track else "Unknown",
                    "order": pt.order,
                    "audio_url": f"{FILE_SERVER_URL}/{pt.track.location}"
                    if pt.track
                    else None,
                }
                for pt in playlist.tracks
            ]
            return {"status": "completed", "results": tracks}

        return {"status": "completed", "results": []}
    elif res.failed():
        return {"status": "failed", "error": str(res.result)}
    else:
        return {"status": "processing"}
