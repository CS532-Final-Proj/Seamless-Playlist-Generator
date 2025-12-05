import logging
import os

import requests
from psycopg_pool import ConnectionPool
from worker import celery
from celery.signals import worker_process_init


S3_BUCKET = os.getenv("S3_BUCKET")
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000/predict")
DATABASE_URL = os.getenv("POSTGRES_URL", "")

pool = None


@worker_process_init.connect
def init_worker(**kwargs):
    global pool
    pool = ConnectionPool(
        conninfo=DATABASE_URL, min_size=1, max_size=10, kwargs={"prepare_threshold": 0}
    )
    pool.open(True, timeout=4)


@celery.task
def run_process_upload(upload_id: str):
    logging.info(f"Processing upload: {upload_id}")

    # Get embedding from inference API
    print(
        f"Requesting embedding for upload {upload_id} from {INFERENCE_API_URL} with S3 path s3://{S3_BUCKET}/{upload_id}.mp3"
    )
    try:
        response = requests.post(
            INFERENCE_API_URL,
            json={"location": f"s3://{S3_BUCKET}/{upload_id}.mp3"},
            timeout=60,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        logging.info(f"Received embedding for upload {upload_id}: {embedding[:5]}...")
    except requests.RequestException as e:
        logging.error(f"Failed to get embedding for upload {upload_id}: {e}")
        return {"status": "error", "message": str(e)}

    # Get similar tracks
    print(f"Finding similar tracks for upload {upload_id}")

    similar_tracks = []
    try:
        emb_str = str(embedding)

        with pool.connection() as conn:
            with conn.cursor() as cur:
                # 1. Create the Playlist
                cur.execute(
                    "INSERT INTO playlists (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                    (upload_id,),
                    prepare=True,
                )

                # 2. Find similar tracks
                cur.execute(
                    """
                    SELECT track_id, (embedding <-> %(emb)s::vector) AS similarity
                    FROM track_embeddings
                    ORDER BY embedding <-> %(emb)s::vector
                    LIMIT 5
                """,
                    {"emb": emb_str},
                    prepare=True,
                )

                similar_tracks = cur.fetchall()
                logging.info(
                    f"Found {len(similar_tracks)} similar tracks for {upload_id}"
                )

                # 3. Insert into PlaylistTrack
                for index, (track_id, similarity) in enumerate(similar_tracks):
                    import uuid

                    pt_id = str(uuid.uuid4())

                    cur.execute(
                        """
                        INSERT INTO playlist_tracks (id, playlist_id, track_id, "order")
                        VALUES (%s, %s, %s, %s)
                        """,
                        (pt_id, upload_id, track_id, index),
                        prepare=True,
                    )

    except Exception as e:
        logging.error(f"Database error for upload {upload_id}: {e}")
        return {"status": "error", "message": str(e)}

    logging.info(f"Completed processing for upload: {upload_id}")

    return {
        "status": "completed",
        "upload_id": upload_id,
        "similar_tracks": similar_tracks,
    }
