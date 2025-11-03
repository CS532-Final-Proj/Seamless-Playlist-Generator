from worker import celery
import logging


@celery.task
def run_process_upload(upload_id: str):
    logging.info(f"Processing upload: {upload_id}")

    # TODO: Add model inference and processing logic here
