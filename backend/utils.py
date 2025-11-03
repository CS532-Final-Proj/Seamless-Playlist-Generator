from minio import Minio
from minio.error import S3Error
import logging
from datetime import timedelta


def create_presigned_post(
    s3_client: Minio,
    bucket_name,
    object_name,
):
    """Generate a presigned URL S3 POST request to upload a file"""

    try:
        response = s3_client.presigned_put_object(
            bucket_name, object_name, expires=timedelta(minutes=5)
        )
    except S3Error as e:
        logging.error(e)
        return None

    return response
