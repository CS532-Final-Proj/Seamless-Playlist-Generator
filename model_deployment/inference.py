import os
import tempfile
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
import concurrent.futures
from urllib.parse import urlparse

import pebble
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.functions import array_to_vector
from pyspark.sql.functions import col

# Import feature extraction logic from common.py
from common import init_worker, safe_extract_worker, feature_schema

# Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "embedding-model")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
FILE_SERVER_URL = os.getenv("FILE_SERVER_URL", "http://localhost:8080")

# Global Spark Session and Model
spark = None
model = None
pool = None


def init_spark():
    global spark, model, pool

    if spark is not None:
        return

    print("Initializing Spark Session...")
    spark = (
        SparkSession.builder.appName("MusicInferenceService")
        .master(
            "local[*]"
        )  # Run in local mode to avoid blocking the shared Spark cluster
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    # Configure S3A
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
    hadoop_conf.set("fs.s3a.access.key", MINIO_ACCESS_KEY)
    hadoop_conf.set("fs.s3a.secret.key", MINIO_SECRET_KEY)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", str(MINIO_SECURE).lower())
    hadoop_conf.set("fs.s3a.endpoint.region", "garage")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    print("Loading Spark ML Model from S3...")
    model_path = f"s3a://{MINIO_BUCKET}/spark_ml_model"
    try:
        model = PipelineModel.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        print("WARNING: Model not loaded. /predict will fail until model is available.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_spark()

    global pool
    print("Initializing Process Pool...")
    pool = pebble.ProcessPool(max_workers=6, max_tasks=50, initializer=init_worker)

    yield
    # Shutdown
    if pool:
        print("Shutting down Process Pool...")
        pool.close()
        pool.join()

    if spark:
        spark.stop()


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    location: str


class PredictResponse(BaseModel):
    embedding: List[float]


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    global model, pool

    print(f"Received prediction request for location: {request.location}")

    if model is None:
        print("Model not loaded, attempting to reload...")
        try:
            model_path = f"s3a://{MINIO_BUCKET}/spark_ml_model"
            model = PipelineModel.load(model_path)
        except Exception:
            raise HTTPException(status_code=503, detail="Model not loaded")

    location = request.location

    tmp_mp3_path = None
    try:
        parsed_url = urlparse(location)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = tmp_mp3.name

            if parsed_url.scheme in ("s3", "s3a"):
                s3_path = location.replace("s3://", "s3a://")
                try:
                    df = spark.read.format("binaryFile").load(s3_path)
                    row = df.select("content").first()
                    if row is None:
                        raise HTTPException(
                            status_code=404, detail="File not found in S3"
                        )
                    tmp_mp3.write(row["content"])
                except Exception as e:
                    print(f"Error loading from S3: {e}")
                    raise HTTPException(
                        status_code=400, detail=f"Failed to load from S3: {str(e)}"
                    )
            elif parsed_url.scheme in ("http", "https"):
                response = requests.get(location, timeout=30)
                response.raise_for_status()
                tmp_mp3.write(response.content)
            else:
                # Fallback to File Server for relative paths
                url = f"{FILE_SERVER_URL}/{location}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                tmp_mp3.write(response.content)

        # Isolate Essentia extraction in a separate process
        future = pool.schedule(safe_extract_worker, args=[0, tmp_mp3_path], timeout=30)
        try:
            features_dict = future.result()
        except concurrent.futures.TimeoutError:
            raise HTTPException(status_code=504, detail="Feature extraction timed out")
        except pebble.ProcessExpired:
            raise HTTPException(
                status_code=500, detail="Feature extraction worker crashed"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Feature extraction failed: {str(e)}"
            )

        if not features_dict or "features" not in features_dict:
            raise HTTPException(status_code=500, detail="Failed to extract features")

        features_list = features_dict["features"]

        rows = [(0, features_list)]
        df = spark.createDataFrame(rows, schema=feature_schema)

        num_features = 80
        df = df.withColumn(
            "feature_vector",
            array_to_vector(col("features")).alias(
                "feature_vector", metadata={"numFeatures": num_features}
            ),
        )

        prediction = model.transform(df)

        result_row = prediction.select("embedding").first()
        embedding = result_row["embedding"].toArray().tolist()

        return PredictResponse(embedding=embedding)

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch audio: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_mp3_path:
            Path(tmp_mp3_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
