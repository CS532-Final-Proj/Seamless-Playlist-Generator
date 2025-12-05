import os
import sys
import concurrent.futures
import pebble

import psycopg
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, Normalizer, StandardScaler
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from common import init_worker, safe_extract_worker

# Configuration from environment
POSTGRES_URL = os.getenv(
    "POSTGRES_URL", "postgresql://user:password@localhost:5432/music"
)
FILE_SERVER_URL = os.getenv("FILE_SERVER_URL", "http://localhost:8080")

# MinIO (Destination) Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "embeddings")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

STAGING = os.getenv("STAGING", "true").lower() == "true"
TRACK_LIMIT = int(os.getenv("TRACK_LIMIT", "0"))  # 0 means no limit
FEATURES_CACHE_PATH = os.getenv("FEATURES_CACHE_PATH", "data/features")
FAILED_TRACKS_PATH = os.getenv("FAILED_TRACKS_PATH", "data/failed_tracks")

processing_schema = StructType(
    [
        StructField("track_id", IntegerType(), False),
        StructField("features", ArrayType(FloatType()), True),
        StructField("error", StringType(), True),
    ]
)


def log(msg):
    """Log message to stdout for Spark executor visibility."""
    sys.stdout.write(f"[Executor] {msg}\n")
    sys.stdout.flush()


def process_partition_audio(iterator):
    """Process a partition of audio bytes sequentially reusing Essentia algos."""

    log("Starting process_partition_audio")

    try:
        # Use a persistent pool for this partition to isolate Essentia
        # max_tasks=50 restarts the worker periodically to clear memory leaks
        with pebble.ProcessPool(
            max_workers=1, max_tasks=50, initializer=init_worker
        ) as pool:
            count = 0
            for row in iterator:
                count += 1
                track_id = row.track_id
                audio_path = row.audio
                error = row.error

                log(f"Processing track {track_id}, path: {audio_path}, error: {error}")

                if error:
                    print(f"[WARNING] Upstream error for {track_id}: {error}")
                    yield {"track_id": track_id, "features": None, "error": error}
                    continue

                if not audio_path:
                    yield {
                        "track_id": track_id,
                        "features": None,
                        "error": "Empty audio",
                    }
                    continue

                try:
                    # Extract (in child process)
                    future = pool.schedule(
                        safe_extract_worker, args=[track_id, audio_path], timeout=240
                    )

                    try:
                        feat_result = future.result()
                        feat_result["error"] = None
                        yield feat_result
                    except pebble.ProcessExpired:
                        print(f"[FATAL] Worker segfaulted on {track_id}")
                        yield {
                            "track_id": track_id,
                            "features": None,
                            "error": "Worker Segfault",
                        }
                    except concurrent.futures.TimeoutError:
                        print(f"[ERROR] Timeout processing {track_id}")
                        yield {
                            "track_id": track_id,
                            "features": None,
                            "error": "Timeout",
                        }
                    except Exception as e:
                        print(f"[ERROR] Worker exception for {track_id}: {e}")
                        yield {"track_id": track_id, "features": None, "error": str(e)}

                except Exception as e:
                    print(f"[ERROR] System failed for {track_id}: {e}")
                    yield {"track_id": track_id, "features": None, "error": str(e)}

            log(f"Finished processing partition. Total rows: {count}")
    finally:
        pass


def init_database_pre_load() -> bool:
    """
    Initialize PostgreSQL table: Create table, DROP index if exists to speed up insertion.
    """
    table_name = "track_embeddings_staging" if STAGING else "track_embeddings"
    try:
        # Connect using psycopg3
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                # Create pgvector extension if not exists
                print("   Enabling pgvector extension...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create table if not exists
                print(f"   Creating {table_name} table...")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        track_id INTEGER PRIMARY KEY,
                        embedding vector(50) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Drop index to speed up bulk insert
                print(
                    f"   Dropping HNSW index (if exists) to speed up insertion on {table_name}..."
                )
                cur.execute(f"DROP INDEX IF EXISTS {table_name}_embedding_idx;")

                conn.commit()
        return True

    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        return False


def finalize_database_post_load() -> bool:
    """
    Create HNSW index after data insertion.
    """
    table_name = "track_embeddings_staging" if STAGING else "track_embeddings"
    try:
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                print(
                    f"   Creating HNSW index for vector similarity search on {table_name}..."
                )
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                    ON {table_name}
                    USING hnsw (embedding vector_cosine_ops);
                """)
                conn.commit()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create index: {e}")
        return False


def main():
    """Main entry point for embedding generation pipeline."""

    print("=" * 80)
    print("Starting Embedding Model Generation Pipeline (Spark ML)")
    print("=" * 80)

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("MusicEmbeddingGenerator")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.scheduler.mode", "FAIR")  # Enable concurrent job execution
        .getOrCreate()
    )

    # Configure S3A for MinIO (Destination)
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()

    # Global S3A settings (defaults)
    hadoop_conf.set("fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
    hadoop_conf.set("fs.s3a.access.key", MINIO_ACCESS_KEY)
    hadoop_conf.set("fs.s3a.secret.key", MINIO_SECRET_KEY)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", str(MINIO_SECURE).lower())
    hadoop_conf.set("fs.s3a.fast.upload", "true")
    hadoop_conf.set("fs.s3a.endpoint.region", "garage")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    print("\n[1/6] Initialized Spark Session with S3A support")

    # ---------------------------------------------------------
    # Step 2: Identify Delta (Incremental Update)
    # ---------------------------------------------------------
    print("\n[2/6] Identifying tracks to process (Delta)...")

    # 1. Read track metadata from Postgres (JDBC)
    jdbc_url = POSTGRES_URL.replace("postgresql://", "jdbc:postgresql://")
    metadata_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "tracks")
        .option("driver", "org.postgresql.Driver")
        .load()
        .select("id")
        .withColumnRenamed("id", "track_id")
    )

    total_tracks = metadata_df.count()
    print(f"   Total tracks in DB: {total_tracks}")

    if TRACK_LIMIT > 0:
        metadata_df = metadata_df.limit(TRACK_LIMIT)
        print(f"   Limiting to first {TRACK_LIMIT} tracks")

    # 2. Load existing features
    existing_features_df = None
    if os.path.exists(FEATURES_CACHE_PATH) and (
        os.path.isdir(FEATURES_CACHE_PATH) and len(os.listdir(FEATURES_CACHE_PATH)) > 0
    ):
        print(f"   Loading existing features from {FEATURES_CACHE_PATH}...")
        try:
            existing_features_df = spark.read.parquet(FEATURES_CACHE_PATH)
            print(f"   Found {existing_features_df.count()} existing feature records.")
        except Exception as e:
            print(f"   [WARNING] Failed to read existing cache: {e}")
            existing_features_df = None

    # 3. Calculate Delta
    if existing_features_df is not None:
        # Identify tracks with no features (features is None)
        failed_features_df = existing_features_df.filter(
            col("features").isNull()
        ).select("track_id")
        # Union with tracks that are not present in existing features
        tracks_to_process_df = metadata_df.join(
            existing_features_df,
            metadata_df.track_id == existing_features_df.track_id,
            how="left_anti",
        ).unionByName(failed_features_df)
    else:
        tracks_to_process_df = metadata_df

    count_to_process = tracks_to_process_df.count()
    print(f"   Tracks to process (Delta): {count_to_process}")

    # ---------------------------------------------------------
    # Step 3: Process Audio (Load & Extract)
    # ---------------------------------------------------------
    if count_to_process > 0:
        print(f"\n[3/6] Processing {count_to_process} tracks...")

        # Use Spark's binaryFile format to list files, but only select the path
        # This avoids reading the file content into memory
        AUDIO_MOUNT_PATH = os.getenv("AUDIO_MOUNT_PATH", "/opt/spark/data/tracks")

        print(f"   Scanning audio files in {AUDIO_MOUNT_PATH}...")

        # Read file paths
        # pathGlobFilter ensures we only look at mp3s
        # recursiveFileLookup allows nested folders if needed
        audio_files_df = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.mp3")
            .option("recursiveFileLookup", "true")
            .load(AUDIO_MOUNT_PATH)
            .select("path")
        )

        # Extract track_id from path using regex
        from pyspark.sql.functions import regexp_extract

        # Regex to find digits before .mp3 at the end of path, preceded by a slash
        # Example: /opt/data/tracks/000/000536.mp3 -> 000536
        audio_df = audio_files_df.withColumn(
            "track_id",
            regexp_extract(col("path"), r"/(\d+)\.mp3$", 1).cast(IntegerType()),
        ).withColumnRenamed(
            "path", "audio"
        )  # Rename path to audio to match expected schema

        # Filter to only include tracks we need to process (Delta)
        # Join with tracks_to_process_df
        processing_df = audio_df.join(tracks_to_process_df, "track_id", "inner")

        # Add empty error column to match schema expected by process_partition_audio
        from pyspark.sql.functions import lit

        processing_df = processing_df.withColumn("error", lit(None).cast(StringType()))

        # Repartition for concurrency
        processing_df = processing_df.repartition(48)

        features_rdd = processing_df.rdd.mapPartitions(process_partition_audio)
        results_df = spark.createDataFrame(features_rdd, schema=processing_schema)

        # Cache results
        results_df.cache()

        # Write success
        success_df = results_df.filter(col("error").isNull()).select(
            "track_id", "features"
        )
        success_df.write.mode("append").parquet(FEATURES_CACHE_PATH)

        # Write failure
        failed_df = results_df.filter(
            (col("error").isNotNull()) | (col("features").isNull())
        ).select("track_id", "error")
        if failed_df.count() > 0:
            failed_df.write.mode("append").parquet(FAILED_TRACKS_PATH)

        results_df.unpersist()
        print("   [Done] Processing complete.")

    else:
        print("   No new tracks to process. Using existing cache.")

    # 4. Final Load for Training
    print(f"   Reloading all features from {FEATURES_CACHE_PATH} for training...")
    features_df = spark.read.parquet(FEATURES_CACHE_PATH)

    if TRACK_LIMIT > 0:
        print(f"   Enforcing track limit of {TRACK_LIMIT} on training data...")
        features_df = features_df.limit(TRACK_LIMIT)

    features_df.cache()

    feature_count = features_df.count()
    print(f"   Successfully extracted features from {feature_count} tracks")

    if feature_count == 0:
        print("[ERROR] No features extracted. Exiting.")
        spark.stop()
        return

    # Convert feature arrays to Spark ML Vectors
    print("\n[4/6] Generating embeddings with Spark ML Pipeline...")

    # We calculate the total number of features:
    # 10 (scalars) + 20 (tempogram) + 12 (chroma_mean) + 12 (chroma_std) + 13 (mfcc_mean) + 13 (mfcc_std) = 80
    num_features = 80

    features_df = features_df.withColumn(
        "feature_vector",
        array_to_vector(col("features")).alias(
            "feature_vector", metadata={"numFeatures": num_features}
        ),
    )

    # Build Spark ML Pipeline
    # Step 1: StandardScaler for normalization
    scaler = StandardScaler(
        inputCol="feature_vector",
        outputCol="scaled_features",
        withStd=True,
        withMean=True,
    )

    # Step 2: PCA for dimensionality reduction
    pca = PCA(k=50, inputCol="scaled_features", outputCol="pca_features")

    # Step 3: Normalizer for unit length embeddings (L2)
    normalizer = Normalizer(inputCol="pca_features", outputCol="embedding", p=2.0)

    # Create and fit pipeline
    pipeline = Pipeline(stages=[scaler, pca, normalizer])
    model = pipeline.fit(features_df)

    # Transform the data
    result_df = model.transform(features_df)

    # Select final columns
    result_df = result_df.select("track_id", "embedding", "features")

    print(f"   Generated embeddings with {feature_count} tracks x 50 dimensions")

    # Extract PCA model for variance info
    pca_model = model.stages[1]
    explained_variance = sum(pca_model.explainedVariance.toArray())
    print(f"   Explained variance ratio: {explained_variance:.4f}")

    # Initialize database (Create table and DROP index for speed)
    postgres_ready = init_database_pre_load()
    if not postgres_ready:
        print("[WARNING] Database initialization failed. Skipping PostgreSQL write.")

    # Write to PostgreSQL if ready
    if postgres_ready:
        print("\n[5/7] Writing embeddings to PostgreSQL staging table...")

        jdbc_url = POSTGRES_URL.replace("postgresql://", "jdbc:postgresql://")

        try:
            # Write to PostgreSQL with pgvector using native Spark JDBC
            # Cast the vector column to string because pgvector accepts '[1.0,2.0,...]' format
            # Use overwrite mode with truncate=true to preserve the schema (vector type)
            # IMPORTANT: Specify stringtype=unspecified to allow Postgres to cast the string to vector
            result_df.select("track_id", col("embedding").cast("string")).write.format(
                "jdbc"
            ).option("url", jdbc_url).option(
                "dbtable", "track_embeddings_staging" if STAGING else "track_embeddings"
            ).option("driver", "org.postgresql.Driver").option(
                "truncate", "true"
            ).option("stringtype", "unspecified").mode("overwrite").save()

            print("   [SUCCESS] Wrote embeddings to track_embeddings_staging table")

            # Re-create index after bulk load
            finalize_database_post_load()

        except Exception as e:
            print(f"[ERROR] Failed to write embeddings to PostgreSQL: {e}")

    # Save artifacts to S3 (MinIO) using native Spark/Hadoop
    print("\n[6/7] Saving model artifacts to MinIO (Native S3)...")

    s3_base_path = f"s3a://{MINIO_BUCKET}"

    try:
        # 1. Save Spark ML Model directly to S3
        print(f"   Saving Spark ML model to {s3_base_path}/spark_ml_model...")
        model.write().overwrite().save(f"{s3_base_path}/spark_ml_model")

        # 2. Save Features directly to S3
        print(f"   Saving features to {s3_base_path}/features...")
        features_df.write.mode("overwrite").parquet(f"{s3_base_path}/features")

        # 3. Save Embeddings directly to S3
        print(f"   Saving embeddings to {s3_base_path}/embeddings...")
        result_df.select("track_id", "embedding").write.mode("overwrite").parquet(
            f"{s3_base_path}/embeddings"
        )

        print("\n" + "=" * 80)
        print("✓ Embedding Model Generation Complete!")
        print("=" * 80)
        print("Used number of tracks:", feature_count)
        print("\nArtifacts Summary:")
        print("\nEmbeddings written to PostgreSQL table: track_embeddings")
        print(f"\nModel artifacts saved to MinIO bucket: {MINIO_BUCKET}")
        print("  - spark_ml_model/ (Spark ML Pipeline)")
        print("  - features/ (Features)")
        print("  - embeddings/ (Embeddings & Track IDs)")

    except Exception as e:
        print(f"\n[ERROR] Failed to save artifacts to MinIO: {e}")

    # Stop Spark
    spark.stop()
    print("\nSpark session stopped.")


if __name__ == "__main__":
    main()
