import os
import tempfile
from pathlib import Path

import essentia.standard as es
import psycopg
import requests
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, StandardScaler, Normalizer
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

from common import get_essentia_algos, _extract_features_from_signal, feature_schema

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

# Source S3/Rclone Configuration (Optional - for Hadoop/S3A access)
SOURCE_ENDPOINT = os.getenv("SOURCE_ENDPOINT")  # e.g., "localhost:8081"
SOURCE_ACCESS_KEY = os.getenv("SOURCE_ACCESS_KEY", "minioadmin")
SOURCE_SECRET_KEY = os.getenv("SOURCE_SECRET_KEY", "minioadmin")
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "music-data")

TRACK_LIMIT = int(os.getenv("TRACK_LIMIT", "0"))  # 0 means no limit
FEATURES_CACHE_PATH = os.getenv("FEATURES_CACHE_PATH", "data/features.parquet")


def process_partition_s3(iterator):
    """Process a partition of audio bytes (S3/Hadoop)."""
    algos = get_essentia_algos()

    for row in iterator:
        track_id = row.id
        audio_bytes = row.content

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                tmp_mp3.write(audio_bytes)
                mp3_path = tmp_mp3.name

            try:
                # Load and standardize
                y = es.MonoLoader(filename=mp3_path, sampleRate=22050)()

                # Trim
                sr = 22050
                duration_samples = len(y)
                target_samples = 30 * sr
                if duration_samples > target_samples:
                    mid = duration_samples // 2
                    start = mid - (target_samples // 2)
                    end = start + target_samples
                    y = y[start:end]

                # Extract
                result = _extract_features_from_signal(track_id, y, algos)
                yield result

            finally:
                Path(mp3_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"[ERROR] Failed to process track {track_id}: {e}")
            continue


def process_partition_http(iterator):
    """Process a partition of audio URLs (HTTP)."""
    algos = get_essentia_algos()
    session = requests.Session()

    for row in iterator:
        track_id = row.id
        location = row.location

        try:
            mp3_url = f"{FILE_SERVER_URL}/{location}"
            response = session.get(mp3_url, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                tmp_mp3.write(response.content)
                mp3_path = tmp_mp3.name

            try:
                y = es.MonoLoader(filename=mp3_path, sampleRate=22050)()

                sr = 22050
                duration_samples = len(y)
                target_samples = 30 * sr
                if duration_samples > target_samples:
                    mid = duration_samples // 2
                    start = mid - (target_samples // 2)
                    end = start + target_samples
                    y = y[start:end]

                result = _extract_features_from_signal(track_id, y, algos)
                yield result

            finally:
                Path(mp3_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"[ERROR] Failed to process track {track_id}: {e}")
            continue


def init_database_pre_load() -> bool:
    """
    Initialize PostgreSQL table: Create table, DROP index if exists to speed up insertion.
    """
    try:
        # Connect using psycopg3
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                # Create pgvector extension if not exists
                print("   Enabling pgvector extension...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create table if not exists
                print("   Creating track_embeddings table...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS track_embeddings (
                        track_id INTEGER PRIMARY KEY,
                        embedding vector(50) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Drop index to speed up bulk insert
                print("   Dropping HNSW index (if exists) to speed up insertion...")
                cur.execute("DROP INDEX IF EXISTS track_embeddings_embedding_idx;")

                conn.commit()
        return True

    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        return False


def finalize_database_post_load() -> bool:
    """
    Create HNSW index after data insertion.
    """
    try:
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                print("   Creating HNSW index for vector similarity search...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS track_embeddings_embedding_idx 
                    ON track_embeddings 
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

    # Configure Source S3/Rclone if provided (Per-bucket configuration)
    if SOURCE_ENDPOINT:
        print(
            f"\n[INFO] Configuring Source S3/Rclone bucket: {SOURCE_BUCKET} at {SOURCE_ENDPOINT}"
        )
        hadoop_conf.set(
            f"fs.s3a.bucket.{SOURCE_BUCKET}.endpoint", f"http://{SOURCE_ENDPOINT}"
        )
        hadoop_conf.set(f"fs.s3a.bucket.{SOURCE_BUCKET}.access.key", SOURCE_ACCESS_KEY)
        hadoop_conf.set(f"fs.s3a.bucket.{SOURCE_BUCKET}.secret.key", SOURCE_SECRET_KEY)
        hadoop_conf.set(f"fs.s3a.bucket.{SOURCE_BUCKET}.path.style.access", "true")
        hadoop_conf.set(
            f"fs.s3a.bucket.{SOURCE_BUCKET}.connection.ssl.enabled", "false"
        )  # Rclone usually http

    print("\n[1/6] Initialized Spark Session with S3A support")

    # Read tracks from Postgres
    print("\n[2/6] Reading tracks from Postgres...")
    jdbc_url = POSTGRES_URL.replace("postgresql://", "jdbc:postgresql://")

    # Optimization: Read in parallel using partitionColumn
    # 1. Get min/max ID to define partition bounds
    bounds_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option(
            "dbtable",
            "(SELECT min(id) as min_id, max(id) as max_id FROM tracks) as tmp",
        )
        .option("driver", "org.postgresql.Driver")
        .load()
    )

    tracks_df = None
    if bounds_df.count() > 0:
        row = bounds_df.first()
        min_id, max_id = row["min_id"], row["max_id"]

        if min_id is not None and max_id is not None:
            print(
                f"   Reading tracks with partitions: min_id={min_id}, max_id={max_id}, partitions=20"
            )
            tracks_df = (
                spark.read.format("jdbc")
                .option("url", jdbc_url)
                .option("driver", "org.postgresql.Driver")
                .option("dbtable", "tracks")
                .option("partitionColumn", "id")
                .option("lowerBound", min_id)
                .option("upperBound", max_id)
                .option("numPartitions", 20)
                .load()
            )

    # Fallback if table is empty or bounds failed
    if tracks_df is None:
        tracks_df = (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("driver", "org.postgresql.Driver")
            .option("dbtable", "tracks")
            .load()
        )

    track_count = tracks_df.count()
    print(f"   Loaded {track_count} tracks from database")

    if track_count == 0:
        print("   No tracks to process. Exiting.")
        spark.stop()
        return

    if TRACK_LIMIT > 0:
        tracks_df = tracks_df.limit(TRACK_LIMIT)
        print(f"   Limiting to first {TRACK_LIMIT} tracks for processing")

    # Select required columns
    tracks_df = tracks_df.select("id", "location")

    # Check for cached features to avoid re-running expensive extraction
    if os.path.exists(FEATURES_CACHE_PATH):
        print(f"\n[3/6] Loading cached features from {FEATURES_CACHE_PATH}...")
        features_df = spark.read.parquet(FEATURES_CACHE_PATH)
    else:
        print("\n[3/6] Extracting audio features...")

        if SOURCE_ENDPOINT:
            print(
                f"   Using Hadoop S3A to read audio files from s3a://{SOURCE_BUCKET}/"
            )

            # Optimization: If we have a limited set of tracks, pass specific paths to avoid scanning the whole bucket
            # This prevents reading content of files we don't need.
            use_specific_paths = track_count < 10000

            if use_specific_paths:
                print(
                    f"   [Optimization] Track count is small ({track_count}). Reading specific files only."
                )
                locations = [
                    row.location for row in tracks_df.select("location").collect()
                ]
                file_paths = [f"s3a://{SOURCE_BUCKET}/{loc}" for loc in locations]

                # ignoreMissingFiles ensures we don't crash if DB has a file that S3 doesn't
                audio_df = (
                    spark.read.format("binaryFile")
                    .option("ignoreMissingFiles", "true")
                    .load(file_paths)
                )
            else:
                # Read all audio files from the source bucket
                audio_df = (
                    spark.read.format("binaryFile")
                    .option("pathGlobFilter", "*.mp3")
                    .option("recursiveFileLookup", "true")
                    .load(f"s3a://{SOURCE_BUCKET}/")
                )

            # Create a 'location' column in audio_df by stripping the prefix
            prefix = f"s3a://{SOURCE_BUCKET}/"
            audio_df = audio_df.withColumn(
                "location", expr(f"substring(path, {len(prefix) + 1}, length(path))")
            )

            from pyspark.sql.functions import broadcast

            joined_df = audio_df.join(broadcast(tracks_df), "location")

            # Optimization: Use mapPartitions to amortize Essentia initialization cost
            # and avoid intermediate WAV writes.
            features_rdd = joined_df.rdd.mapPartitions(process_partition_s3)
            features_df = spark.createDataFrame(features_rdd, schema=feature_schema)

        else:
            if tracks_df.rdd.getNumPartitions() < 20:
                tracks_df = tracks_df.repartition(20)

            features_rdd = tracks_df.rdd.mapPartitions(process_partition_http)
            features_df = spark.createDataFrame(features_rdd, schema=feature_schema)

        # Save features to disk for future runs
        print(f"   Saving extracted features to {FEATURES_CACHE_PATH}...")
        features_df.write.mode("overwrite").parquet(FEATURES_CACHE_PATH)

        # Reload from disk to break lineage and avoid re-computation
        features_df = spark.read.parquet(FEATURES_CACHE_PATH)

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
        array_to_vector(col("features")).alias("feature_vector", metadata={"numFeatures": num_features}),
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
        print("\n[5/7] Writing embeddings to PostgreSQL...")

        try:
            # Write to PostgreSQL with pgvector using native Spark JDBC
            # Cast the vector column to string because pgvector accepts '[1.0,2.0,...]' format
            # Use overwrite mode with truncate=true to preserve the schema (vector type)
            # IMPORTANT: Specify stringtype=unspecified to allow Postgres to cast the string to vector
            result_df.select("track_id", col("embedding").cast("string")).write.format(
                "jdbc"
            ).option("url", jdbc_url).option("dbtable", "track_embeddings").option(
                "driver", "org.postgresql.Driver"
            ).option("truncate", "true").option("stringtype", "unspecified").mode(
                "overwrite"
            ).save()

            print("   [SUCCESS] Wrote embeddings to track_embeddings table")

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
