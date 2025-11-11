import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.schema import CreateTable

Base = declarative_base()

class TrackEmbedding(Base):
    __tablename__ = 'track_embedding'
    id = Column(Integer, primary_key=True)
    track_id = Column(String, unique=True)
    embedding = Column(String)  

DATABASE_URL = os.getenv("POSTGRES_URL")
if not DATABASE_URL:
    print("Error: POSTGRES_URL environment variable not set.")
    sys.exit(1)

def load_embeddings_to_pgvector():
    df = pd.read_parquet("features.parquet")
    embeddings = np.load("embeddings.npy")

    # Create SQLAlchemy engine for query generation
    engine = create_engine(DATABASE_URL)

    # Connect to PostgreSQL with psycopg2 for execution
    conn = psycopg2.connect(dsn=DATABASE_URL)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    create_table_sql = str(CreateTable(TrackEmbedding.__table__).compile(engine)).replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS')
    cur.execute(create_table_sql)
    conn.commit()

    # Prepare data for batch insert
    data = []
    for idx, row in df.iterrows():
        track_id = str(row['track_id'])
        emb = embeddings[idx]
        emb_str = '[' + ','.join(str(float(v)) for v in emb) + ']'
        data.append((track_id, emb_str))

    execute_batch(
        cur,
        "INSERT INTO track_embedding (track_id, embedding) VALUES (%s, %s::vector) ON CONFLICT (track_id) DO NOTHING",
        data,
        page_size=1000
    )
    conn.commit()

    cur.close()
    conn.close()
    print(f"Successfully loaded {len(data)} embeddings into pgvector.")

if __name__ == "__main__":
    load_embeddings_to_pgvector()