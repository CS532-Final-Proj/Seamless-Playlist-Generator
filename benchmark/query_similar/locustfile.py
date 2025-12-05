import random
from locust import task
from locust.contrib.postgres import PostgresUser
import os


# Embedding dimension
EMBEDDING_DIM = 50


def generate_random_embedding() -> list[float]:
    """Generate a random embedding vector similar to real embeddings."""
    return [random.uniform(-0.5, 0.5) for _ in range(EMBEDDING_DIM)]


def embedding_to_pg_vector(embedding: list[float]) -> str:
    """Convert embedding list to PostgreSQL vector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


class QuerySimilarUser(PostgresUser):
    """
    Load test pgvector similarity queries on track_embeddings table.

    Requires environment variables for PostgreSQL connection:
        PGHOST, PGPORT, PGUSER, PGPASS, PGDB
    """

    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = os.getenv("PGPORT", "5432")
    PGDB = os.getenv("PGDB", "test_db")
    PGUSER = os.getenv("PGUSER", "postgres")
    PGPASS = os.getenv("PGPASS", "postgres")

    conn_string = f"postgresql://{PGUSER}:{PGPASS}@{PGHOST}:{PGPORT}/{PGDB}"

    @task
    def query_similar_tracks(self):
        """Query for similar tracks using pgvector cosine distance."""
        # Generate random embedding
        embedding = generate_random_embedding()
        emb_str = embedding_to_pg_vector(embedding)

        # Execute pgvector similarity query
        query = f"""
            SELECT track_id, (embedding <-> '{emb_str}'::vector) AS similarity
            FROM track_embeddings
            ORDER BY embedding <-> '{emb_str}'::vector
            LIMIT 10
        """
        self.client.execute_query(query)
