import os
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
