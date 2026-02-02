import os
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cursor = conn.cursor()

# Create table once
cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id SERIAL PRIMARY KEY,
    platform TEXT,
    user_id TEXT,
    username TEXT,
    question TEXT,
    answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

def save_chat(platform, user_id, username, question, answer):
    cursor.execute(
        """
        INSERT INTO chats (platform, user_id, username, question, answer)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (platform, user_id, username, question, answer)
    )
