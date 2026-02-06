import os
import psycopg2
import logging

DATABASE_URL = os.getenv("DATABASE_URL")

conn = None
cursor = None

if DATABASE_URL:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cursor = conn.cursor()

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

        logging.info("Database connected")

    except Exception as e:
        logging.error("DB connection failed:", e)

else:
    logging.warning("DATABASE_URL not set — running without DB")


def save_chat(platform, user_id, username, question, answer):
    if not cursor:
        return

    cursor.execute(
        """
        INSERT INTO chats (platform, user_id, username, question, answer)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (platform, user_id, username, question, answer)
    )


def save_question(question):
    pass
