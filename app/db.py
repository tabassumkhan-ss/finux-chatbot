import os
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cursor = conn.cursor()

# Create chats table
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

# QUESTIONS TABLE (if not already)
cursor.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id SERIAL PRIMARY KEY,
    question TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

def save_question(question):
    try:
        cursor.execute(
            "INSERT INTO questions (question) VALUES (%s)",
            (question,)
        )
    except Exception as e:
        print("save_question error:", e)

def save_chat(platform, user_id, username, question, answer):
    try:
        cursor.execute(
            """
            INSERT INTO chats (platform, user_id, username, question, answer)
            VALUES (%s,%s,%s,%s,%s)
            """,
            (
                platform or "web",
                str(user_id or "0"),
                username or "",
                question,
                answer,
            )
        )
    except Exception as e:
        print("save_chat error:", e)
