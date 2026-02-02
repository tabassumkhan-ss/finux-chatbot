import os
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cursor = conn.cursor()

# ----------------------------
# Chats table
# ----------------------------
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

# ----------------------------
# Questions table
# ----------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id SERIAL PRIMARY KEY,
    question TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ----------------------------
# Save chat (full conversation)
# ----------------------------
def save_chat(platform, user_id, username, question, answer):
    cursor.execute(
        """
        INSERT INTO chats (platform, user_id, username, question, answer)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (platform, user_id, username, question, answer)
    )

# ----------------------------
# Save only questions
# ----------------------------
def save_question(question):
    cursor.execute(
        """
        INSERT INTO questions (question)
        VALUES (%s)
        """,
        (question,)
    )
