import os
import psycopg2
import logging

# Load DATABASE URL
DATABASE_URL = os.getenv("DATABASE_URL")

logging.info(f"USING DB: {DATABASE_URL}")


# ================= CONNECTION =================

def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


# ================= INIT DB =================

def init_db():
    try:
        conn = get_conn()
        cur = conn.cursor()

        # chats table
        cur.execute("""
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

        # users table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        cur.close()
        conn.close()

        logging.info("✅ Database initialized")

    except Exception as e:
        logging.error(f"❌ DB INIT ERROR: {e}")


# ================= CHAT =================

def save_chat(platform, user_id, session_id, question, answer):
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO chats (platform, user_id, session_id, question, answer)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (platform, user_id, session_id, question, answer)
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("SAVE CHAT ERROR:", e)  


# ================= USERS =================

def create_user(username, password):
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO users (username, password)
            VALUES (%s, %s)
            """,
            (username, password)
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f"❌ CREATE USER ERROR: {e}")
        raise


def get_user(username):
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT username, password FROM users WHERE username=%s
            """,
            (username,)
        )

        user = cur.fetchone()

        cur.close()
        conn.close()

        return user

    except Exception as e:
        logging.error(f"❌ GET USER ERROR: {e}")
        return None