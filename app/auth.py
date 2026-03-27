from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from app.db import create_user, get_user
import logging

router = APIRouter()

# 🔐 SECRET KEY (change in production)
SECRET_KEY = "finux-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ================= MODELS =================

class UserSignup(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str


# ================= HELPERS =================

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ================= ROUTES =================

@router.post("/signup")
def signup(user: UserSignup):
    try:
        existing = get_user(user.username)

        if existing:
            raise HTTPException(status_code=400, detail="User already exists")

        hashed = hash_password(user.password)
        create_user(user.username, hashed)

        return {"message": "User created successfully"}

    except Exception as e:
        logging.error(f"SIGNUP ERROR: {e}")
        return {"error": str(e)}


@router.post("/login")
def login(user: UserLogin):

    db_user = get_user(user.username)

    if not db_user:
        # 🔥 AUTO SIGNUP
        hashed = hash_password(user.password)
        create_user(user.username, hashed)
    else:
        username, hashed_password = db_user

        if not verify_password(user.password, hashed_password):
            raise HTTPException(status_code=400, detail="Wrong password")

    token = create_token({"sub": user.username})

    return {"access_token": token}