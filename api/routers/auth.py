from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from jose import jwt
from ..schemas.schemas import get_db
from ..schemas.schemas import User

import os
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN_EXPIRE_MINUTES = 9999
SALT = os.getenv("SALT", "m2-boards")

security = HTTPBearer()

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

class RegisterRequest(BaseModel):
    login: str
    password: str
    email: Optional[str] = None


class LoginRequest(BaseModel):
    login: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str
    email: Optional[str]


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("SECRET_KEY", "your-secret-key-here"), algorithm="HS256")
    return encoded_jwt


def hash_password(password: str, salt: str = SALT) -> tuple[str, str]:
    """Hash a password with a random salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(
        'utf-8'), salt.encode('utf-8'), 100000)
    return hashed.hex()


@router.post("/register", response_model=TokenResponse)
def register(user_data: RegisterRequest, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(
        User.login == user_data.login).first()
    if existing_user:
        raise HTTPException(
            status_code=400, detail="User with this login already exists")

    # Check if email already exists (if provided)
    if user_data.email:
        existing_email = db.query(User).filter(
            User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=400, detail="User with this email already exists")

    # Hash the password
    hashed_password = hash_password(user_data.password)

    # Create new user
    new_user = User(
        login=user_data.login,
        email=user_data.email,
        hashed_password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Create access token for the new user
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.login}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": new_user.id,
        "username": new_user.login,
        "email": new_user.email
    }


@router.post("/login", response_model=TokenResponse)
def login(user_data: LoginRequest, db: Session = Depends(get_db)):
    # Find user by login or email
    user = db.query(User).filter((User.login == user_data.login) | (User.email == user_data.login)).first()

    if not user:
        print('this login')
        raise HTTPException(
            status_code=400, detail="Invalid login or password")

    # Hash the provided password with the stored salt
    hashed_password = hash_password(user_data.password)

    # Check if passwords match
    if hashed_password != user.hashed_password:
        print('this pass')
        raise HTTPException(
            status_code=400, detail="Invalid login or password")

    print(f"Creating token for user login: {user.login}")
    # Create access token for the user (always use login in token)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.login}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.login,
        "email": user.email
    }

# @router.get("/check")
# def check():
#     return {
#         "message": "Ok"
#     }
