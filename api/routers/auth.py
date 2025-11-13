from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import hashlib
import secrets
from ..schemas.schemas import get_db
from ..schemas.schemas import User

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

SALT = "m2-board"

# Pydantic models for request/response


class RegisterRequest(BaseModel):
    login: str
    password: str


class LoginRequest(BaseModel):
    login: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


def hash_password(password: str, salt: str = SALT) -> tuple[str, str]:
    """Hash a password with a random salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(
        'utf-8'), salt.encode('utf-8'), 100000)
    return hashed.hex()


@router.post("/register", response_model=dict)
def register(user_data: RegisterRequest, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(
        User.login == user_data.login).first()
    if existing_user:
        raise HTTPException(
            status_code=400, detail="User with this login already exists")

    # Hash the password
    hashed_password = hash_password(user_data.password)

    # Create new user
    new_user = User(
        login=user_data.login,
        hashed_password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully",
        "user_id": new_user.id
    }


@router.post("/login")
def login(user_data: LoginRequest, db: Session = Depends(get_db)):
    # Find user by login
    user = db.query(User).filter(User.login == user_data.login).first()

    if not user:
        raise HTTPException(
            status_code=400, detail="Invalid login or password")

    # Hash the provided password with the stored salt
    hashed_password = hash_password(user_data.password)

    # Check if passwords match
    if hashed_password != user.hashed_password:
        raise HTTPException(
            status_code=400, detail="Invalid login or password")

    return {
        "message": "Login successful",
        "user_id": user.id
    }
    
# @router.get("/check")
# def check():
#     return {
#         "message": "Ok"
#     }
