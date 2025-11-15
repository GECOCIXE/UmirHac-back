from fastapi import HTTPException, Request, Depends
from sqlalchemy.orm import Session
from sqlalchemy import or_
from ..schemas.schemas import User, Folder, get_db


def get_current_user(request: Request):
    """
    Извлекает информацию о текущем пользователе из JWT-токена
    """
    if not hasattr(request.state, 'user') or request.state.user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Получаем пользователя из базы данных по логину или email
    db_gen = get_db()
    db = next(db_gen)
    try:
        # Печатаем отладочную информацию
        print(f"Looking for user with identifier: {request.state.user}")

        # Сначала ищем по логину
        user = db.query(User).filter(User.login == request.state.user).first()
        if user:
            print(f"Found user by login: {user.login} (id: {user.id})")
            return user

        # Если по логину не нашли, ищем по email
        user = db.query(User).filter(User.email == request.state.user).first()
        if user:
            print(f"Found user by email: {user.login} (email: {user.email}, id: {user.id})")
            return user

        # Печатаем всех пользователей для отладки
        all_users = db.query(User).all()
        print(f"All users in DB: {[{'id': u.id, 'login': u.login, 'email': u.email} for u in all_users]}")
        raise HTTPException(status_code=401, detail="User not found")
    finally:
        db.close()


def get_folder_by_id(folder_id: int, user_id: int, db: Session):
    """
    Получает папку по ID и проверяет, принадлежит ли она пользователю
    """
    folder = db.query(Folder).filter(Folder.id == folder_id, Folder.user_id == user_id).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder