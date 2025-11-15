# back/api/db_models.py
# (Новый файл: SQLAlchemy модели для БД)

from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Enum
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

# Настройка БД (SQLite для примера)
DATABASE_URL = "sqlite:///./umir_db.sqlite"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums для статусов
# class ResumeAnalysisStatus(str, enum.Enum):
#     suitable = "suitable"
#     not_suitable = "not_suitable"
#     analyzing = "analyzing"

# class CallStatus(str, enum.Enum):
#     not_planned = "not_planned"
#     planned = "planned"
#     in_progress = "in_progress"
#     completed = "completed"

class ProjectStatus(str, enum.Enum):
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    login = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)  # Новый параметр
    hashed_password = Column(String, nullable=False)


class Folder(Base):
    __tablename__ = "folders"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)  # Название папки
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    archived = Column(Boolean, default=False)  # Статус архивности папки
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Связь с пользователем
    user = relationship("User", backref="folders")


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)  # Название проекта
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=True)  # ID папки, к которой принадлежит проект
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(Enum(ProjectStatus), nullable=False, default=ProjectStatus.in_progress)  # Статус генерации сценария
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    result_path = Column(String, nullable=True)  # Путь к сгенерированному JSON файлу
    product_description = Column(Text, nullable=True)  # Описание продукта (промпт для генерации сценария)
    image_generation_status = Column(Text, nullable=True)  # Статус генерации изображений в JSON формате
    image_paths = Column(Text, nullable=True)  # Все пути к изображениям в JSON формате
    image_descriptions = Column(Text, nullable=True)  # Все описания изображений в JSON формате

    # Связь с пользователем и папкой
    user = relationship("User", backref="projects")
    folder = relationship("Folder", backref="projects")


class ScenarioElementImage(Base):
    __tablename__ = "scenario_element_images"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    element_index = Column(Integer, nullable=False)  # Индекс элемента в сценарии
    image_path = Column(String, nullable=True)  # Путь к изображению для конкретного элемента
    image_description = Column(Text, nullable=True)  # Описание для генерации изображения
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    status = Column(Enum(ProjectStatus), nullable=False, default=ProjectStatus.in_progress)

    # Связь с проектом
    project = relationship("Project", backref="scenario_element_images")


# Создание таблиц
Base.metadata.create_all(bind=engine)

# Dependency для сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Экспортируем модели, чтобы их можно было импортировать
__all__ = ["get_db", "User", "Folder", "Project", "ProjectStatus", "ScenarioElementImage"]