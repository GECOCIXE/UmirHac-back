# back/api/db_models.py
# (Новый файл: SQLAlchemy модели для БД)

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Enum
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
    hashed_password = Column(String, nullable=False)


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(Enum(ProjectStatus), nullable=False, default=ProjectStatus.in_progress)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    result_path = Column(String, nullable=True)  # Путь к сгенерированному JSON файлу
    image_path = Column(String, nullable=True)  # Путь к основному изображению
    image_description = Column(Text, nullable=True)  # Описание изображения для генерации

    # Связь с пользователем
    user = relationship("User", backref="projects")


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