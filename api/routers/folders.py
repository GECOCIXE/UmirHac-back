from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from ..schemas.schemas import get_db, User, Folder, Project
from .dependencies import get_current_user

security = HTTPBearer()

router = APIRouter(
    prefix="/folders",
    tags=["folders"],
    dependencies=[Depends(security)]
)

class FolderRequest(BaseModel):
    name: str
    archived: Optional[bool] = False


class ProjectInfo(BaseModel):
    id: int
    name: Optional[str]
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    product_description: Optional[str] = None


class FolderResponse(BaseModel):
    id: int
    name: str
    user_id: int
    archived: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    projects: List[ProjectInfo]


class UpdateFolderRequest(BaseModel):
    name: Optional[str] = None
    archived: Optional[bool] = None


@router.post("/", response_model=FolderResponse)
async def create_folder(
    request: FolderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Создать новую папку
    """
    new_folder = Folder(
        name=request.name,
        user_id=current_user.id,
        archived=request.archived
    )
    db.add(new_folder)
    db.commit()
    db.refresh(new_folder)

    # Получаем проекты, связанные с этой папкой
    projects = db.query(Project).filter(Project.folder_id == new_folder.id).all()
    project_infos = [
        ProjectInfo(
            id=proj.id,
            name=proj.name,
            status=proj.status.value,
            created_at=proj.created_at,
            updated_at=proj.updated_at,
            product_description=proj.product_description
        )
        for proj in projects
    ]

    return FolderResponse(
        id=new_folder.id,
        name=new_folder.name,
        user_id=new_folder.user_id,
        archived=new_folder.archived,
        created_at=new_folder.created_at,
        updated_at=new_folder.updated_at,
        projects=project_infos
    )


@router.get("/", response_model=List[FolderResponse])
async def get_user_folders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить все папки пользователя с информацией о проектах
    """
    folders = db.query(Folder).filter(Folder.user_id == current_user.id).all()

    folders_response = []
    for folder in folders:
        # Получаем проекты, связанные с этой папкой
        projects = db.query(Project).filter(Project.folder_id == folder.id).all()
        project_infos = [
            ProjectInfo(
                id=proj.id,
                name=proj.name,
                status=proj.status.value,
                created_at=proj.created_at,
                updated_at=proj.updated_at,
                product_description=proj.product_description
            )
            for proj in projects
        ]

        folders_response.append(
            FolderResponse(
                id=folder.id,
                name=folder.name,
                user_id=folder.user_id,
                archived=folder.archived,
                created_at=folder.created_at,
                updated_at=folder.updated_at,
                projects=project_infos
            )
        )

    return folders_response


@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить информацию о папке и проектах в ней
    """
    folder = db.query(Folder).filter(Folder.id == folder_id, Folder.user_id == current_user.id).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    # Получаем проекты, связанные с этой папкой
    projects = db.query(Project).filter(Project.folder_id == folder.id).all()
    project_infos = [
        ProjectInfo(
            id=proj.id,
            name=proj.name,
            status=proj.status.value,
            created_at=proj.created_at,
            updated_at=proj.updated_at,
            product_description=proj.product_description
        )
        for proj in projects
    ]

    return FolderResponse(
        id=folder.id,
        name=folder.name,
        user_id=folder.user_id,
        archived=folder.archived,
        created_at=folder.created_at,
        updated_at=folder.updated_at,
        projects=project_infos
    )


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: int,
    request: UpdateFolderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Обновить папку
    """
    folder = db.query(Folder).filter(Folder.id == folder_id, Folder.user_id == current_user.id).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    # Обновляем поля, если они предоставлены
    if request.name is not None:
        folder.name = request.name
    if request.archived is not None:
        folder.archived = request.archived

    db.commit()
    db.refresh(folder)

    # Получаем проекты, связанные с этой папкой
    projects = db.query(Project).filter(Project.folder_id == folder.id).all()
    project_infos = [
        ProjectInfo(
            id=proj.id,
            name=proj.name,
            status=proj.status.value,
            created_at=proj.created_at,
            updated_at=proj.updated_at,
            product_description=proj.product_description
        )
        for proj in projects
    ]

    return FolderResponse(
        id=folder.id,
        name=folder.name,
        user_id=folder.user_id,
        archived=folder.archived,
        created_at=folder.created_at,
        updated_at=folder.updated_at,
        projects=project_infos
    )


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Удалить папку
    """
    folder = db.query(Folder).filter(Folder.id == folder_id, Folder.user_id == current_user.id).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    # Проверяем, есть ли проекты в папке
    projects_count = db.query(Project).filter(Project.folder_id == folder.id).count()
    if projects_count > 0:
        raise HTTPException(status_code=400, detail="Cannot delete folder with projects inside")

    db.delete(folder)
    db.commit()

    return {"message": "Folder deleted successfully"}