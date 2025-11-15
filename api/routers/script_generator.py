from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import os
import json
import base64
from pathlib import Path
from datetime import datetime
import requests
from PIL import Image
import io
import threading
import time

from deep_translator import GoogleTranslator

# Импорт из других модулей
from ..schemas.schemas import get_db, User, Folder, Project, ProjectStatus, ScenarioElementImage
from ..scripts.script_generator import generate_ad_script
from .dependencies import get_current_user, get_folder_by_id

security = HTTPBearer()

router = APIRouter(
    prefix="/script-generator",
    tags=["script-generator"],
    dependencies=[Depends(security)]
)

# Модели Pydantic для запросов и ответов
class GenerateScriptRequest(BaseModel):
    product_description: str  # Убран user_id, т.к. он будет извлекаться из JWT
    folder_id: Optional[int] = None  # ID папки, в которую нужно сохранить проект
    project_name: Optional[str] = None  # Название проекта

class GenerateImageRequest(BaseModel):
    image_description: str  # Убран user_id, т.к. он будет извлекаться из JWT

class GenerateScriptResponse(BaseModel):
    project_id: int
    status: str
    message: str

class GenerateImageResponse(BaseModel):
    project_id: int
    status: str
    message: str

class GenerateElementImageRequest(BaseModel):
    project_id: int
    element_index: Optional[int] = None  # Если None, то генерировать для всех элементов

class EditElementImageRequest(BaseModel):
    project_id: int
    element_index: int
    image_description: str

class EditImageForBlockRequest(BaseModel):
    project_id: int
    block_index: int  # Индекс блока в сценарии (1-нумерация)
    use_block_prompt: bool = True  # Если True, использовать промт из блока, иначе - использовать custom_prompt
    custom_prompt: Optional[str] = None  # Пользовательский промт, если use_block_prompt = False

from typing import List, Dict, Any

class ImageBlockData(BaseModel):
    index: int
    image_path: Optional[str] = None
    image_description: Optional[str] = None

class ImageGenerationStatus(BaseModel):
    index: int
    status: str

class ProjectStatusResponse(BaseModel):
    project_id: int
    user_id: int
    folder_id: Optional[int]
    project_name: Optional[str]
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    result_path: Optional[str] = None
    image_path: Optional[str] = None
    image_description: Optional[str] = None
    product_description: Optional[str] = None
    # New structured fields for multiple images
    image_generation_status: Optional[List[ImageGenerationStatus]] = None
    image_paths: Optional[List[ImageBlockData]] = None
    image_descriptions: Optional[List[ImageBlockData]] = None


def translate_ru_to_en(text: str) -> str:
    """
    Переводит текст с русского на английский.
    Если перевод не удался — возвращает исходный текст.
    """
    try:
        return GoogleTranslator(source="ru", target="en").translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


@router.post("/generate", response_model=GenerateScriptResponse)
async def generate_script_endpoint(
    request: GenerateScriptRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт для генерации рекламного сценария
    Принимает описание продукта, пользователь определяется из JWT
    """
    # Проверяем, что папка существует и принадлежит пользователю
    folder_id = None
    if request.folder_id:
        folder = get_folder_by_id(request.folder_id, current_user.id, db)
        folder_id = folder.id

    # Создаем запись проекта в БД со статусом "in_progress"
    project = Project(
        name=request.project_name,
        folder_id=folder_id,
        user_id=current_user.id,
        status=ProjectStatus.in_progress,
        product_description=request.product_description
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # Определяем путь для сохранения JSON файла
    user_data_dir = Path("api/users_data") / str(current_user.id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    output_file = user_data_dir / f"{current_user.id}_{project.id}_scenario.json"

    # Добавляем задачу в фон для генерации сценария
    background_tasks.add_task(
        process_script_generation,
        project.id,
        request.product_description,
        str(output_file),
        db
    )

    return GenerateScriptResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Script generation started for project {project.id}"
    )

def process_script_generation(
    project_id: int,
    product_description: str,
    output_file_path: str,
    db: Session
):
    """
    Фоновая задача для генерации сценария
    """
    try:
        # Вызываем функцию генерации сценария
        result = generate_ad_script(
            product_description=product_description,
            output_file=output_file_path
        )

        if result:
            # Обновляем статус проекта на "completed" и сохраняем путь к файлу
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = ProjectStatus.completed
                project.result_path = output_file_path
                db.commit()
        else:
            # Если произошла ошибка при генерации, ставим статус "failed"
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = ProjectStatus.failed
                db.commit()

    except Exception as e:
        # В случае ошибки обновляем статус на "failed"
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()

class GenerateImageForBlockRequest(BaseModel):
    project_id: int
    block_index: int  # Индекс блока в сценарии (1-нумерация)

@router.post("/generate_image", response_model=GenerateImageResponse)
async def generate_image_endpoint(
    request: GenerateImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт для генерации изображения
    Принимает описание изображения, пользователь определяется из JWT
    """
    # Создаем запись проекта в БД со статусом "in_progress"
    project = Project(
        user_id=current_user.id,
        status=ProjectStatus.in_progress,
        image_description=request.image_description
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # Определяем путь для сохранения PNG файла
    user_data_dir = Path("api/users_data") / str(current_user.id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    image_file = user_data_dir / f"{current_user.id}_{project.id}_image.png"

    # Добавляем задачу в фон для генерации изображения
    background_tasks.add_task(
        process_image_generation,
        project.id,
        request.image_description,
        str(image_file),
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Image generation started for project {project.id}"
    )

@router.post("/generate_image_for_block", response_model=GenerateImageResponse)
async def generate_image_for_block_endpoint(
    request: GenerateImageForBlockRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт для генерации изображения для конкретного блока сценария
    Берет промт для генерации из JSON-файла сценария
    """
    # Проверяем, что проект существует и принадлежит пользователю
    project = db.query(Project).filter(Project.id == request.project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что проект имеет сценарий
    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario JSON not found")

    # Читаем сценарий из JSON файла
    with open(project.result_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    blocks = scenario_data.get('blocks', [])

    # Проверяем, что указанный индекс блока корректен
    if request.block_index < 1 or request.block_index > len(blocks):
        raise HTTPException(status_code=400, detail="Invalid block index")

    # Получаем нужный блок
    block = blocks[request.block_index - 1]

    block_type = block.get('type')
    if block_type != 'action':
        raise HTTPException(
            status_code=400,
            detail="Image generation is allowed only for 'action' blocks"
        )

    block_content = block.get('content', {})
    description = block_content.get('description', '')
    image_description = f"Действие: {description}"

    # Обновляем статус проекта на "in_progress"
    project.status = ProjectStatus.in_progress
    db.commit()

    # Определяем путь для сохранения PNG файла
    user_data_dir = Path("api/users_data") / str(current_user.id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    image_file = user_data_dir / f"{current_user.id}_{project.id}_block_{request.block_index}_image.png"

    # Добавляем задачу в фон для генерации изображения
    background_tasks.add_task(
        process_image_generation,
        project.id,
        image_description,  # Используем промт, извлеченный из JSON-блока
        str(image_file),
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Image generation started for block {request.block_index} of project {project.id}"
    )

def process_image_generation(
    project_id: int,
    image_description: str,
    output_file_path: str,
    db: Session
):
    """
    Фоновая задача для генерации изображения
    """
    try:
        # Подключаемся к серверу заглушке для генерации изображения
        import httpx
        import asyncio
        import json

        # Асинхронная функция для генерации изображения
        async def generate_image_async():
            translated_prompt = translate_ru_to_en(image_description)
            print(f"Translated prompt: {translated_prompt}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:3339/generate_image",
                    json={"prompt": translated_prompt},
                    timeout=60.0  # таймаут 60 секунд
                )

                if response.status_code == 200:
                    result = response.json()
                    image_data = result["image"]

                    # Декодируем base64 изображение и сохраняем в файл
                    image_bytes = base64.b64decode(image_data)
                    with open(output_file_path, "wb") as f:
                        f.write(image_bytes)

                    # Обновляем статус проекта на "completed" и сохраняем путь к файлу
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.completed
                        project.image_path = output_file_path
                        # For single image generation, also update the JSON fields with a single block
                        # Extract block index from the filename if it's for a specific block
                        import re
                        match = re.search(r'block_(\d+)_', output_file_path)
                        if match:
                            block_index = int(match.group(1))
                            # Update JSON fields for block-specific image
                            image_paths = json.loads(project.image_paths) if project.image_paths else {"blocks": []}
                            existing_blocks = image_paths.get("blocks", [])
                            # Check if this block already exists in the list
                            block_exists = any(block.get("index") == block_index for block in existing_blocks)
                            if not block_exists:
                                existing_blocks.append({
                                    "index": block_index,
                                    "image_path": output_file_path
                                })
                            else:
                                # Update existing entry
                                for block in existing_blocks:
                                    if block.get("index") == block_index:
                                        block["image_path"] = output_file_path
                                        break
                            project.image_paths = json.dumps({"blocks": existing_blocks})

                            image_descriptions = json.loads(project.image_descriptions) if project.image_descriptions else {"blocks": []}
                            existing_descriptions = image_descriptions.get("blocks", [])
                            desc_exists = any(block.get("index") == block_index for block in existing_descriptions)
                            if not desc_exists:
                                existing_descriptions.append({
                                    "index": block_index,
                                    "image_description": image_description
                                })
                            else:
                                # Update existing entry
                                for block in existing_descriptions:
                                    if block.get("index") == block_index:
                                        block["image_description"] = image_description
                                        break
                            project.image_descriptions = json.dumps({"blocks": existing_descriptions})

                            image_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
                            existing_status = image_status.get("blocks", [])
                            status_exists = any(block.get("index") == block_index for block in existing_status)
                            if not status_exists:
                                existing_status.append({
                                    "index": block_index,
                                    "status": ProjectStatus.completed.value
                                })
                            else:
                                # Update existing entry
                                for block in existing_status:
                                    if block.get("index") == block_index:
                                        block["status"] = ProjectStatus.completed.value
                                        break
                            project.image_generation_status = json.dumps({"blocks": existing_status})
                        else:
                            # For non-block specific images, just update the old field
                            project.image_path = output_file_path
                        db.commit()

                    return True
                else:
                    # Если произошла ошибка при генерации, ставим статус "failed"
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.failed
                        # Update status in JSON fields too
                        import re
                        match = re.search(r'block_(\d+)_', output_file_path)
                        if match:
                            block_index = int(match.group(1))
                            image_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
                            existing_status = image_status.get("blocks", [])
                            status_exists = any(block.get("index") == block_index for block in existing_status)
                            if not status_exists:
                                existing_status.append({
                                    "index": block_index,
                                    "status": ProjectStatus.failed.value
                                })
                            else:
                                # Update existing entry
                                for block in existing_status:
                                    if block.get("index") == block_index:
                                        block["status"] = ProjectStatus.failed.value
                                        break
                            project.image_generation_status = json.dumps({"blocks": existing_status})
                        db.commit()
                    return False

        # Запускаем асинхронную функцию
        asyncio.run(generate_image_async())

    except Exception as e:
        # В случае ошибки обновляем статус на "failed"
        print(f"Error during image generation: {e}")
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()


class EditImageRequest(BaseModel):
    project_id: int
    image_description: str


@router.post("/edit_image", response_model=GenerateImageResponse)
async def edit_image_endpoint(
    request: EditImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт для редактирования изображения
    Принимает ID проекта и новое описание изображения
    """
    # Проверяем, что проект существует и принадлежит пользователю
    project = db.query(Project).filter(Project.id == request.project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что у проекта есть изображение для редактирования
    if not project.image_path or not os.path.exists(project.image_path):
        raise HTTPException(status_code=404, detail="Original image not found")

    # Обновляем статус проекта на "in_progress" и сохраняем новое описание
    project.status = ProjectStatus.in_progress
    project.image_description = request.image_description
    db.commit()

    # Определяем путь для сохранения отредактированного PNG файла
    user_data_dir = Path("api/users_data") / str(current_user.id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    image_file = user_data_dir / f"{current_user.id}_{project.id}_image.png"

    # Добавляем задачу в фон для редактирования изображения
    background_tasks.add_task(
        process_image_editing,
        project.id,
        request.image_description,
        project.image_path,  # путь к оригинальному изображению
        str(image_file),
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Image editing started for project {project.id}"
    )


def process_image_editing(
    project_id: int,
    image_description: str,
    original_image_path: str,
    output_file_path: str,
    db: Session
):
    """
    Фоновая задача для редактирования изображения
    """
    try:
        import httpx
        import asyncio
        import base64
        import json
        import re

        # Асинхронная функция для редактирования изображения
        async def edit_image_async():
            # Читаем оригинальное изображение и кодируем в base64
            with open(original_image_path, "rb") as f:
                original_image_bytes = f.read()
            original_image_base64 = base64.b64encode(original_image_bytes).decode('utf-8')

            translated_prompt = translate_ru_to_en(image_description)
            print(f"Translated prompt: {translated_prompt}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:3339/edit_image",
                    json={
                        "prompt": translated_prompt,
                        "image_base64": original_image_base64
                    },
                    timeout=180.0  # больший таймаут для редактирования
                )

                if response.status_code == 200:
                    result = response.json()
                    image_data = result["image"]

                    # Декодируем base64 изображение и сохраняем в файл
                    image_bytes = base64.b64decode(image_data)
                    with open(output_file_path, "wb") as f:
                        f.write(image_bytes)

                    # Обновляем статус проекта на "completed" и сохраняем путь к файлу
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.completed
                        project.image_path = output_file_path
                        # For edited block-specific images, update the JSON fields
                        match = re.search(r'block_(\d+)_', output_file_path)
                        if match:
                            block_index = int(match.group(1))
                            # Update JSON fields for block-specific image
                            image_paths = json.loads(project.image_paths) if project.image_paths else {"blocks": []}
                            existing_blocks = image_paths.get("blocks", [])
                            # Check if this block already exists in the list
                            block_exists = any(block.get("index") == block_index for block in existing_blocks)
                            if not block_exists:
                                existing_blocks.append({
                                    "index": block_index,
                                    "image_path": output_file_path
                                })
                            else:
                                # Update existing entry
                                for block in existing_blocks:
                                    if block.get("index") == block_index:
                                        block["image_path"] = output_file_path
                                        break
                            project.image_paths = json.dumps({"blocks": existing_blocks})

                            image_descriptions = json.loads(project.image_descriptions) if project.image_descriptions else {"blocks": []}
                            existing_descriptions = image_descriptions.get("blocks", [])
                            desc_exists = any(block.get("index") == block_index for block in existing_descriptions)
                            if not desc_exists:
                                existing_descriptions.append({
                                    "index": block_index,
                                    "image_description": image_description
                                })
                            else:
                                # Update existing entry
                                for block in existing_descriptions:
                                    if block.get("index") == block_index:
                                        block["image_description"] = image_description
                                        break
                            project.image_descriptions = json.dumps({"blocks": existing_descriptions})

                            image_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
                            existing_status = image_status.get("blocks", [])
                            status_exists = any(block.get("index") == block_index for block in existing_status)
                            if not status_exists:
                                existing_status.append({
                                    "index": block_index,
                                    "status": ProjectStatus.completed.value
                                })
                            else:
                                # Update existing entry
                                for block in existing_status:
                                    if block.get("index") == block_index:
                                        block["status"] = ProjectStatus.completed.value
                                        break
                            project.image_generation_status = json.dumps({"blocks": existing_status})
                        db.commit()

                    return True
                else:
                    # Если произошла ошибка при редактировании, ставим статус "failed"
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.failed
                        # Update status in JSON fields too
                        match = re.search(r'block_(\d+)_', output_file_path)
                        if match:
                            block_index = int(match.group(1))
                            image_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
                            existing_status = image_status.get("blocks", [])
                            status_exists = any(block.get("index") == block_index for block in existing_status)
                            if not status_exists:
                                existing_status.append({
                                    "index": block_index,
                                    "status": ProjectStatus.failed.value
                                })
                            else:
                                # Update existing entry
                                for block in existing_status:
                                    if block.get("index") == block_index:
                                        block["status"] = ProjectStatus.failed.value
                                        break
                            project.image_generation_status = json.dumps({"blocks": existing_status})
                        db.commit()
                    return False

        # Запускаем асинхронную функцию
        asyncio.run(edit_image_async())

    except Exception as e:
        # В случае ошибки обновляем статус на "failed"
        print(f"Error during image editing: {e}")
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()


@router.post("/generate_element_images", response_model=GenerateImageResponse)
async def generate_element_images_endpoint(
    request: GenerateElementImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт для генерации изображений для элементов сценария
    Принимает ID проекта и индекс элемента (или None для всех элементов)
    """
    # Проверяем, что проект существует и принадлежит пользователю
    project = db.query(Project).filter(Project.id == request.project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что проект имеет сценарий
    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario JSON not found")

    # Читаем сценарий из JSON файла
    with open(project.result_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    blocks = scenario_data.get('blocks', [])

    if request.element_index is not None:
        # Генерируем изображение только для указанного элемента
        if request.element_index < 1 or request.element_index > len(blocks):
            raise HTTPException(status_code=400, detail="Invalid element index")

        block = blocks[request.element_index - 1]
        if block.get('type') != 'action':
            raise HTTPException(
                status_code=400,
                detail="Image generation is allowed only for 'action' blocks"
            )

        target_blocks = [block]
    else:
        # Генерируем изображения только для action-блоков
        target_blocks = [b for b in blocks if b.get('type') == 'action']
        if not target_blocks:
            raise HTTPException(
                status_code=400,
                detail="No 'action' blocks found in scenario"
            )

    # Определяем директорию для изображений
    user_data_dir = Path("api/users_data") / str(current_user.id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Обновляем статус проекта на "in_progress"
    project.status = ProjectStatus.in_progress
    db.commit()

    # Добавляем задачу в фон для генерации изображений для элементов
    background_tasks.add_task(
        process_element_images_generation,
        project.id,
        request.element_index,
        target_blocks,
        str(user_data_dir),
        db
    )

    if request.element_index is not None:
        message = f"Image generation started for element {request.element_index} of project {project.id}"
    else:
        message = f"Image generation started for all elements of project {project.id}"

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=message
    )


def process_element_images_generation(
    project_id: int,
    element_index: Optional[int],
    target_blocks: list,
    user_data_dir: str,
    db: Session
):
    """
    Фоновая задача для генерации изображений для элементов сценария
    """
    try:
        import httpx
        import asyncio
        import json

        # Асинхронная функция для генерации изображений для элементов
        async def generate_element_images_async():
            success_count = 0

            # Prepare data structures to store all image paths and descriptions
            image_paths_list = []
            image_descriptions_list = []
            image_generation_status_list = []

            for block in target_blocks:
                block_index = block.get('index')
                block_type = block.get('type')

                # На всякий случай ещё раз фильтруем
                if block_type != 'action':
                    continue

                block_content = block.get('content', {})
                description = block_content.get('description', '')
                image_description = f"Действие: {description}"

                output_file_path = f"{user_data_dir}/{project_id}_{block_index}_element_image.png"

                element_image = ScenarioElementImage(
                    project_id=project_id,
                    element_index=block_index,
                    image_description=image_description,
                    status=ProjectStatus.in_progress
                )
                db.add(element_image)
                db.commit()

                try:
                    translated_prompt = translate_ru_to_en(image_description)
                    print(f"Translated prompt: {translated_prompt}")
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://127.0.0.1:3339/generate_image",
                            json={"prompt": translated_prompt},
                            timeout=60.0
                        )

                        if response.status_code == 200:
                            result = response.json()
                            image_data = result["image"]

                            # Декодируем base64 изображение и сохраняем в файл
                            image_bytes = base64.b64decode(image_data)
                            with open(output_file_path, "wb") as f:
                                f.write(image_bytes)

                            # Обновляем запись в базе данных
                            element_image.image_path = output_file_path
                            element_image.status = ProjectStatus.completed
                            db.commit()

                            # Add to our JSON data structures
                            image_paths_list.append({
                                "index": block_index,
                                "image_path": output_file_path
                            })
                            image_descriptions_list.append({
                                "index": block_index,
                                "image_description": image_description
                            })
                            image_generation_status_list.append({
                                "index": block_index,
                                "status": ProjectStatus.completed.value
                            })

                            success_count += 1
                        else:
                            # Обновляем статус элемента как failed
                            element_image.status = ProjectStatus.failed
                            db.commit()
                            # Add failed status to our JSON data structures
                            image_generation_status_list.append({
                                "index": block_index,
                                "status": ProjectStatus.failed.value
                            })
                except Exception as e:
                    # В случае ошибки обновляем статус элемента как failed
                    element_image.status = ProjectStatus.failed
                    db.commit()
                    # Add failed status to our JSON data structures
                    image_generation_status_list.append({
                        "index": block_index,
                        "status": ProjectStatus.failed.value
                    })
                    print(f"Error during image generation for element {block_index}: {e}")

            # После завершения всех генераций обновляем статус проекта и сохраняем JSON данные
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                # Проверяем, есть ли еще элементы в процессе генерации
                remaining_elements = db.query(ScenarioElementImage).filter(
                    ScenarioElementImage.project_id == project_id,
                    ScenarioElementImage.status == ProjectStatus.in_progress
                ).count()

                if remaining_elements == 0:
                    # Если все элементы обработаны, обновляем статус проекта
                    project.status = ProjectStatus.completed
                    # Сохраняем JSON данные в новые поля
                    if image_paths_list:
                        project.image_paths = json.dumps({"blocks": image_paths_list})
                    if image_descriptions_list:
                        project.image_descriptions = json.dumps({"blocks": image_descriptions_list})
                    if image_generation_status_list:
                        project.image_generation_status = json.dumps({"blocks": image_generation_status_list})
                    db.commit()

        # Запускаем асинхронную функцию
        asyncio.run(generate_element_images_async())

    except Exception as e:
        # В случае ошибки обновляем статус проекта как failed
        print(f"Error during element images generation: {e}")
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()


@router.post("/edit_element_image", response_model=GenerateImageResponse)
async def edit_element_image_endpoint(
    request: EditElementImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт для редактирования изображения конкретного элемента сценария
    Принимает ID проекта, индекс элемента и новое описание изображения
    """
    # Проверяем, что проект существует и принадлежит пользователю
    project = db.query(Project).filter(Project.id == request.project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что элемент изображения существует
    element_image = db.query(ScenarioElementImage).filter(
        ScenarioElementImage.project_id == request.project_id,
        ScenarioElementImage.element_index == request.element_index
    ).first()

    if not element_image:
        raise HTTPException(status_code=404, detail="Element image not found")

    # Проверяем, что у элемента есть изображение для редактирования
    if not element_image.image_path or not os.path.exists(element_image.image_path):
        raise HTTPException(status_code=404, detail="Original element image not found")

    # Обновляем статус элемента на "in_progress" и сохраняем новое описание
    element_image.status = ProjectStatus.in_progress
    element_image.image_description = request.image_description
    db.commit()

    # Обновляем также статус проекта как в процессе
    project.status = ProjectStatus.in_progress
    db.commit()

    # Добавляем задачу в фон для редактирования изображения элемента
    background_tasks.add_task(
        process_element_image_editing,
        element_image.id,
        request.image_description,
        element_image.image_path,  # путь к оригинальному изображению
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Element image editing started for element {request.element_index} of project {project.id}"
    )


def process_element_image_editing(
    element_image_id: int,
    image_description: str,
    original_image_path: str,
    db: Session
):
    """
    Фоновая задача для редактирования изображения конкретного элемента
    """
    try:
        import httpx
        import asyncio
        import base64

        # Асинхронная функция для редактирования изображения элемента
        async def edit_element_image_async():
            # Читаем оригинальное изображение и кодируем в base64
            with open(original_image_path, "rb") as f:
                original_image_bytes = f.read()
            original_image_base64 = base64.b64encode(original_image_bytes).decode('utf-8')

            translated_prompt = translate_ru_to_en(image_description)
            print(f"Translated prompt: {translated_prompt}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:3339/edit_image",
                    json={
                        "prompt": translated_prompt,
                        "image_base64": original_image_base64
                    },
                    timeout=180.0  # больший таймаут для редактирования
                )

                if response.status_code == 200:
                    result = response.json()
                    image_data = result["image"]

                    # Новый путь к изображению (тот же файл, просто перезаписываем)
                    element_image = db.query(ScenarioElementImage).filter(
                        ScenarioElementImage.id == element_image_id
                    ).first()

                    if element_image:
                        # Декодируем base64 изображение и сохраняем в файл
                        image_bytes = base64.b64decode(image_data)
                        with open(element_image.image_path, "wb") as f:
                            f.write(image_bytes)

                        # Обновляем статус элемента на "completed"
                        element_image.status = ProjectStatus.completed
                        db.commit()

                        # Также проверяем, все ли элементы проекта завершены
                        project = db.query(Project).filter(Project.id == element_image.project_id).first()
                        if project:
                            remaining_elements = db.query(ScenarioElementImage).filter(
                                ScenarioElementImage.project_id == element_image.project_id,
                                ScenarioElementImage.status == ProjectStatus.in_progress
                            ).count()

                            if remaining_elements == 0:
                                # Если все элементы обработаны, обновляем статус проекта
                                project.status = ProjectStatus.completed
                                db.commit()

                    return True
                else:
                    # Если произошла ошибка при редактировании, ставим статус "failed"
                    element_image = db.query(ScenarioElementImage).filter(
                        ScenarioElementImage.id == element_image_id
                    ).first()

                    if element_image:
                        element_image.status = ProjectStatus.failed
                        db.commit()

                    return False

        # Запускаем асинхронную функцию
        asyncio.run(edit_element_image_async())

    except Exception as e:
        # В случае ошибки обновляем статус элемента как failed
        print(f"Error during element image editing: {e}")
        element_image = db.query(ScenarioElementImage).filter(
            ScenarioElementImage.id == element_image_id
        ).first()
        if element_image:
            element_image.status = ProjectStatus.failed
            db.commit()


@router.post("/edit_image_for_block", response_model=GenerateImageResponse)
async def edit_image_for_block_endpoint(
    request: EditImageForBlockRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт для редактирования изображения для конкретного блока сценария
    Можно использовать промт из блока или указать свой
    """
    # Проверяем, что проект существует и принадлежит пользователю
    project = db.query(Project).filter(Project.id == request.project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что проект имеет сценарий
    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario JSON not found")

    # Читаем сценарий из JSON файла
    with open(project.result_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    blocks = scenario_data.get('blocks', [])

    # Проверяем, что указанный индекс блока корректен
    if request.block_index < 1 or request.block_index > len(blocks):
        raise HTTPException(status_code=400, detail="Invalid block index")

    # Получаем нужный блок
    block = blocks[request.block_index - 1]

    block_type = block.get('type')
    if block_type != 'action':
        raise HTTPException(
            status_code=400,
            detail="Image editing is allowed only for 'action' blocks"
        )

    block_content = block.get('content', {})

    # Определяем, какой промт использовать
    if request.use_block_prompt:
        description = block_content.get('description', '')
        image_description = f"Действие: {description}"
    else:
        if not request.custom_prompt:
            raise HTTPException(
                status_code=400,
                detail="Custom prompt is required when use_block_prompt is False"
            )
        image_description = request.custom_prompt

    # Устанавливаем путь к оригинальному изображению
    user_data_dir = Path("api/users_data") / str(current_user.id) / str(project.id)
    original_image_path = user_data_dir / f"{current_user.id}_{project.id}_block_{request.block_index}_image.png"

    # Проверяем, что изображение существует
    if not os.path.exists(original_image_path):
        raise HTTPException(status_code=404, detail="Original image not found. Generate the image first.")

    # Обновляем статус проекта на "in_progress"
    project.status = ProjectStatus.in_progress
    db.commit()

    # Определяем путь для сохранения отредактированного PNG файла
    edited_image_path = user_data_dir / f"{current_user.id}_{project.id}_block_{request.block_index}_edited_image.png"

    # Добавляем задачу в фон для редактирования изображения
    background_tasks.add_task(
        process_image_editing,
        project.id,
        image_description,
        str(original_image_path),
        str(edited_image_path),
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Image editing started for block {request.block_index} of project {project.id}"
    )


@router.get("/status/{project_id}", response_model=ProjectStatusResponse)
def get_project_status(
    project_id: int,
    db: Session = Depends(get_db)
):
    """
    Получить статус проекта по ID (включая информацию о сценарии и изображении)
    """
    import json
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Parse JSON fields if they exist
    image_generation_status = None
    image_paths = None
    image_descriptions = None

    if project.image_generation_status:
        try:
            status_data = json.loads(project.image_generation_status)
            if "blocks" in status_data:
                image_generation_status = [ImageGenerationStatus(**block) for block in status_data["blocks"]]
        except (json.JSONDecodeError, TypeError):
            pass

    if project.image_paths:
        try:
            paths_data = json.loads(project.image_paths)
            if "blocks" in paths_data:
                image_paths = [ImageBlockData(**block) for block in paths_data["blocks"]]
        except (json.JSONDecodeError, TypeError):
            pass

    if project.image_descriptions:
        try:
            descriptions_data = json.loads(project.image_descriptions)
            if "blocks" in descriptions_data:
                image_descriptions = [ImageBlockData(**block) for block in descriptions_data["blocks"]]
        except (json.JSONDecodeError, TypeError):
            pass

    return ProjectStatusResponse(
        project_id=project.id,
        user_id=project.user_id,
        folder_id=project.folder_id,
        project_name=project.name,
        status=project.status.value,
        created_at=project.created_at,
        updated_at=project.updated_at,
        result_path=project.result_path,
        image_path=project.image_path,
        image_description=project.image_description,
        product_description=project.product_description,
        image_generation_status=image_generation_status,
        image_paths=image_paths,
        image_descriptions=image_descriptions
    )

@router.get("/scenario/{project_id}")
async def get_scenario(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить сценарий для проекта по ID
    Проверяет, что проект принадлежит пользователю
    """
    # Проверяем, что проект существует и принадлежит пользователю
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    # Проверяем, что файл сценария существует
    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario file not found")

    # Читаем и возвращаем содержимое сценария
    try:
        with open(project.result_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        return scenario_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scenario file: {str(e)}")

@router.get("/images/{project_id}")
async def get_project_images(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить все изображения проекта
    Проверяет, что проект принадлежит пользователю
    """
    # Проверяем, что проект существует и принадлежит пользователю
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    # Собираем информацию о всех изображениях проекта
    images_info = []

    # Добавляем основное изображение проекта, если оно существует
    if project.image_path and os.path.exists(project.image_path):
        images_info.append({
            "type": "project_main",
            "path": project.image_path,
            "description": project.image_description,
            "created_at": project.created_at
        })

    # Добавляем изображения элементов сценария
    scenario_images = db.query(ScenarioElementImage).filter(
        ScenarioElementImage.project_id == project.id
    ).all()

    for img in scenario_images:
        if img.image_path and os.path.exists(img.image_path):
            images_info.append({
                "type": "element_image",
                "element_index": img.element_index,
                "path": img.image_path,
                "description": img.image_description,
                "status": img.status.value,
                "created_at": img.created_at,
                "updated_at": img.updated_at
            })

    import json

    # Parse JSON fields if they exist for structured format
    parsed_image_generation_status = None
    parsed_image_paths = None
    parsed_image_descriptions = None

    if project.image_generation_status:
        try:
            status_data = json.loads(project.image_generation_status)
            if "blocks" in status_data:
                parsed_image_generation_status = [ImageGenerationStatus(**block) for block in status_data["blocks"]]
        except (json.JSONDecodeError, TypeError):
            pass

    if project.image_paths:
        try:
            paths_data = json.loads(project.image_paths)
            if "blocks" in paths_data:
                parsed_image_paths = [ImageBlockData(**block) for block in paths_data["blocks"]]
        except (json.JSONDecodeError, TypeError):
            pass

    if project.image_descriptions:
        try:
            descriptions_data = json.loads(project.image_descriptions)
            if "blocks" in descriptions_data:
                parsed_image_descriptions = [ImageBlockData(**block) for block in descriptions_data["blocks"]]
        except (json.JSONDecodeError, TypeError):
            pass

    # Return image information in both old format and new structured format
    return {
        "project_id": project.id,
        "project_name": project.name,
        "images_count": len(images_info),
        "images": images_info,
        "image_generation_status": parsed_image_generation_status,  # Structured format
        "image_paths": parsed_image_paths,  # Structured format
        "image_descriptions": parsed_image_descriptions  # Structured format
    }

@router.get("/projects")
async def get_user_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить все проекты пользователя
    """
    import json
    # Получаем все проекты пользователя
    projects = db.query(Project).filter(Project.user_id == current_user.id).all()

    projects_info = []
    for project in projects:
        # Подсчитываем количество изображений у проекта
        images_count = 0
        if project.image_path and os.path.exists(project.image_path):
            images_count += 1

        element_images_count = db.query(ScenarioElementImage).filter(
            ScenarioElementImage.project_id == project.id
        ).count()
        images_count += element_images_count

        # Parse JSON fields if they exist for structured format
        parsed_image_generation_status = None
        parsed_image_paths = None
        parsed_image_descriptions = None

        if project.image_generation_status:
            try:
                status_data = json.loads(project.image_generation_status)
                if "blocks" in status_data:
                    parsed_image_generation_status = [ImageGenerationStatus(**block) for block in status_data["blocks"]]
            except (json.JSONDecodeError, TypeError):
                pass

        if project.image_paths:
            try:
                paths_data = json.loads(project.image_paths)
                if "blocks" in paths_data:
                    parsed_image_paths = [ImageBlockData(**block) for block in paths_data["blocks"]]
            except (json.JSONDecodeError, TypeError):
                pass

        if project.image_descriptions:
            try:
                descriptions_data = json.loads(project.image_descriptions)
                if "blocks" in descriptions_data:
                    parsed_image_descriptions = [ImageBlockData(**block) for block in descriptions_data["blocks"]]
            except (json.JSONDecodeError, TypeError):
                pass

        projects_info.append({
            "id": project.id,
            "name": project.name,
            "folder_id": project.folder_id,
            "status": project.status.value,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "has_scenario": project.result_path is not None and os.path.exists(project.result_path),
            "has_images": images_count > 0,
            "images_count": images_count,
            "result_path": project.result_path,
            "image_path": project.image_path,
            "product_description": project.product_description,
            "image_generation_status": parsed_image_generation_status,  # Structured format
            "image_paths": parsed_image_paths,  # Structured format
            "image_descriptions": parsed_image_descriptions  # Structured format
        })

    return {
        "user_id": current_user.id,
        "username": current_user.login,
        "projects_count": len(projects_info),
        "projects": projects_info
    }