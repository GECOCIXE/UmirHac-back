from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Set
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
import mimetypes
import base64
from pathlib import Path

from deep_translator import GoogleTranslator

from ..schemas.schemas import get_db, User, Folder, Project, ProjectStatus, ScenarioElementImage
from ..scripts.script_generator import generate_ad_script
from .dependencies import get_current_user, get_folder_by_id

security = HTTPBearer()

router = APIRouter(
    prefix="/script-generator",
    tags=["script-generator"],
    dependencies=[Depends(security)]
)

class BlocksImagesRequest(BaseModel):
    block_indices: List[int]

class ScenarioBlock(BaseModel):
    type: str
    content: Dict[str, Any]
    formatting: Dict[str, Any]
    index: Optional[int] = None 

class EditImageForBlockRequest(BaseModel):
    project_id: int
    block_index: int  
    use_block_prompt: bool = True
    custom_prompt: Optional[str] = None

class ScenarioBlockCreate(BaseModel):
    """
    Модель для создания нового блока сценария.
    index не передаём — его выдаёт бэкенд.
    """
    type: str
    content: Dict[str, Any]
    formatting: Dict[str, Any]


class ScenarioBlockUpdate(BaseModel):
    """
    Модель для частичного обновления блока.
    Можно прислать только то, что меняется.
    """
    type: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    formatting: Optional[Dict[str, Any]] = None


class ScenarioUpdateRequest(BaseModel):
    product_description: Optional[str] = None
    original_blocks_count: Optional[int] = None
    blocks: List[ScenarioBlock]

# Модели Pydantic для запросов и ответов
class GenerateScriptRequest(BaseModel):
    product_description: str  
    folder_id: Optional[int] = None 
    project_name: Optional[str] = None 

class GenerateImageRequest(BaseModel):
    image_description: str  

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
    element_index: Optional[int] = None 

class EditElementImageRequest(BaseModel):
    project_id: int
    element_index: int
    image_description: str

class EditImageForBlockRequest(BaseModel):
    project_id: int
    block_index: int
    use_block_prompt: bool = True  
    custom_prompt: Optional[str] = None 


class ImagePathData(BaseModel):
    index: int
    image_path: Optional[str] = None

class ImageDescriptionData(BaseModel):
    index: int
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
    image_generation_status: Optional[List[ImageGenerationStatus]] = None
    image_paths: Optional[List[ImagePathData]] = None
    image_descriptions: Optional[List[ImageDescriptionData]] = None
    
class ScenarioReorderRequest(BaseModel):
    new_order: List[int]
    
class EditImageRequest(BaseModel):
    project_id: int
    image_description: str

class GenerateImageForBlockRequest(BaseModel):
    project_id: int
    block_index: int

def _remap_indices_for_images(project: Project, index_map: Dict[int, int], db: Session):
    """
    Перекидывает индексы картинок и статусов по произвольному отображению:
    old_index -> new_index.
    Используется при полном reorder блоков.
    """
    if not index_map:
        return

    # 1. ScenarioElementImage
    images = db.query(ScenarioElementImage).filter(
        ScenarioElementImage.project_id == project.id
    ).all()

    for img in images:
        old_idx = img.element_index
        if old_idx in index_map:
            img.element_index = index_map[old_idx]

    # 2. JSON-поля проекта
    def _remap_json(field_value: Optional[str]) -> Optional[str]:
        if not field_value:
            return field_value
        try:
            data = json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            return field_value

        blocks = data.get("blocks")
        if not isinstance(blocks, list):
            return field_value

        for b in blocks:
            idx = b.get("index")
            if isinstance(idx, int) and idx in index_map:
                b["index"] = index_map[idx]

        return json.dumps(data, ensure_ascii=False)

    project.image_paths = _remap_json(project.image_paths)
    project.image_descriptions = _remap_json(project.image_descriptions)
    project.image_generation_status = _remap_json(project.image_generation_status)

def _shift_indices_for_images(project: Project, start_index: int, delta: int, db: Session):
    """
    Смещает индексы для всех блоков >= start_index на delta
    и в БД, и в JSON-полях проекта.
    Используется при вставке/удалении блоков.
    """
    if delta == 0:
        return

    # 1. ScenarioElementImage
    images = db.query(ScenarioElementImage).filter(
        ScenarioElementImage.project_id == project.id,
        ScenarioElementImage.element_index >= start_index,
    ).all()

    for img in images:
        img.element_index = img.element_index + delta

    # 2. JSON-поля проекта
    def _shift_json(field_value: Optional[str]) -> Optional[str]:
        if not field_value:
            return field_value
        try:
            data = json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            return field_value

        blocks = data.get("blocks")
        if not isinstance(blocks, list):
            return field_value

        for b in blocks:
            idx = b.get("index")
            if isinstance(idx, int) and idx >= start_index:
                b["index"] = idx + delta

        return json.dumps(data, ensure_ascii=False)

    project.image_paths = _shift_json(project.image_paths)
    project.image_descriptions = _shift_json(project.image_descriptions)
    project.image_generation_status = _shift_json(project.image_generation_status)


def _clear_images_for_block(project: Project, block_index: int, db: Session):
    """
    Удаляет все данные по изображениям для блока с данным index:
    - записи в ScenarioElementImage
    - записи в JSON-полях Project.image_paths / image_descriptions / image_generation_status
    """
    # 1. Удаляем записи ScenarioElementImage и сами файлы (если есть)
    element_images = db.query(ScenarioElementImage).filter(
        ScenarioElementImage.project_id == project.id,
        ScenarioElementImage.element_index == block_index
    ).all()

    for img in element_images:
        if img.image_path and os.path.exists(img.image_path):
            try:
                os.remove(img.image_path)
            except OSError:
                # Не критично, если файл не удалился
                pass
        db.delete(img)

    # 2. Удаляем блок из JSON-полей по index
    def _remove_from_json(field_value: Optional[str]) -> Optional[str]:
        if not field_value:
            return field_value
        try:
            data = json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            return field_value

        blocks = data.get("blocks")
        if not isinstance(blocks, list):
            return field_value

        data["blocks"] = [b for b in blocks if b.get("index") != block_index]
        return json.dumps(data, ensure_ascii=False)

    project.image_paths = _remove_from_json(project.image_paths)
    project.image_descriptions = _remove_from_json(project.image_descriptions)
    project.image_generation_status = _remove_from_json(project.image_generation_status)


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

    # Do not change the overall project status, only track image generation status
    # Create or update the image generation status separately
    
    current_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
    existing_blocks = current_status.get("blocks", [])

    # Check if this block already exists in the status list
    block_exists = False
    for block in existing_blocks:
        if block.get("index") == request.block_index:
            block["status"] = ProjectStatus.in_progress.value
            block_exists = True
            break

    if not block_exists:
        existing_blocks.append({
            "index": request.block_index,
            "status": ProjectStatus.in_progress.value
        })

    project.image_generation_status = json.dumps({"blocks": existing_blocks})
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
        

        # Асинхронная функция для генерации изображения
        async def generate_image_async():
            translated_prompt = translate_ru_to_en(image_description)
            print(f"Translated prompt: {translated_prompt}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:3339/generate_image",
                    json={"prompt": translated_prompt},
                    timeout=6000.0  # таймаут 60 секунд
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
                        # Don't change the main project status, only update image generation status
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
            # Don't change the main project status, only update image generation status
            # For general image generation (not block-specific), we still use JSON fields
            
            current_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
            existing_blocks = current_status.get("blocks", [])

            # Since this is a general image, we may not know the block index from the filename
            # Mark any in-progress blocks as failed
            for block in existing_blocks:
                if block.get("status") == ProjectStatus.in_progress.value:
                    block["status"] = ProjectStatus.failed.value

            project.image_generation_status = json.dumps({"blocks": existing_blocks})
            db.commit()

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
                    timeout=6000.0  # больший таймаут для редактирования
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
                        # Don't change the main project status, only update image generation status
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
            # Don't change the main project status, only update image generation status
            # For general image editing (not block-specific), we still use JSON fields
            
            current_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
            existing_blocks = current_status.get("blocks", [])

            # Mark any in-progress blocks as failed
            for block in existing_blocks:
                if block.get("status") == ProjectStatus.in_progress.value:
                    block["status"] = ProjectStatus.failed.value

            project.image_generation_status = json.dumps({"blocks": existing_blocks})
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

    # Do not change the overall project status, only track image generation status
    # Create or update the image generation status separately
    
    current_status = json.loads(project.image_generation_status) if project.image_generation_status else {"blocks": []}
    existing_blocks = current_status.get("blocks", [])

    # Check if this block already exists in the status list
    block_exists = False
    for block in existing_blocks:
        if block.get("index") == request.block_index:
            block["status"] = ProjectStatus.in_progress.value
            block_exists = True
            break

    if not block_exists:
        existing_blocks.append({
            "index": request.block_index,
            "status": ProjectStatus.in_progress.value
        })

    project.image_generation_status = json.dumps({"blocks": existing_blocks})
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
                # Create ImagePathData objects only with index and image_path
                image_paths = []
                for block in paths_data["blocks"]:
                    image_paths.append(ImagePathData(index=block["index"], image_path=block.get("image_path")))
        except (json.JSONDecodeError, TypeError):
            pass

    if project.image_descriptions:
        try:
            descriptions_data = json.loads(project.image_descriptions)
            if "blocks" in descriptions_data:
                # Create ImageDescriptionData objects only with index and image_description
                image_descriptions = []
                for block in descriptions_data["blocks"]:
                    image_descriptions.append(ImageDescriptionData(index=block["index"], image_description=block.get("image_description")))
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
                # Create ImagePathData objects only with index and image_path
                parsed_image_paths = []
                for block in paths_data["blocks"]:
                    parsed_image_paths.append(ImagePathData(index=block["index"], image_path=block.get("image_path")))
        except (json.JSONDecodeError, TypeError):
            pass

    if project.image_descriptions:
        try:
            descriptions_data = json.loads(project.image_descriptions)
            if "blocks" in descriptions_data:
                # Create ImageDescriptionData objects only with index and image_description
                parsed_image_descriptions = []
                for block in descriptions_data["blocks"]:
                    parsed_image_descriptions.append(ImageDescriptionData(index=block["index"], image_description=block.get("image_description")))
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
                    # Create ImagePathData objects only with index and image_path
                    parsed_image_paths = []
                    for block in paths_data["blocks"]:
                        parsed_image_paths.append(ImagePathData(index=block["index"], image_path=block.get("image_path")))
            except (json.JSONDecodeError, TypeError):
                pass

        if project.image_descriptions:
            try:
                descriptions_data = json.loads(project.image_descriptions)
                if "blocks" in descriptions_data:
                    # Create ImageDescriptionData objects only with index and image_description
                    parsed_image_descriptions = []
                    for block in descriptions_data["blocks"]:
                        parsed_image_descriptions.append(ImageDescriptionData(index=block["index"], image_description=block.get("image_description")))
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
    
@router.put("/scenario/{project_id}")
async def update_scenario(
    project_id: int,
    request: ScenarioUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Обновить сценарий проекта:
    - редактирование / добавление блоков
    - синхронизация ScenarioElementImage и JSON-полей для картинок
    """
    # 1. Проверяем проект и файл сценария
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario file not found")

    # 2. Читаем текущий JSON сценария
    try:
        with open(project.result_path, "r", encoding="utf-8") as f:
            current_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scenario file: {str(e)}")

    old_blocks = current_data.get("blocks") or []

    # Мапа старых блоков по index
    old_by_index: Dict[int, Dict[str, Any]] = {}
    for b in old_blocks:
        idx = b.get("index")
        if isinstance(idx, int):
            old_by_index[idx] = b
    old_indices: Set[int] = set(old_by_index.keys())

    # 3. Собираем новые блоки из запроса
    new_blocks: List[Dict[str, Any]] = []
    new_by_index: Dict[int, Dict[str, Any]] = {}

    # стартовый max_index для генерации индексов новым блокам
    max_index = max(old_indices) if old_indices else 0

    for block_model in request.blocks:
        block_dict = block_model.dict()
        idx = block_dict.get("index")

        # Новый блок — index не передан
        if idx is None:
            max_index += 1
            idx = max_index
            block_dict["index"] = idx
        elif not isinstance(idx, int):
            raise HTTPException(status_code=400, detail="Block index must be integer")

        if idx in new_by_index:
            raise HTTPException(status_code=400, detail=f"Duplicate block index {idx} in request")

        new_by_index[idx] = block_dict
        new_blocks.append(block_dict)

    new_indices: Set[int] = set(new_by_index.keys())

    # 4. Находим удалённые и изменённые блоки
    common_indices = old_indices & new_indices

    def normalize_block(b: Dict[str, Any]) -> Dict[str, Any]:
        # Сравниваем только то, что реально важно для смысла блока
        return {
            "type": b.get("type"),
            "content": b.get("content"),
            "formatting": b.get("formatting"),
        }

    modified_indices: Set[int] = set()
    for idx in common_indices:
        if normalize_block(old_by_index[idx]) != normalize_block(new_by_index[idx]):
            modified_indices.add(idx)

    # Блоки, которые исчезли из сценария
    deleted_indices: Set[int] = old_indices - new_indices

    # Для этих индексов удаляем/обнуляем картинки и статусы
    indices_to_drop_from_images: Set[int] = deleted_indices | modified_indices

    # Индексы, для которых картинки валидны и их можно оставить
    valid_indices_for_images: Set[int] = new_indices - modified_indices

    # 5. Синхронизация таблицы ScenarioElementImage
    element_images = (
        db.query(ScenarioElementImage)
        .filter(ScenarioElementImage.project_id == project.id)
        .all()
    )

    for img in element_images:
        # Если блока больше нет в сценарии или его содержание изменилось —
        # удаляем запись и при желании сам файл
        if img.element_index not in valid_indices_for_images:
            if img.image_path and os.path.exists(img.image_path):
                try:
                    os.remove(img.image_path)
                except OSError:
                    pass
            db.delete(img)

    # 6. Синхронизация JSON-полей с картинками/статусами в Project
    def sync_json_field(field_value: Optional[str]) -> Optional[str]:
        """
        Оставляем в JSON только те блоки, чьи index есть в новом сценарии
        и блок не был изменён (modified_indices уже исключены).
        """
        if not field_value:
            return field_value
        try:
            data = json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            return field_value

        blocks = data.get("blocks")
        if not isinstance(blocks, list):
            return field_value

        filtered_blocks = [
            b for b in blocks
            if isinstance(b.get("index"), int)
            and b["index"] in valid_indices_for_images
        ]
        data["blocks"] = filtered_blocks
        return json.dumps(data, ensure_ascii=False)

    project.image_paths = sync_json_field(project.image_paths)
    project.image_descriptions = sync_json_field(project.image_descriptions)
    project.image_generation_status = sync_json_field(project.image_generation_status)

    db.commit()

    # 7. Обновляем сам JSON сценария
    incoming = request.dict(exclude_unset=True)

    product_description = incoming.get(
        "product_description",
        current_data.get("product_description", "")
    )

    original_blocks_count = incoming.get(
        "original_blocks_count",
        current_data.get("original_blocks_count")
    )

    final_blocks_count = len(new_blocks)
    if original_blocks_count is None:
        # если вообще нет значения — считаем исходным текущий финальный
        original_blocks_count = current_data.get("original_blocks_count", final_blocks_count)

    new_data = dict(current_data)
    new_data["product_description"] = product_description
    new_data["original_blocks_count"] = original_blocks_count
    new_data["final_blocks_count"] = final_blocks_count
    new_data["blocks"] = new_blocks

    try:
        with open(project.result_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing scenario file: {str(e)}")

    return new_data

@router.post("/scenario/{project_id}/blocks")
async def add_scenario_block(
    project_id: int,
    block_data: ScenarioBlockCreate,
    position: Optional[int] = Query(
        None,
        ge=0,
        description=(
            "Позиция вставки в массиве blocks (0-based). "
            "Если не указана или больше длины массива — блок добавится в конец."
        ),
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Добавить новый блок в сценарий.
    - index для блока определяется по позиции: всегда последовательные 1..N
    - все блоки ПОСЛЕ точки вставки сдвигаются (и их index, и индексы картинок).
    """
    # 1. Проверяем проект
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    # 2. Проверяем наличие сценария
    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario file not found")

    # 3. Читаем сценарий
    try:
        with open(project.result_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scenario file: {str(e)}")

    blocks = scenario_data.get("blocks") or []
    n_before = len(blocks)

    # 4. Определяем позицию вставки (0-based)
    if position is None or position > n_before:
        insert_pos = n_before
    else:
        insert_pos = position

    # Будущий index нового блока: insert_pos (0-based) -> index = insert_pos + 1
    new_index = insert_pos + 1

    # 5. Сдвигаем индексы картинок/статусов для блоков с index >= new_index
    _shift_indices_for_images(project, start_index=new_index, delta=1, db=db)

    # 6. Сдвигаем индексы существующих блоков в сценарии
    for b in blocks:
        idx = b.get("index")
        if isinstance(idx, int) and idx >= new_index:
            b["index"] = idx + 1

    # 7. Создаём новый блок
    new_block = {
        "type": block_data.type,
        "content": block_data.content,
        "formatting": block_data.formatting,
        "index": new_index,
    }

    blocks.insert(insert_pos, new_block)
    scenario_data["blocks"] = blocks

    # 8. Обновляем счётчики
    scenario_data["final_blocks_count"] = len(blocks)
    if "original_blocks_count" not in scenario_data:
        scenario_data["original_blocks_count"] = len(blocks)

    # 9. Сохраняем сценарий
    try:
        with open(project.result_path, "w", encoding="utf-8") as f:
            json.dump(scenario_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing scenario file: {str(e)}")

    # 10. Обновим updated_at и закоммитим изменения проекта (JSON-поля уже изменены)
    project.updated_at = datetime.utcnow()
    db.add(project)
    db.commit()

    return {
        "added_block": new_block,
        "scenario": scenario_data
    }


@router.patch("/scenario/{project_id}/blocks/{block_index}")
async def update_scenario_block(
    project_id: int,
    block_index: int,
    block_update: ScenarioBlockUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Частично обновить блок сценария по его index.
    - индексы блоков НЕ меняются
    - если меняется содержимое блока типа action (или тип с action на другой),
      то очищаются все связанные изображения и статусы, чтобы потом вызвать новый generate.
    """
    # 1. Проверяем проект
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario file not found")

    # 2. Читаем сценарий
    try:
        with open(project.result_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scenario file: {str(e)}")

    blocks = scenario_data.get("blocks") or []

    # 3. Ищем блок
    target_block = None
    for b in blocks:
        if b.get("index") == block_index:
            target_block = b
            break

    if not target_block:
        raise HTTPException(status_code=404, detail="Block with given index not found")

    old_type = target_block.get("type")
    old_content = target_block.get("content")

    # 4. Применяем изменения (partial update)
    update_data = block_update.dict(exclude_unset=True)

    if "type" in update_data:
        target_block["type"] = update_data["type"]
    if "content" in update_data:
        target_block["content"] = update_data["content"]
    if "formatting" in update_data:
        target_block["formatting"] = update_data["formatting"]

    new_type = target_block.get("type")
    new_content = target_block.get("content")

    # 5. Решаем, нужно ли сбрасывать картинки
    should_clear_images = False

    if old_type == "action":
        # Если сменился тип (action -> что-то ещё) или изменился контент блока
        if new_type != "action" or old_content != new_content:
            should_clear_images = True

    if should_clear_images:
        _clear_images_for_block(project, block_index, db)
        # После этого фронт должен вызвать новый generate для этого блока.

    scenario_data["blocks"] = blocks
    scenario_data["final_blocks_count"] = len(blocks)
    if "original_blocks_count" not in scenario_data:
        scenario_data["original_blocks_count"] = len(blocks)

    # 6. Сохраняем сценарий
    try:
        with open(project.result_path, "w", encoding="utf-8") as f:
            json.dump(scenario_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing scenario file: {str(e)}")

    project.updated_at = datetime.utcnow()
    db.add(project)
    db.commit()

    return {
        "updated_block": target_block,
        "scenario": scenario_data
    }
    
@router.post("/scenario/{project_id}/blocks/reorder")
async def reorder_scenario_blocks(
    project_id: int,
    reorder_request: ScenarioReorderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Полностью переупорядочить блоки сценария.

    Тело запроса:
    {
      "new_order": [3, 1, 2, 4, ...]  # старые index блоков в новом порядке
    }

    Логика:
    - блоки в JSON переставляются согласно new_order
    - индексы блоков перенумеровываются 1..N в новом порядке
    - индексы картинок/статусов ремапятся по той же схеме (old_index -> new_index)
    - сами изображения НЕ удаляются и НЕ регенерируются
    """
    # 1. Проверяем проект
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario file not found")

    # 2. Читаем сценарий
    try:
        with open(project.result_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scenario file: {str(e)}")

    blocks = scenario_data.get("blocks") or []
    if not blocks:
        raise HTTPException(status_code=400, detail="Scenario has no blocks to reorder")

    # Список текущих индексов
    existing_indices = [b.get("index") for b in blocks if isinstance(b.get("index"), int)]
    if len(existing_indices) != len(blocks):
        raise HTTPException(status_code=500, detail="Some scenario blocks have no valid index")

    existing_set = set(existing_indices)
    new_order = reorder_request.new_order

    # 3. Валидация new_order
    if len(new_order) != len(existing_indices):
        raise HTTPException(
            status_code=400,
            detail="new_order length must match number of blocks"
        )
    if set(new_order) != existing_set:
        raise HTTPException(
            status_code=400,
            detail="new_order must be a permutation of existing block indices"
        )

    # 4. Строим мапу old_index -> block
    old_by_index: Dict[int, Dict[str, Any]] = {b["index"]: b for b in blocks}

    # 5. Собираем блоки в новом порядке
    new_blocks: List[Dict[str, Any]] = []
    index_map: Dict[int, int] = {}  # old_index -> new_index

    for i, old_idx in enumerate(new_order):
        block = old_by_index[old_idx]
        new_index = i + 1  # индексы снова подряд 1..N
        index_map[old_idx] = new_index

        # создаём новый dict, чтобы не запутаться со старыми ссылками
        new_block = dict(block)
        new_block["index"] = new_index
        new_blocks.append(new_block)

    # 6. Обновляем индексы картинок/статусов по той же мапе
    _remap_indices_for_images(project, index_map, db)

    # 7. Обновляем сценарий
    scenario_data["blocks"] = new_blocks
    scenario_data["final_blocks_count"] = len(new_blocks)
    # original_blocks_count не трогаем — reorder не меняет количество блоков

    # 8. Сохраняем сценарий
    try:
        with open(project.result_path, "w", encoding="utf-8") as f:
            json.dump(scenario_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing scenario file: {str(e)}")

    project.updated_at = datetime.utcnow()
    db.add(project)
    db.commit()

    return {
        "index_map": index_map,   # на всякий случай фронту может пригодиться
        "scenario": scenario_data
    }

@router.delete("/scenario/{project_id}/blocks/{block_index}")
async def delete_scenario_block(
    project_id: int,
    block_index: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Удалить блок сценария по его index.
    - для удалённого блока полностью чистятся данные по картинкам
    - все блоки с index > block_index сдвигаются на -1
      (и в сценарии, и в БД/JSON по картинкам).
    """
    # 1. Проверяем проект
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario file not found")

    # 2. Читаем сценарий
    try:
        with open(project.result_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scenario file: {str(e)}")

    blocks = scenario_data.get("blocks") or []
    original_len = len(blocks)

    # 3. Ищем позицию блока в массиве по его index
    delete_pos = None
    for i, b in enumerate(blocks):
        if b.get("index") == block_index:
            delete_pos = i
            break

    if delete_pos is None:
        raise HTTPException(status_code=404, detail="Block with given index not found")

    # Удаляем блок из массива
    blocks.pop(delete_pos)

    # 4. Чистим данные по картинкам для удалённого блока
    _clear_images_for_block(project, block_index, db)

    # 5. Сдвигаем индексы картинок/статусов для всех блоков, которые были после
    # (у них старый index > block_index)
    _shift_indices_for_images(project, start_index=block_index + 1, delta=-1, db=db)

    # 6. Сдвигаем индексы в самом сценарии
    for b in blocks:
        idx = b.get("index")
        if isinstance(idx, int) and idx > block_index:
            b["index"] = idx - 1

    scenario_data["blocks"] = blocks
    scenario_data["final_blocks_count"] = len(blocks)

    if "original_blocks_count" not in scenario_data:
        # если поля не было, считаем исходным количество до удаления
        scenario_data["original_blocks_count"] = original_len

    # 7. Сохраняем сценарий
    try:
        with open(project.result_path, "w", encoding="utf-8") as f:
            json.dump(scenario_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing scenario file: {str(e)}")

    project.updated_at = datetime.utcnow()
    db.add(project)
    db.commit()

    return {
        "deleted_index": block_index,
        "scenario": scenario_data
    }

@router.post("/scenario/{project_id}/blocks/images")
async def get_images_for_blocks(
    project_id: int,
    request: BlocksImagesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Вернуть КАРТИНКИ для нескольких блоков сразу.
    Источники:
    1) scenario_element_images (новая схема)
    2) project.image_paths (старая схема: JSON с путями до файлов)

    Формат ответа:
    {
      "project_id": 1,
      "results": [
        {
          "block_index": 2,
          "images": [
            {
              "image_id": 10,               # может быть None, если из JSON
              "mime_type": "image/png",
              "data_base64": "iVBORw0KGgoAAA..."
            }
          ]
        },
        ...
      ]
    }
    """
    # 1. Проверяем проект
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id,
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    indices = sorted(set(request.block_indices or []))
    if not indices:
        return {"project_id": project_id, "results": []}

    # Подготовим структуру для результата
    grouped: Dict[int, List[dict]] = {idx: [] for idx in indices}

    # Базовая директория проекта (чтобы работать с относительными путями)
    base_dir = Path(".")  # можно заменить на Path(__file__).resolve().parents[2] при желании

    # --- 2. Сначала пробуем взять картинки из ScenarioElementImage (новая схема) ---
    images = (
        db.query(ScenarioElementImage)
        .filter(
            ScenarioElementImage.project_id == project.id,
            ScenarioElementImage.element_index.in_(indices),
        )
        .all()
    )

    seen_paths: Set[str] = set()  # чтобы не дублировать одну и ту же картинку

    for img in images:
        if not img.image_path:
            continue

        img_path = Path(img.image_path)
        if not img_path.is_absolute():
            img_path = base_dir / img_path

        if not img_path.exists():
            continue

        try:
            with open(img_path, "rb") as f:
                raw = f.read()
        except OSError:
            continue

        b64 = base64.b64encode(raw).decode("ascii")
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        norm_path = str(img_path.resolve())
        seen_paths.add(norm_path)

        grouped.setdefault(img.element_index, []).append(
            {
                "image_id": img.id,
                "mime_type": mime_type,
                "data_base64": b64,
            }
        )

    # --- 3. Дополняем из project.image_paths (старая схема) ---
    if project.image_paths:
        try:
            paths_data = json.loads(project.image_paths)
            blocks_list = paths_data.get("blocks") or []
        except (json.JSONDecodeError, TypeError):
            blocks_list = []

        for entry in blocks_list:
            idx = entry.get("index")
            rel_path = entry.get("image_path")
            if not isinstance(idx, int) or idx not in grouped:
                continue
            if not rel_path:
                continue

            img_path = Path(rel_path)
            if not img_path.is_absolute():
                img_path = base_dir / img_path

            if not img_path.exists():
                continue

            norm_path = str(img_path.resolve())
            # Если уже брали эту картинку из ScenarioElementImage — пропускаем
            if norm_path in seen_paths:
                continue

            try:
                with open(img_path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue

            b64 = base64.b64encode(raw).decode("ascii")
            mime_type, _ = mimetypes.guess_type(str(img_path))
            if mime_type is None:
                mime_type = "application/octet-stream"

            seen_paths.add(norm_path)

            grouped[idx].append(
                {
                    "image_id": None,  # из JSON, без отдельной записи в таблице
                    "mime_type": mime_type,
                    "data_base64": b64,
                }
            )

    # --- 4. Формируем ответ ---
    results = []
    for idx in indices:
        results.append(
            {
                "block_index": idx,
                "images": grouped.get(idx, []),
            }
        )

    return {
        "project_id": project_id,
        "results": results,
    }
